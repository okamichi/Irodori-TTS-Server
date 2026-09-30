"""Bounded, per-runtime reference caches. Called inside the runtime inference lock."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from irodori_tts.model import patch_sequence_with_mask


def file_signature(path: str) -> tuple:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return (str(resolved), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class TensorCache:
    def __init__(self, entries: int, max_bytes: int):
        self.entries = entries
        self.max_bytes = max_bytes
        self.bytes = 0
        self.values: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()

    def get(self, key: tuple) -> Any:
        value = self.values.get(key)
        if value is None:
            return None
        self.values.move_to_end(key)
        return value[0]

    def put(self, key: tuple, tensors: tuple, messages: tuple = ()) -> None:
        size = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        if self.entries == 0 or size > self.max_bytes:
            return
        old = self.values.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        while self.values and (
            len(self.values) >= self.entries or self.bytes + size > self.max_bytes
        ):
            _, (_, removed_size) = self.values.popitem(last=False)
            self.bytes -= removed_size
        self.values[key] = ((tensors, messages), size)
        self.bytes += size

    def clear(self) -> None:
        self.values.clear()
        self.bytes = 0


class ReferenceCache:
    def __init__(self, runtime: Any, entries: int, max_bytes: int):
        self.runtime = runtime
        # One budget shared by latent and speaker entries, stored on CPU.
        self.cache = TensorCache(entries, max_bytes)
        self.load_latent = runtime._load_reference_latent
        self.load_embedding = runtime._load_speaker_embedding_condition

    def key(self, req: Any) -> tuple | None:
        if req.no_ref or not self.runtime.model_cfg.use_speaker_condition_resolved:
            return None
        wavs = ([req.ref_wav] if req.ref_wav else []) + list(req.ref_wavs or [])
        latents = ([req.ref_latent] if req.ref_latent else []) + list(req.ref_latents or [])
        # Leave validation and errors to the upstream loader; never let a hit hide them.
        if (req.ref_wav and req.ref_wavs) or (req.ref_latent and req.ref_latents):
            return None
        if bool(wavs) == bool(latents):
            return None
        if wavs and not self.runtime.codec.deterministic_encode:
            return None
        return (
            "wav" if wavs else "latent",
            tuple(file_signature(p) for p in (wavs or latents)),
            req.max_ref_seconds,
            self.runtime.default_max_ref_seconds,
            req.ref_normalize_db,
            req.ref_ensure_max,
        )

    def latent(self, *, req: Any, batch_size: int, messages: list) -> tuple:
        key = self.key(req)
        if key is None:
            return self.load_latent(req=req, batch_size=batch_size, messages=messages)
        key = ("latent", key)
        cached = self.cache.get(key)
        if cached is None:
            start = len(messages)
            tensors = self.load_latent(req=req, batch_size=1, messages=messages)
            tensors = tuple(t.detach().cpu().clone() if t is not None else None for t in tensors)
            self.cache.put(key, tensors, tuple(messages[start:]))
        else:
            tensors, warnings = cached
            messages.extend(warnings)
            messages.append("info: reference latent cache hit.")
        dtype = next(self.runtime.model.parameters()).dtype
        return tuple(
            t.to(device=self.runtime.model_device, dtype=dtype if i == 0 else torch.bool)
            .expand(batch_size, *t.shape[1:])
            .clone()
            if t is not None
            else None
            for i, t in enumerate(tensors)
        )

    def speaker(self, *, req: Any, batch_size: int, messages: list) -> tuple:
        direct = self.load_embedding(req=req, batch_size=batch_size, messages=messages)
        if direct[0] is not None or req.ref_embed is not None:
            return direct
        key = self.key(req)
        model = self.runtime.model
        if key is None or getattr(model, "speaker_inversion", None) is not None:
            return direct
        # Adapter contents affect the speaker encoder; base/adapter never share state.
        adapter = self.runtime._resolve_lora_adapter_path(req.lora_adapter)
        adapter_signature = None
        if adapter is not None:
            adapter_signature = tuple(
                file_signature(str(p)) for p in sorted(Path(adapter).rglob("*")) if p.is_file()
            )
        key = ("speaker", key, adapter_signature)
        cached = self.cache.get(key)
        if cached is None:
            start = len(messages)
            latent, mask = self.latent(req=req, batch_size=1, messages=messages)
            latent, mask = patch_sequence_with_mask(latent, mask, model.cfg.speaker_patch_size)
            state = model.speaker_norm(model.speaker_encoder(latent, mask))
            state, mask = model._prepend_masked_mean_token(state, mask)
            tensors = (state.detach().cpu().clone(), mask.detach().cpu().clone())
            self.cache.put(key, tensors, tuple(messages[start:]))
        else:
            tensors, warnings = cached
            messages.extend(warnings)
            messages.append("info: speaker state cache hit.")
        dtype = next(model.parameters()).dtype
        return tuple(
            t.to(device=self.runtime.model_device, dtype=dtype if i == 0 else torch.bool)
            .expand(batch_size, *t.shape[1:])
            .clone()
            for i, t in enumerate(tensors)
        )


def install_reference_cache(runtime: Any, *, entries: int, max_mb: int) -> ReferenceCache:
    cache = ReferenceCache(runtime, entries, max_mb * 1024 * 1024)
    runtime._load_reference_latent = MethodType(lambda _self, **kw: cache.latent(**kw), runtime)
    runtime._load_speaker_embedding_condition = MethodType(
        lambda _self, **kw: cache.speaker(**kw), runtime
    )
    runtime.reference_cache = cache
    return cache
