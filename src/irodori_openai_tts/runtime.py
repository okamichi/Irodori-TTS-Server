from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from types import FunctionType

import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
    download_hf_checkpoint,
    resolve_runtime_dtype,
)

from .config import Settings

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


def _probe_mps_bf16() -> None:
    """Fail before loading a large checkpoint if this MPS build lacks BF16."""
    try:
        value = torch.ones((2, 2), device="mps", dtype=torch.bfloat16)
        result = value @ value
        torch.mps.synchronize()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("BF16 probe returned non-finite values")
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError("This PyTorch/macOS environment cannot run BF16 on MPS.") from exc


def _load_runtime(key: RuntimeKey) -> InferenceRuntime:
    mps_bf16 = any(
        str(device).strip().lower() == "mps" and str(precision).strip().lower() == "bf16"
        for device, precision in (
            (key.model_device, key.model_precision),
            (key.codec_device, key.codec_precision),
        )
    )
    if not mps_bf16:
        return InferenceRuntime.from_key(key)
    _probe_mps_bf16()
    logger.warning("Using experimental MPS BF16 inference after a successful device probe")

    def dtype_resolver(*, precision, device):
        if str(precision).strip().lower() == "bf16" and device.type == "mps":
            return torch.bfloat16
        return resolve_runtime_dtype(precision=precision, device=device)

    # Relax only this loader's dtype guard. Do not modify site-packages or the
    # process-wide upstream resolver used by other runtimes.
    original = InferenceRuntime.from_key.__func__
    namespace = dict(original.__globals__)
    namespace["resolve_runtime_dtype"] = dtype_resolver
    loader = FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    loader.__kwdefaults__ = original.__kwdefaults__
    return loader(InferenceRuntime, key)


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime: InferenceRuntime | None = None
        self._checkpoint_path: str | None = None

    def get(self) -> InferenceRuntime:
        if self._runtime is not None:
            return self._runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Model is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )

        try:
            if self._runtime is None:
                logger.info("loading runtime")
                t0 = time.perf_counter()
                self._checkpoint_path = self._resolve_checkpoint_path()
                logger.info("checkpoint resolved: %s", self._checkpoint_path)
                runtime = _load_runtime(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device=self._resolve_device(self.settings.model_device),
                        codec_repo=str(self.settings.codec_repo),
                        model_precision=str(self.settings.model_precision),
                        codec_device=self._resolve_device(self.settings.codec_device),
                        codec_precision=str(self.settings.codec_precision),
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=bool(self.settings.compile_model),
                        compile_dynamic=bool(self.settings.compile_dynamic),
                    )
                )
                try:
                    if self.settings.inference_backend == "mlx":
                        from .mlx_backend import install_mlx_backend

                        install_mlx_backend(runtime, dit_precision=self.settings.mlx_dit_precision)
                    if (
                        self.settings.reference_cache_entries
                        and self.settings.reference_cache_max_mb
                    ):
                        from .reference_cache import install_reference_cache

                        install_reference_cache(
                            runtime,
                            entries=self.settings.reference_cache_entries,
                            max_mb=self.settings.reference_cache_max_mb,
                        )
                except Exception:
                    self._runtime = None
                    raise
                self._runtime = runtime
                elapsed = time.perf_counter() - t0
                logger.info("runtime loaded in %.2fs", elapsed)
            return self._runtime
        finally:
            self._lock.release()

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._runtime is None and self._lock.locked()

    def _resolve_checkpoint_path(self) -> str:
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            path = Path(str(self.settings.checkpoint)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return str(path)

        repo_id = str(self.settings.hf_checkpoint).strip()
        if repo_id == "":
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        logger.info("downloading checkpoint assets from hf://%s", repo_id)
        t0 = time.perf_counter()
        path = download_hf_checkpoint(repo_id)
        elapsed = time.perf_counter() - t0
        logger.info("checkpoint download/cache lookup completed in %.2fs", elapsed)
        return path

    @staticmethod
    def _resolve_device(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in {"", "auto"}:
            return default_runtime_device()
        return str(value)
