from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from irodori_openai_tts.reference_cache import TensorCache, install_reference_cache
from irodori_tts.inference_runtime import SamplingRequest


class Speaker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, latent, mask):
        self.calls += 1
        return latent * self.weight


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(speaker_patch_size=2)
        self.speaker_encoder = Speaker()
        self.speaker_norm = torch.nn.Identity()

    @staticmethod
    def _prepend_masked_mean_token(state, mask):
        return (
            torch.cat((state.mean(1, keepdim=True), state), dim=1),
            torch.cat((torch.ones(mask.shape[0], 1, dtype=torch.bool), mask), dim=1),
        )


class Runtime:
    def __init__(self):
        self.model = Model()
        self.model_cfg = SimpleNamespace(use_speaker_condition_resolved=True)
        self.codec = SimpleNamespace(deterministic_encode=True)
        self.model_device = torch.device("cpu")
        self.default_max_ref_seconds = 120
        self.loads = 0

    def _load_reference_latent(self, *, req, batch_size, messages):
        if req.ref_wav and req.ref_wavs:
            raise ValueError("ref_wav and ref_wavs cannot be used together")
        self.loads += 1
        messages.append("warning: reference trimmed")
        return torch.ones(batch_size, 6, 4), torch.ones(batch_size, 6, dtype=torch.bool)

    def _load_speaker_embedding_condition(self, **kw):
        return None, None

    def _resolve_lora_adapter_path(self, path):
        return path


def test_reuses_latents_and_state_across_text_and_batch_sizes(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"first")
    runtime = Runtime()
    install_reference_cache(runtime, entries=8, max_mb=1)
    req = SamplingRequest(text="first", ref_wav=str(wav))
    one = runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    req.text = "second"
    messages = []
    two = runtime._load_speaker_embedding_condition(req=req, batch_size=2, messages=messages)
    assert runtime.loads == runtime.model.speaker_encoder.calls == 1
    torch.testing.assert_close(two[0], one[0].expand(2, -1, -1))
    assert "warning: reference trimmed" in messages
    assert "info: speaker state cache hit." in messages
    two[0].zero_()
    three = runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    torch.testing.assert_close(three[0], one[0])


def test_reference_changes_invalidate_both_caches(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"first")
    runtime = Runtime()
    install_reference_cache(runtime, entries=8, max_mb=1)
    req = SamplingRequest(text="first", ref_wav=str(wav))
    for _ in range(2):
        runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
        wav.write_bytes(b"longer content")
    assert runtime.loads == runtime.model.speaker_encoder.calls == 2
    req.ref_normalize_db = -20
    runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    assert runtime.loads == runtime.model.speaker_encoder.calls == 3


def test_adapter_changes_only_invalidate_state(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"voice")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter.safetensors"
    weights.write_bytes(b"first")
    runtime = Runtime()
    install_reference_cache(runtime, entries=8, max_mb=1)
    req = SamplingRequest(text="first", ref_wav=str(wav))
    runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    req.lora_adapter = str(adapter)
    runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    weights.write_bytes(b"modified weights")
    runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    assert runtime.loads == 1
    assert runtime.model.speaker_encoder.calls == 3


def test_validation_cannot_be_hidden_by_a_cache_hit(tmp_path):
    wav = tmp_path / "voice.wav"
    wav.write_bytes(b"voice")
    runtime = Runtime()
    install_reference_cache(runtime, entries=8, max_mb=1)
    req = SamplingRequest(text="first", ref_wav=str(wav))
    runtime._load_speaker_embedding_condition(req=req, batch_size=1, messages=[])
    req.ref_wavs = [str(wav)]
    with pytest.raises(ValueError, match="cannot be used together"):
        runtime._load_reference_latent(req=req, batch_size=1, messages=[])


def test_lru_obeys_memory_and_entry_limits():
    cache = TensorCache(entries=2, max_bytes=24)
    for i in range(3):
        cache.put((i,), (torch.ones(4),))
    assert cache.get((0,)) is None
    assert cache.get((1,)) is None
    assert cache.get((2,)) is not None
    assert cache.bytes == 16
    cache.put((3,), (torch.ones(7),))
    assert cache.get((2,)) is not None
    cache.clear()
    assert cache.bytes == 0
    assert not cache.values
