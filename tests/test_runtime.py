from __future__ import annotations

import threading

import pytest
import torch

from irodori_openai_tts import runtime as runtime_module
from irodori_openai_tts.config import Settings
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError, RuntimeManager
from irodori_tts.inference_runtime import RuntimeKey, resolve_runtime_dtype


def test_mps_bf16_loader_relaxes_only_its_private_resolver(monkeypatch):
    probes = []
    monkeypatch.setattr(runtime_module, "_probe_mps_bf16", lambda: probes.append(True))

    def fake_loader(cls, key):
        return (
            resolve_runtime_dtype(
                precision=key.model_precision, device=torch.device(key.model_device)
            ),
            resolve_runtime_dtype(
                precision=key.codec_precision, device=torch.device(key.codec_device)
            ),
        )

    monkeypatch.setattr(runtime_module.InferenceRuntime, "from_key", classmethod(fake_loader))
    result = runtime_module._load_runtime(
        RuntimeKey(checkpoint="unused", model_device="mps", model_precision="bf16")
    )
    assert probes == [True]
    assert result == (torch.bfloat16, torch.float32)
    with pytest.raises(ValueError, match="CUDA or XPU"):
        resolve_runtime_dtype(precision="bf16", device=torch.device("mps"))


def test_mps_bf16_probe_failure_stops_loading(monkeypatch):
    def fail_probe():
        raise RuntimeError("BF16 unavailable")

    monkeypatch.setattr(runtime_module, "_probe_mps_bf16", fail_probe)
    with pytest.raises(RuntimeError, match="BF16 unavailable"):
        runtime_module._load_runtime(
            RuntimeKey(checkpoint="unused", model_device="mps", model_precision="bf16")
        )


def test_sampling_steps_default_to_model_default(monkeypatch):
    monkeypatch.delenv("IRODORI_DEFAULT_NUM_STEPS", raising=False)

    assert Settings(_env_file=None).default_num_steps is None


@pytest.mark.parametrize("steps", [4, 40])
def test_sampling_steps_environment_override(monkeypatch, steps):
    monkeypatch.setenv("IRODORI_DEFAULT_NUM_STEPS", str(steps))

    assert Settings(_env_file=None).default_num_steps == steps


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_mlx_precision_is_independent_of_torch_precision(monkeypatch, precision):
    monkeypatch.setenv("IRODORI_MLX_DIT_PRECISION", precision)
    settings = Settings(model_precision="fp32", codec_precision="fp32", _env_file=None)
    assert settings.mlx_dit_precision == precision
    assert settings.model_precision == settings.codec_precision == "fp32"
    monkeypatch.setenv("IRODORI_MLX_DIT_PRECISION", "int4")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
def test_runtime_passes_mlx_precision_to_installer(tmp_path, monkeypatch, precision):
    from irodori_openai_tts import mlx_backend

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    loaded_runtime = object()
    calls = []
    monkeypatch.setattr(
        runtime_module.InferenceRuntime, "from_key", staticmethod(lambda key: loaded_runtime)
    )
    monkeypatch.setattr(
        mlx_backend,
        "install_mlx_backend",
        lambda runtime, *, dit_precision: calls.append((runtime, dit_precision)),
    )
    manager = RuntimeManager(
        Settings(
            checkpoint=str(checkpoint),
            inference_backend="mlx",
            mlx_dit_precision=precision,
            model_device="cpu",
            codec_device="cpu",
            reference_cache_entries=0,
            _env_file=None,
        )
    )
    assert manager.get() is loaded_runtime
    assert calls == [(loaded_runtime, precision)]


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        reference_cache_entries=0,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


def test_runtime_resolves_local_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    manager = RuntimeManager(Settings(checkpoint=str(checkpoint), _env_file=None))

    assert manager._resolve_checkpoint_path() == str(checkpoint)


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    manager = RuntimeManager(
        Settings(checkpoint=str(tmp_path / "missing.safetensors"), _env_file=None)
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager._resolve_checkpoint_path()


def test_runtime_downloads_hf_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", _env_file=None))

    def fake_download(repo_id):
        assert repo_id == "owner/repo"
        return "/cache/model.safetensors"

    monkeypatch.setattr(runtime_module, "download_hf_checkpoint", fake_download)

    assert manager._resolve_checkpoint_path() == "/cache/model.safetensors"
