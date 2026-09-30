"""Compare Torch/MPS and hybrid MLX using one local checkpoint; save review audio.

Run from the repository root with .venv/bin/python tools/benchmark_backends.py.
No checkpoint files or server settings are changed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from irodori_openai_tts.config import Settings
from irodori_openai_tts.mlx_backend import install_mlx_backend, set_dit_precision
from irodori_openai_tts.reference_cache import install_reference_cache
from irodori_openai_tts.runtime import RuntimeManager
from irodori_tts.inference_runtime import SamplingRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="../Irodori-TTS-v4-Large/model.safetensors")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/irodori-mlx-validation"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--mlx-precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--compare-mlx-precisions", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        checkpoint=args.checkpoint,
        inference_backend="torch",
        model_device="mps",
        codec_device="mps",
        model_precision=args.precision,
        codec_precision="fp32",
        compile_model=False,
        reference_cache_entries=0,
        api_key=None,
        _env_file=None,
    )
    load_start = time.perf_counter()
    runtime = RuntimeManager(settings).get()
    report = {
        "steps": args.steps,
        "seconds": args.seconds,
        "precision": args.precision,
        "mlx_dit_precision": "fp32" if args.compare_mlx_precisions else args.mlx_precision,
        "load_seconds": time.perf_counter() - load_start,
    }
    req = SamplingRequest(
        text="こんにちは。今日はいい天気ですね。",
        no_ref=True,
        caption="落ち着いた自然な女性の声。",
        seconds=args.seconds,
        num_steps=args.steps,
        seed=19,
        trim_tail=False,
    )

    def run(name, request):
        start = time.perf_counter()
        result = runtime.synthesize(request, log_fn=lambda text: print(text, flush=True))
        seconds = time.perf_counter() - start
        audio = result.audio.detach().cpu().float().numpy().T
        if not np.isfinite(audio).all():
            raise RuntimeError(f"{name}: non-finite audio")
        sf.write(args.output / f"{name}.wav", audio, result.sample_rate)
        report[name] = {
            "elapsed_seconds": seconds,
            "audio_seconds": len(audio) / result.sample_rate,
            "rtf": seconds / (len(audio) / result.sample_rate),
            "stages": dict(result.stage_timings),
            "messages": result.messages,
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
        return result

    # Cold and warm measurements use exactly the same settings and seed.
    run("torch_cold", req)
    baseline = run("torch_warm", req)
    start = time.perf_counter()
    install_mlx_backend(runtime, dit_precision=report["mlx_dit_precision"])
    report["mlx_conversion_seconds"] = time.perf_counter() - start
    run("mlx_cold", req)
    result = run("mlx_warm", req)
    difference = result.audio.cpu().float() - baseline.audio.cpu().float()
    report["audio_difference"] = {
        "rmse": float(torch.mean(difference.square()).sqrt()),
        "max_abs": float(difference.abs().max()),
    }
    report["warm_speedup"] = (
        report["torch_warm"]["elapsed_seconds"] / report["mlx_warm"]["elapsed_seconds"]
    )
    if args.compare_mlx_precisions:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        report["mlx_fp32_weight_bytes"] = sum(
            value.nbytes for _, value in tree_flatten(runtime.mlx_dit.parameters())
        )
        for i in range(3):
            result = run(f"mlx_fp32_warm_{i}", req)
        report["mlx_fp32_warm_mean_seconds"] = (
            sum(report[f"mlx_fp32_warm_{i}"]["elapsed_seconds"] for i in range(3)) / 3
        )
        baseline_mlx = result.audio.cpu().float()
        set_dit_precision(runtime.mlx_dit, "fp16")
        # Compiled graphs capture weights. Use a fresh graph before measuring.
        runtime.mlx_sampler.recompile()
        mx.clear_cache()
        report["mlx_fp16_weight_bytes"] = sum(
            value.nbytes for _, value in tree_flatten(runtime.mlx_dit.parameters())
        )
        run("mlx_fp16_cold", req)
        warm_results = [run(f"mlx_fp16_warm_{i}", req) for i in range(3)]
        difference = warm_results[-1].audio.cpu().float() - baseline_mlx
        report["fp16_audio_difference"] = {
            "rmse": float(torch.mean(difference.square()).sqrt()),
            "max_abs": float(difference.abs().max()),
        }
        report["mlx_fp16_warm_mean_seconds"] = (
            sum(report[f"mlx_fp16_warm_{i}"]["elapsed_seconds"] for i in range(3)) / 3
        )
        report["fp16_speedup_over_mlx_fp32"] = (
            report["mlx_fp32_warm_mean_seconds"] / report["mlx_fp16_warm_mean_seconds"]
        )
    # Exercise normalization, MLX codec encode, speaker cache, and duration prediction.
    install_reference_cache(runtime, entries=8, max_mb=256)
    clone = SamplingRequest(
        text="今日はいい天気ですね。",
        ref_wav=str(args.output / "torch_warm.wav"),
        num_steps=args.steps,
        seed=19,
        max_seconds=4.0,
    )
    run("mlx_reference_cold", clone)
    cached = run("mlx_reference_warm", clone)
    assert "info: speaker state cache hit." in cached.messages
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
