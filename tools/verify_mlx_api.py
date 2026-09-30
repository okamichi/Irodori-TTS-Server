"""Local HTTP/SSE smoke test of the configured MLX server, using a temporary port."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--ref-wav", default="/private/tmp/irodori-mlx-compiled/torch_warm.wav")
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/irodori-mlx-api"))
    parser.add_argument("--mlx-precision", choices=["fp32", "fp16"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    env = dict(
        os.environ,
        IRODORI_INFERENCE_BACKEND="mlx",
        IRODORI_API_KEY="irodori-local-smoke",
        IRODORI_PRELOAD="false",
    )
    if args.mlx_precision is not None:
        env["IRODORI_MLX_DIT_PRECISION"] = args.mlx_precision
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "irodori_openai_tts",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Smoke server exited; see {args.output / 'server.log'}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=1) as response:
                        health = json.load(response)
                    assert health["model"]["inference_backend"] == "mlx"
                    if args.mlx_precision is not None:
                        assert health["model"]["mlx_dit_precision"] == args.mlx_precision
                    break
                except urllib.error.URLError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.2)
            payload = {
                "model": "irodori-tts",
                "input": "こんにちは。今日はいい天気ですね。最後の文章です。",
                "response_format": "wav",
                "stream_format": "sse",
                "irodori": {
                    "ref_wav": args.ref_wav,
                    "seed": 19,
                    "num_steps": 40,
                    "max_seconds": 4,
                    "chunk_min_chars": 8,
                    "first_sentence_chunk_min_chars": 1,
                },
            }
            request = urllib.request.Request(
                base + "/v1/audio/speech",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer irodori-local-smoke",
                },
            )
            start = time.perf_counter()
            events = []
            with urllib.request.urlopen(request, timeout=300) as response:
                assert response.headers["Content-Type"].startswith("text/event-stream")
                kind = ""
                for line in response:
                    text = line.decode().strip()
                    if text.startswith("event:"):
                        kind = text[6:].strip()
                    elif text.startswith("data:"):
                        data = json.loads(text[5:])
                        if kind == "error":
                            raise RuntimeError(data)
                        if kind == "audio_chunk":
                            raw = base64.b64decode(data.pop("audio_base64"))
                            audio, rate = sf.read(io.BytesIO(raw), always_2d=True)
                            assert rate == 48000 and len(audio) > 0 and np.isfinite(audio).all()
                            (args.output / f"chunk-{data['index']}.wav").write_bytes(raw)
                            data["received_seconds"] = time.perf_counter() - start
                            events.append(data)
                            print(json.dumps(data, ensure_ascii=False), flush=True)
                        if kind == "done":
                            assert data["chunks"] == 3
            assert [item["index"] for item in events] == [0, 1, 2]
            assert events[1]["stage_timings"]["prepare_reference"] < 0.05
            report = {"chunks": events, "elapsed_seconds": time.perf_counter() - start}
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(f"Passed HTTP/SSE smoke test; artifacts: {args.output}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
