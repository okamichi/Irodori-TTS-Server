"""Play SSE speech chunks as they arrive while subsequent chunks are generated."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import threading
import urllib.request

import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--url", default="http://127.0.0.1:8088/v1/audio/speech")
    parser.add_argument("--voice", default="none")
    parser.add_argument("--caption")
    parser.add_argument("--first-chars", type=int, default=1)
    args = parser.parse_args()
    options = {"first_sentence_chunk_min_chars": args.first_chars}
    if args.caption is not None:
        options["caption"] = args.caption
    payload = {
        "model": "irodori-tts",
        "input": args.text,
        "voice": args.voice,
        "response_format": "wav",
        "stream_format": "sse",
        "irodori": options,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if os.environ.get("IRODORI_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["IRODORI_API_KEY"]
    request = urllib.request.Request(args.url, data=json.dumps(payload).encode(), headers=headers)
    chunks = queue.Queue(maxsize=2)
    stop = threading.Event()
    done = object()

    def enqueue(item):
        while not stop.is_set():
            try:
                chunks.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    with urllib.request.urlopen(request, timeout=600) as response:

        def receive():
            event = ""
            try:
                for line in response:
                    if stop.is_set():
                        break
                    text = line.decode("utf-8").strip()
                    if text.startswith("event:"):
                        event = text[6:].strip()
                    elif text.startswith("data:"):
                        data = json.loads(text[5:])
                        if event == "audio_chunk":
                            enqueue(base64.b64decode(data["audio_base64"]))
                        elif event == "error":
                            raise RuntimeError(data["error"]["message"])
                        elif event == "done":
                            break
            except Exception as exc:
                enqueue(exc)
            finally:
                enqueue(done)

        worker = threading.Thread(target=receive, daemon=True)
        worker.start()
        try:
            while True:
                chunk = chunks.get()
                if chunk is done:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                audio, sample_rate = sf.read(io.BytesIO(chunk), dtype="float32", always_2d=True)
                sd.play(audio, sample_rate)
                sd.wait()
        finally:
            stop.set()
            sd.stop()
    worker.join(timeout=1)


if __name__ == "__main__":
    main()
