#!/usr/bin/env python3
"""Test a French WAV against the deployed API; preserve measurements and raw response."""
import argparse
import json
import math
import os
from pathlib import Path
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid


def validate(result, min_speakers):
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise ValueError("Empty transcript")
    duration = result.get("duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Missing or invalid audio duration")
    words = result.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("Word timestamps are absent (alignment may have failed)")
    speakers = set()
    for word in words:
        if not isinstance(word.get("word"), str) or not word["word"].strip():
            raise ValueError("Invalid word")
        start, end = word.get("start"), word.get("end")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)):
            raise ValueError("Invalid timestamp")
        if not 0 <= start <= end <= duration + 0.1:
            raise ValueError("Timestamp outside audio duration")
        speaker = word.get("speaker")
        if not isinstance(speaker, int) or isinstance(speaker, bool) or speaker < 1:
            raise ValueError("Missing speaker label")
        speakers.add(speaker)
    if len(speakers) < min_speakers:
        raise ValueError(f"Expected >= {min_speakers} speakers; found {len(speakers)}")
    return {"duration_seconds": duration, "words": len(words), "speakers": sorted(speakers)}


def run(url, audio, token, timeout, output, min_speakers):
    base = url.rstrip("/")
    if not base.startswith("https://") and not base.startswith("http://127.0.0.1:"):
        raise ValueError("Use HTTPS for remote services")
    # Do not follow redirects carrying credentials to a different origin.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    # Python.org's macOS installation may not have its own CA bundle populated.
    # Use the system CA bundle when present; certificate verification stays enabled.
    ca_file = os.environ.get("SSL_CERT_FILE")
    if not ca_file and sys.platform == "darwin" and Path("/etc/ssl/cert.pem").is_file():
        ca_file = "/etc/ssl/cert.pem"
    context = ssl.create_default_context(cafile=ca_file)
    client = urllib.request.build_opener(NoRedirect, urllib.request.HTTPSHandler(context=context))

    def get(path, authenticated=True):
        headers = {"Authorization": "Bearer " + token} if authenticated else {}
        with client.open(urllib.request.Request(base + path, headers=headers), timeout=timeout) as response:
            return json.load(response)

    try:
        get("/v1/models", authenticated=False)
    except urllib.error.HTTPError as error:
        if error.code not in (401, 403):
            raise
    else:
        raise ValueError("Unauthenticated model-list request succeeded")
    ready_started = time.monotonic()
    ready = get("/ready")
    if ready.get("ready") is not True:
        raise ValueError("Service is not ready")
    readiness_seconds = time.monotonic() - ready_started
    models = get("/v1/models")
    asr_models = [m for m in models.get("data", []) if m.get("capability") == "transcription"]
    if not asr_models or not all(asr_models[0].get(k) for k in ("word_timestamps", "speaker_diarization")):
        raise ValueError("Model does not advertise timestamps and diarization")
    boundary = "nemotron-" + uuid.uuid4().hex
    parts = []
    for name, value in {"language": "fr", "response_format": "verbose_json", "diarization": "true"}.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    parts.extend([audio.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    body = b"".join(parts)
    if len(body) > 128 * 1024**2:
        raise ValueError("Upload exceeds the configured 128 MiB limit")
    request = urllib.request.Request(base + "/v1/audio/transcriptions", data=body, headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "multipart/form-data; boundary=" + boundary,
    })
    started = time.monotonic()
    with client.open(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    # Save failed alignment/diarization evidence as well as successful responses.
    output.mkdir(parents=True, exist_ok=True)
    (output / "transcription.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = validate(result, min_speakers)
    summary.update({"request_seconds": round(elapsed, 3), "readiness_request_seconds": round(readiness_seconds, 3),
                    "real_time_factor": round(elapsed / summary["duration_seconds"], 4),
                    "model": asr_models[0]["id"], "authentication": "unauthenticated request rejected"})
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--min-speakers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    secret = os.environ.get("AI_APP_TOKEN", "")
    if not secret:
        parser.error("Set AI_APP_TOKEN in your environment")
    run(args.url, args.audio, secret, args.timeout, args.output, args.min_speakers)
