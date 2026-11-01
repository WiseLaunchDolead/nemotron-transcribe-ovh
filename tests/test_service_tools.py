import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


downloader = load("download_models")
smoke = load("smoke_test")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.data = b"test weights"
        self.entry = {"name": "model.gguf", "size": len(self.data),
                      "sha256": hashlib.sha256(self.data).hexdigest()}
        self.manifest = {"repository": "example/model", "revision": "a" * 40, "files": [self.entry]}

    def test_verified_cache_never_uses_network(self):
        (self.directory / "model.gguf").write_bytes(self.data)
        with patch.object(downloader.subprocess, "run") as fetch:
            downloader.download(self.manifest, self.directory)
            fetch.assert_not_called()

    def test_corrupt_cache_fails_before_network(self):
        (self.directory / "model.gguf").write_bytes(b"bad")
        with patch.object(downloader.subprocess, "run") as fetch:
            with self.assertRaisesRegex(RuntimeError, "Corrupt"):
                downloader.download(self.manifest, self.directory)
            fetch.assert_not_called()

    def test_complete_partial_is_recovered(self):
        (self.directory / "model.gguf.part").write_bytes(self.data)
        with patch.object(downloader.subprocess, "run") as fetch:
            downloader.download(self.manifest, self.directory)
            fetch.assert_not_called()
        self.assertEqual((self.directory / "model.gguf").read_bytes(), self.data)

    def test_wrong_download_is_never_promoted(self):
        def fetch(*args, **kwargs):
            (self.directory / "model.gguf.part").write_bytes(b"x" * len(self.data))
        with patch.object(downloader.subprocess, "run", side_effect=fetch):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                downloader.download(self.manifest, self.directory)
        self.assertFalse((self.directory / "model.gguf").exists())
        self.assertFalse((self.directory / "model.gguf.part").exists())


class AcceptanceTests(unittest.TestCase):
    def result(self):
        return {"text": "Bonjour. Salut.", "duration": 5,
                "words": [{"word": "Bonjour", "start": 0.1, "end": 1, "speaker": 1},
                          {"word": "Salut", "start": 2, "end": 3, "speaker": 2}]}

    def test_two_speakers_and_timestamps(self):
        self.assertEqual(smoke.validate(self.result(), 2)["speakers"], [1, 2])

    def test_transcript_without_alignment_is_a_failure(self):
        result = self.result()
        result["words"] = []
        with self.assertRaisesRegex(ValueError, "absent"):
            smoke.validate(result, 2)

    def test_single_speaker_is_a_failure_for_two_voice_fixture(self):
        result = self.result()
        result["words"][1]["speaker"] = 1
        with self.assertRaisesRegex(ValueError, "Expected"):
            smoke.validate(result, 2)

    def test_out_of_bounds_or_nonfinite_timestamp_is_rejected(self):
        for end in (6, float("nan"), float("inf")):
            with self.subTest(end=end):
                result = self.result()
                result["words"][0]["end"] = end
                with self.assertRaises(ValueError):
                    smoke.validate(result, 2)


class SmokeIntegrationTests(unittest.TestCase):
    def test_auth_multipart_and_saved_metrics(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer test-secret":
                    self.send_response(401)
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                result = {"ready": True} if self.path == "/ready" else {"data": [{
                    "id": "nemotron-transcribe", "capability": "transcription",
                    "word_timestamps": True, "speaker_diarization": True,
                }]}
                self.wfile.write(json.dumps(result).encode())

            def do_POST(self):
                received.append((self.path, self.headers.get("Authorization"),
                                 self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(AcceptanceTests().result()).encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                audio = directory / "input.wav"
                audio.write_bytes(b"RIFF-test-audio")
                smoke.run(f"http://127.0.0.1:{server.server_port}", audio, "test-secret", 5,
                          directory / "results", 2)
                metrics = json.loads((directory / "results/metrics.json").read_text())
                self.assertEqual(metrics["speakers"], [1, 2])
                self.assertTrue((directory / "results/transcription.json").exists())
            self.assertEqual(len(received), 1)
            path, auth, body = received[0]
            self.assertEqual(path, "/v1/audio/transcriptions")
            self.assertEqual(auth, "Bearer test-secret")
            for expected in (b'name="file"', b'name="language"', b'fr',
                             b'name="diarization"', b'true', b'verbose_json', b'RIFF-test-audio'):
                self.assertIn(expected, body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class StartupTests(unittest.TestCase):
    def test_auth_errors_fail_before_gpu_or_download(self):
        for mode, key in (("api-key", ""), ("ovh", "conflicting-key"), ("unknown", "")):
            with self.subTest(mode=mode):
                result = subprocess.run(["bash", str(ROOT / "start.sh")],
                                        env={**os.environ, "AUTH_MODE": mode,
                                             "NEMO_SPEECH_HTTP_API_KEY": key},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("nvidia-smi", result.stderr)
