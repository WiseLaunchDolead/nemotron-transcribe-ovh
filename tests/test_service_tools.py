import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
