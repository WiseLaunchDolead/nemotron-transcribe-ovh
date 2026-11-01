#!/usr/bin/env python3
"""Download the pinned public GGUF pair, resume interruptions, verify every byte."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == entry["sha256"]


def download(manifest, directory):
    directory.mkdir(parents=True, exist_ok=True)
    # A shared cache must never expose partially downloaded weights to another process.
    with (directory / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in manifest["files"]:
            target = directory / entry["name"]
            if valid(target, entry):
                print(f"Verified cached model: {entry['name']}", flush=True)
                continue
            if target.exists():
                raise RuntimeError(f"Corrupt cached model: {target}. Remove it explicitly and retry.")
            partial = target.with_suffix(target.suffix + ".part")
            # A fully downloaded .part is recoverable after a process was interrupted.
            if valid(partial, entry):
                partial.replace(target)
                continue
            if partial.exists() and partial.stat().st_size >= entry["size"]:
                partial.unlink()
            remaining = entry["size"] - (partial.stat().st_size if partial.exists() else 0)
            if shutil.disk_usage(directory).free < remaining + 1024**3:
                raise RuntimeError("Insufficient model-cache space (allow 25 GB for the pair).")
            url = (f"https://huggingface.co/{manifest['repository']}/resolve/"
                   f"{manifest['revision']}/{entry['name']}")
            print(f"Downloading pinned model: {entry['name']}", flush=True)
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https",
                "--retry", "5", "--retry-delay", "5", "--connect-timeout", "30",
                "--max-time", "7200", "--speed-limit", "1024", "--speed-time", "120",
                "--continue-at", "-", "--output", str(partial), url,
            ], check=True)
            if not valid(partial, entry):
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Size/SHA-256 mismatch for {entry['name']}")
            partial.replace(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    # In the image, script and manifest are adjacent; in the repo, the script is in scripts/.
    manifest_path = Path(__file__).resolve().parent / "models.lock.json"
    if not manifest_path.exists():
        manifest_path = manifest_path.parent.parent / "models.lock.json"
    download(json.loads(manifest_path.read_text()), args.directory)
