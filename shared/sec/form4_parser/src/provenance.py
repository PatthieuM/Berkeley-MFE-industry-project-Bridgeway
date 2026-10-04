"""Content-addressed raw observations and atomic, reproducible file manifests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from urllib.parse import urlsplit


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    """Write and fsync a temporary file, then atomically replace ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, value) -> None:
    """Serialize ``value`` as deterministic JSON and atomically write it."""
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n").encode())


def environment_manifest() -> dict:
    """Describe the runtime and relevant installed package versions."""
    packages = {}
    for name in (
        "pandas",
        "numpy",
        "pyarrow",
        "scipy",
        "statsmodels",
        "exchange-calendars",
        "requests",
        "lxml",
        "pytest",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def file_manifest(paths) -> list[dict]:
    """Return absolute paths, sizes, and hashes for existing input files."""
    return [{"path": str(Path(p).resolve()), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
            for p in sorted(map(Path, paths)) if p.is_file()]


class RawArchive:
    """Keep every fetched revision; a URL pointer never substitutes for raw bytes.

    Callers must use public URLs without passwords or API tokens. Imported legacy
    bytes explicitly carry unknown retrieval time, rather than fabricated history.
    """
    def __init__(self, root: str | Path):
        """Store raw filings under ``root``."""
        self.root = Path(root)

    def record(self, url: str, content: bytes, *, retrieved_at=None, headers=None,
               source="public", imported=False) -> dict:
        """Archive response bytes and record a URL observation manifest."""
        parts = urlsplit(url)
        if (
            parts.username
            or parts.password
            or any(x in parts.query.lower() for x in ("api_key=", "token=", "password="))
        ):
            raise ValueError("Secret-bearing URLs must not enter provenance")
        digest = hashlib.sha256(content).hexdigest()
        raw_path = self.root / "objects" / digest[:2] / f"{digest}.bin"
        if raw_path.exists():
            if sha256_file(raw_path) != digest:
                raise ValueError(f"Raw archive corruption: {raw_path}")
        else:
            atomic_bytes(raw_path, content)
        record = {"url": url, "sha256": digest, "bytes": len(content), "path": str(raw_path.resolve()),
                  "retrieved_at": None if imported else (retrieved_at or utc_now()),
                  "recorded_at": utc_now(), "source": source,
                  "retrieval_status": "legacy_cache_time_unknown" if imported else "retrieved",
                  "http": {k.lower(): v for k, v in (headers or {}).items()
                           if k.lower() in ("etag", "last-modified", "content-type", "date")}}
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        observation_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        atomic_json(self.root / "observations" / url_hash / f"{observation_id}.json", record)
        atomic_json(self.root / "urls" / f"{url_hash}.json", record)
        return record

    def cached(self, url: str) -> bytes | None:
        """Return verified cached bytes for ``url``, or ``None`` if absent."""
        pointer = self.root / "urls" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
        if not pointer.exists():
            return None
        record = json.loads(pointer.read_text())
        path = Path(record["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Archive pointer escapes configured root")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Raw archive hash mismatch: {path}")
        return path.read_bytes()
