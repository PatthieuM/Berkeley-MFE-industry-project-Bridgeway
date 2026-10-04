"""Provide rate-limited, retrying access to SEC EDGAR resources.

Responses from the submissions API and filing archive can be stored in a
content-addressed raw cache for reproducible downstream parsing.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

if __package__:
    from .src.provenance import RawArchive, atomic_bytes
else:
    from src.provenance import RawArchive, atomic_bytes

logger = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL = 0.15  # seconds between requests (~7/s, under SEC's 10/s cap)


class EdgarClient:
    """Rate-limited, retrying EDGAR client with an optional on-disk raw cache."""

    def __init__(
        self,
        contact: Optional[str] = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_retries: int = 4,
        cache_dir: Optional[str | Path] = None,
        offline: bool = False,
    ) -> None:
        """Create a rate-limited, retrying EDGAR client.

        The User-Agent comes only from ``SEC_USER_AGENT`` and must contain an email.
        ``min_interval`` (seconds between requests, at least 0.1) keeps the client
        under SEC's 10 requests/second limit. ``cache_dir`` stores raw responses so
        re-runs do not re-download; ``offline`` serves from that cache only.
        """
        configured = os.environ.get("SEC_USER_AGENT", "").strip()
        if not configured:
            raise ValueError(
                "SEC_USER_AGENT is required for every EDGAR request; set it to "
                "'Organization Name contact@example.com'."
            )
        if contact is not None and contact != configured:
            raise ValueError("Pass SEC identity through SEC_USER_AGENT, not a hard-coded contact argument.")
        contact = configured
        if "@" not in contact:
            raise ValueError("EDGAR User-Agent must include a contact email.")
        self.session = requests.Session()
        # Do NOT pin a Host header: EDGAR spans data.sec.gov (submissions API)
        # and www.sec.gov (archives); requests derives Host per-URL.
        self.session.headers.update(
            {"User-Agent": contact, "Accept-Encoding": "gzip, deflate"}
        )
        if min_interval < 0.1 or max_retries < 1:
            raise ValueError("Use at most 10 requests/second and at least one attempt")
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.offline = offline
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.archive = RawArchive(self.cache_dir / "archive") if self.cache_dir else None
        self._last_ts = 0.0

    # -- cache helpers -----------------------------------------------------
    def _cache_path(self, url: str) -> Optional[Path]:
        """Return the cache file for ``url`` (SHA-1 of the URL), or None when caching is off."""
        if not self.cache_dir:
            return None
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.bin"

    def _throttle(self) -> None:
        """Sleep as needed so consecutive requests are at least ``min_interval`` seconds apart."""
        elapsed = time.monotonic() - self._last_ts
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_ts = time.monotonic()

    # -- fetching ----------------------------------------------------------
    def get_bytes(self, url: str, use_cache: bool = True) -> bytes:
        """Read raw bytes; offline mode replays cache entries even for refresh requests."""
        read_cache = use_cache or self.offline
        path = self._cache_path(url) if read_cache else None
        if read_cache and self.archive:
            cached = self.archive.cached(url)
            if cached is not None:
                return cached
        if path and path.exists():
            content = path.read_bytes()
            if self.archive:
                self.archive.record(url, content, imported=True, source="sec_legacy_cache")
            return content
        if self.offline:
            raise FileNotFoundError(f"Offline cache miss: {url}")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code in (401, 403):
                    raise PermissionError(f"SEC access rejected ({resp.status_code}): {url}")
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = requests.HTTPError(f"EDGAR HTTP {resp.status_code}: {url}", response=resp)
                    if attempt == self.max_retries:
                        break
                    wait = min(2 ** attempt, 30)
                    logger.warning("EDGAR %s on %s; backoff %ss", resp.status_code, url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                if self.archive:
                    self.archive.record(url, resp.content, headers=dict(resp.headers), source="sec")
                if path:
                    atomic_bytes(path, resp.content)
                return resp.content
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                wait = min(2 ** attempt, 30)
                logger.warning("Request error on %s (%d/%d): %s; retry in %ss",
                               url, attempt, self.max_retries, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"Failed to GET {url} after {self.max_retries} tries") from last_exc

    def get_json(self, url: str, use_cache: bool = False) -> dict:
        """GET a UTF-8 JSON object, optionally using the raw cache."""
        import json
        return json.loads(self.get_bytes(url, use_cache=use_cache).decode("utf-8"))
