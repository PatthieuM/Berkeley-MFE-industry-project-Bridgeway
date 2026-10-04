"""Archive integrity and path-containment tests using temporary files only."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from shared.sec.form4_parser.src.provenance import RawArchive


class RawArchiveTests(unittest.TestCase):
    def test_revisions_remain_available_and_latest_pointer_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = RawArchive(Path(temporary) / "archive")
            first = archive.record("https://example.org/filing", b"first")
            second = archive.record("https://example.org/filing", b"second")
            self.assertEqual(Path(first["path"]).read_bytes(), b"first")
            self.assertEqual(archive.cached("https://example.org/filing"), b"second")
            self.assertEqual(len(list((archive.root / "observations").rglob("*.json"))), 2)
            Path(second["path"]).write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                archive.cached("https://example.org/filing")

    def test_parent_traversal_and_symlink_targets_cannot_escape_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive = RawArchive(root / "archive")
            url = "https://example.org/filing"
            record = archive.record(url, b"original")
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            link = archive.root / "outside-link"
            link.symlink_to(outside)
            pointer = archive.root / "urls" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
            for path in (archive.root / ".." / "outside.bin", link):
                with self.subTest(path=path):
                    pointer.write_text(json.dumps({**record, "path": str(path),
                                                  "sha256": hashlib.sha256(b"outside").hexdigest()}))
                    with self.assertRaisesRegex(ValueError, "escapes configured root"):
                        archive.cached(url)


if __name__ == "__main__":
    unittest.main()
