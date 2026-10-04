"""Check the saved files, the saved-result rendering and the input check, without running an analysis."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import render_report
import run_split

HERE = Path(__file__).resolve().parent
PINNED_FILES = {
    "run_split.py": "607d6cb8b9ed530b19ff19dd5ef8b119e0afd1309666b44145a63f138d686efd",
    "results.json": "e59118f6d5d4e847612c56c34b244e55ed66f324fc9fddabd286618892302c1a",
    "RESULTS.md": "bc935f692858ad7fe55126707a166469b593b2a4d615d064599aba44dd0cbf68",
    "train_ledger.csv": "13eb4d7beaf750e455d291e283e0139e3baa5477b7f2e3543bab887a3380b3c0",
}


class SavedFileTests(unittest.TestCase):
    def test_saved_files_unchanged(self):
        for name, expected in PINNED_FILES.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), expected)

    def test_saved_report_contains_primary_intervals_and_contrast(self):
        saved = json.loads((HERE / "results.json").read_text())
        rendered = render_report.render(saved)
        for expected in ("+0.90 pp", "0.031", "[+0.14, +1.63] pp", "[-0.41, +2.11] pp", "251 issuers"):
            self.assertIn(expected, rendered)
        contrast = next(line for line in rendered.splitlines()
                        if line.startswith("- Clustered minus below-threshold purchases:"))
        for expected in ("+0.49 pp", "n=722", "comparison n=1968"):
            self.assertIn(expected, contrast)
        self.assertNotIn("Unavailable", contrast)
        self.assertNotIn("'mean_pp':", rendered)
        self.assertNotIn("wide by construction", rendered)

    def test_report_handles_failed_test_and_price_fits(self):
        saved = json.loads((HERE / "results.json").read_text())
        saved["test"] = {"n": 0, "status": "too_few"}
        saved["fragility"] = None
        saved["descriptives_train_validation"]["train"]["price_screen"]["price_ge_5"].update(
            mean_pp=None, t=None, share_of_events=None, status="too_few")
        self.assertIn("Unavailable (too_few; n=0)", render_report.render(saved))

    def test_renderer_protects_original_input_and_existing_output_paths(self):
        for name in PINNED_FILES:
            with self.subTest(name=name), self.assertRaises(ValueError):
                render_report.validate_output(HERE / "results.json", HERE / name)
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary) / "baseline.md"
            baseline.write_text("preserve")
            with self.assertRaises(FileExistsError):
                render_report.validate_output(HERE / "results.json", baseline)
            with self.assertRaises(ValueError):
                render_report.validate_output(baseline, baseline)
            with self.assertRaises(ValueError):
                render_report.validate_output(HERE / "results.json", Path(temporary) / "events.parquet")
            render_report.validate_output(HERE / "results.json", Path(temporary) / "new.md")
            self.assertEqual(baseline.read_text(), "preserve")


class InputCheckTests(unittest.TestCase):
    """Checks input hashes without changing files."""

    def test_pins_match_the_hashes_recorded_in_the_saved_result(self):
        recorded = json.loads((HERE / "results.json").read_text())["inputs"]
        for name in run_split.INPUTS:
            self.assertEqual(run_split.PINNED_SHA256[name], recorded[name])
        self.assertEqual(run_split.ARCHIVE_SHA256, recorded["archive"])

    @unittest.skipUnless((HERE / "frozen" / "events.parquet").exists(), "the frozen inputs are not installed here")
    def test_installed_inputs_match_their_pins(self):
        hashes = run_split.verify_inputs()
        for name in run_split.INPUTS:
            self.assertEqual(hashes[name], run_split.PINNED_SHA256[name])
        self.assertNotIn("archive", hashes)

    def test_missing_inputs_are_reported_not_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "restore the four inputs"):
                run_split.verify_inputs(temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_corrupted_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in run_split.INPUTS:
                (Path(temporary) / name).write_bytes(b"not the frozen input")
            with self.assertRaisesRegex(RuntimeError, "pinned SHA-256"):
                run_split.verify_inputs(temporary)


if __name__ == "__main__":
    unittest.main()
