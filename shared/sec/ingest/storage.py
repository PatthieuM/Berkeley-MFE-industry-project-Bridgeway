"""SQLite storage for parsed SEC filings; see README.md for schema and retry rules."""

from __future__ import annotations

from pathlib import Path
import sqlite3

from shared.sec.form4_parser.engines import _NUMERIC_COLS, _PROVENANCE
from shared.sec.form4_parser.formats import CANON_FIELDS

COLUMNS = CANON_FIELDS + _PROVENANCE
BOOLEAN_COLUMNS = {name for name in COLUMNS if name.startswith("is_")} | {"rule_10b5_1", "equity_swap_involved"}
NAMES = ", ".join(f'"{name}"' for name in COLUMNS)  # quoted because "table" is an SQL keyword


def _column(name: str) -> str:
    kind = "NUMERIC" if name in _NUMERIC_COLS else "INTEGER" if name in BOOLEAN_COLUMNS else "TEXT"
    return f'"{name}" {kind}'


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS filings (
    accession   TEXT PRIMARY KEY,
    cik         INTEGER NOT NULL,
    form        TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('done', 'failed')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS ownership_rows ({", ".join(map(_column, COLUMNS))});
CREATE INDEX IF NOT EXISTS ownership_rows_accession ON ownership_rows(accession);
"""


class FilingStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            self.connection.executescript(SCHEMA)
            # Begin explicitly; the connection context manager only commits or rolls back.
            existing = {row[1] for row in self.connection.execute("PRAGMA table_info(ownership_rows)")}
            missing = [name for name in COLUMNS if name not in existing]
            if missing:
                with self.connection:
                    self.connection.execute("BEGIN")
                    for name in missing:
                        self.connection.execute(f"ALTER TABLE ownership_rows ADD COLUMN {_column(name)}")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def needs_fetch(self, accession: str) -> bool:
        """Return True for a filing that is not stored yet or that failed."""
        row = self.connection.execute("SELECT status FROM filings WHERE accession = ?", (accession,)).fetchone()
        return row is None or row[0] == "failed"

    def save(self, ref, rows: list[dict]) -> None:
        """Atomically replace one filing's rows and mark it done; blank text becomes NULL."""
        unknown = {name for row in rows for name in row}.difference(COLUMNS)
        if unknown:
            raise ValueError(f"Unknown parser fields: {', '.join(sorted(unknown))}")
        if any(row.get("accession") != ref.accession for row in rows):
            raise ValueError("Every row must belong to the filing being saved")
        values = [[None if row.get(name) == "" else row.get(name) for name in COLUMNS] for row in rows]
        with self.connection:
            self.connection.execute("DELETE FROM ownership_rows WHERE accession = ?", (ref.accession,))
            self.connection.executemany(
                f"INSERT INTO ownership_rows ({NAMES}) VALUES ({', '.join(['?'] * len(COLUMNS))})", values)
            self.connection.execute(
                "INSERT INTO filings (accession, cik, form, filing_date, status) VALUES (?, ?, ?, ?, 'done') "
                "ON CONFLICT(accession) DO UPDATE SET status = 'done', error = NULL, "
                "cik = excluded.cik, form = excluded.form, filing_date = excluded.filing_date",
                (ref.accession, ref.cik, ref.form, ref.filing_date))

    def save_failure(self, ref, error) -> None:
        """Record a failed filing so a later run retries it."""
        with self.connection:
            self.connection.execute(
                "INSERT INTO filings (accession, cik, form, filing_date, status, attempts, error) "
                "VALUES (?, ?, ?, ?, 'failed', 1, ?) "
                "ON CONFLICT(accession) DO UPDATE SET status = 'failed', "
                "attempts = attempts + 1, error = excluded.error",
                (ref.accession, ref.cik, ref.form, ref.filing_date, str(error)))
