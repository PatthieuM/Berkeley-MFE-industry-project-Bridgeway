"""Synthetic classification changes and requested-window crosswalk extraction."""

from contextlib import contextmanager
import re
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from shared.universe import universe_v5 as u


def test_omitted_history_ends_follow_next_change_within_each_identifier_group():
    history = pd.DataFrame([
        {"permno": 1, "gvkey": "A", "effective_date": "2020-06-01", "compustat_sic": 2834},
        {"permno": 1, "gvkey": "A", "effective_date": "2020-01-01", "compustat_sic": 2836},
        {"permno": 1, "gvkey": "A", "effective_date": "2020-01-01", "compustat_sic": 2836},
        {"permno": 1, "gvkey": "B", "effective_date": "2020-04-01", "compustat_sic": 2836},
        {"permno": 2, "gvkey": None, "effective_date": "2020-03-01", "compustat_sic": 2836},
        {"permno": 2, "gvkey": None, "effective_date": "2020-07-01", "compustat_sic": 2834},
    ])
    cleaned = u._prepare_history(history, "compustat_sic", "SIC history")
    assert len(cleaned) == 5
    first = cleaned.loc[cleaned.gvkey.eq("A")].reset_index(drop=True)
    assert first.effective_end.iloc[0] == pd.Timestamp("2020-05-31")
    assert pd.isna(first.effective_end.iloc[1])
    assert cleaned.loc[cleaned.gvkey.eq("B"), "effective_end"].isna().all()
    missing_gvkey = cleaned.loc[cleaned.permno.eq(2)].reset_index(drop=True)
    assert missing_gvkey.effective_end.iloc[0] == pd.Timestamp("2020-06-30")
    assert pd.isna(missing_gvkey.effective_end.iloc[1])
    lookup = u._history_lookup(cleaned, "compustat_sic")
    assert u._latest_value(lookup, 1, "A", pd.Timestamp("2020-05-31")) == "2836"
    assert u._latest_value(lookup, 1, "A", pd.Timestamp("2020-06-01")) == "2834"


@pytest.mark.parametrize("end", [None, "2020-12-31"])
def test_explicit_overlapping_history_ends_are_not_inferred_away(end):
    history = pd.DataFrame({"permno": [1, 1], "gvkey": ["A", "A"],
                            "effective_date": ["2020-01-01", "2020-06-01"],
                            "effective_end": [end, None], "compustat_sic": [2836, 2834]})
    with pytest.raises(ValueError, match="overlapping"):
        u._prepare_history(history, "compustat_sic", "SIC history")


def test_conflicting_changes_on_the_same_date_still_fail_without_end_column():
    history = pd.DataFrame({"permno": [1, 1], "gvkey": ["A", "A"],
                            "effective_date": ["2020-01-01", "2020-01-01"],
                            "compustat_sic": [2836, 2834]})
    with pytest.raises(ValueError, match="one effective date"):
        u._prepare_history(history, "compustat_sic", "SIC history")


class SyntheticSqlConnection:
    """Execute the extraction query on SQLite after translating PostgreSQL syntax."""

    def __init__(self, connection):
        self.connection = connection

    @contextmanager
    def cursor(self):
        cursor = self.connection.cursor()
        class Cursor:
            def execute(inner, query, params):
                bound = {name: value.isoformat() for name, value in params.items() if name != "permnos"}
                identifiers = []
                for index, value in enumerate(params["permnos"]):
                    name = f"permno{index}"
                    identifiers.append(":" + name)
                    bound[name] = value
                query = query.replace("= any(%(permnos)s)", "IN (" + ", ".join(identifiers) + ")")
                query = re.sub(r"::(?:bigint|date|text)\b", "", query)
                query = re.sub(r"%\((\w+)\)s", r":\1", query)
                cursor.execute(query, bound)

            @property
            def description(inner):
                return [SimpleNamespace(name=column[0]) for column in cursor.description]

            def fetchall(inner):
                return cursor.fetchall()
        try:
            yield Cursor()
        finally:
            cursor.close()


def test_crosswalk_query_excludes_history_outside_request_and_keeps_boundary_overlap():
    connection = sqlite3.connect(":memory:")
    try:
        connection.create_function("greatest", -1, max)
        connection.create_function("least", -1, min)
        connection.executescript("""
            ATTACH DATABASE ':memory:' AS crsp;
            ATTACH DATABASE ':memory:' AS comp;
            CREATE TABLE crsp.ccmxpf_lnkhist
                (lpermno, lpermco, linkdt, linkenddt, gvkey, linkprim, linktype);
            CREATE TABLE crsp.comphist
                (gvkey, hcik, hconm, hchgdt, hchgenddt, hipodate, hdldte, hdlrsn);
            CREATE TABLE comp.company (gvkey, cik, conm);
            INSERT INTO crsp.ccmxpf_lnkhist VALUES (1, 10, '2000-01-01', '2030-12-31', 'A', 'P', 'LC');
            INSERT INTO comp.company VALUES ('A', '999', 'Current company');
        """)
        histories = [
            ("A", "101", "Before", "2010-01-01", "2015-12-31", None, None, None),
            ("A", "102", "Inside", "2019-01-01", "2022-12-31", None, None, None),
            ("A", "103", "After", "2026-01-01", "2029-12-31", None, None, None),
            ("A", "104", "Start boundary", "2019-01-01", "2020-01-01", None, None, None),
            ("A", "105", "End boundary", "2025-12-31", "2029-01-01", None, None, None),
        ]
        connection.executemany("INSERT INTO crsp.comphist VALUES (?, ?, ?, ?, ?, ?, ?, ?)", histories)
        result = u.extract_crosswalk(SyntheticSqlConnection(connection), [1],
                                     pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"))
        historical = result.loc[result.cik_source.eq("crsp.comphist_hcik")].set_index("cik")
        assert set(historical.index) == {"102", "104", "105"}
        assert historical.loc["102", "link_start"] == "2020-01-01"
        assert historical.loc["104", "link_start"] == historical.loc["104", "link_end"] == "2020-01-01"
        assert historical.loc["105", "link_start"] == historical.loc["105", "link_end"] == "2025-12-31"
        assert len(u._prepare_crosswalk(result)) == 4  # Three history rows plus current-company fallback.
    finally:
        connection.close()
