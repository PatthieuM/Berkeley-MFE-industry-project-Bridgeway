"""Synthetic checks for opt-in universe rules and atomic pointer publication."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from shared.universe import universe as legacy
from shared.universe import universe_v5 as u


def membership():
    row = {key: None for key in u.SECURITY_REQUIRED}
    row.update(permno=1, permco=10, secinfostartdt="2020-01-01", secinfoenddt="2020-12-31",
               primaryexch="N", securitytype="EQTY", securitysubtype="COM", sharetype="NS",
               issuertype="CORP", usincflg="Y", conditionaltype="RW", tradingstatusflg="A", siccd=2836)
    return u.build_membership_intervals(pd.DataFrame([row]), "2020-01-01", "2020-12-31")


def outputs():
    base = membership()
    master = u.build_classified_intervals(base)
    return {
        "source_membership_intervals": base,
        "master_universe_intervals": master,
        "biotech_universe_intervals": master.copy(),
        "biopharma_universe_intervals": master.copy(),
        "identifier_crosswalk": master.copy(),
        "identifier_link_candidates": u._prepare_crosswalk(None),
        "classification_history": u.build_classification_history(None, None),
        "delisting_history": u.prepare_delisting_history(None),
        "compustat_deletion_history": u.prepare_compustat_deletion_history(None),
    }


@pytest.mark.parametrize("value", ["2836.9", "2836.0000000000001", float("inf"), "nonsense", True])
def test_invalid_codes_are_not_rounded_into_biotech(value):
    with pytest.raises(ValueError):
        u._classification(None, value, None)


def test_zero_sic_has_consistent_missing_metadata():
    result = u._classification(None, 0, "0")
    assert result["classification_source"] == "UNCLASSIFIED"
    assert result["classification_rule"] == "NO_CLASSIFICATION"
    assert result["classification_code"] is None
    assert legacy._normalize_integer("2836.9") == 2836


@pytest.mark.parametrize("identifiers", [[1, None], [1, 1.5], ["abc"], ["1.0000000000000001"], [True], [2**63]])
def test_validator_rejects_invalid_identifiers(identifiers):
    frame = pd.DataFrame({"permno": identifiers, "effective_start": "2020-01-01", "effective_end": "2020-12-31"})
    with pytest.raises(ValueError):
        u.validate_universe(frame)


@pytest.mark.parametrize("column", ["link_start", "link_end"])
def test_malformed_crosswalk_bounds_are_not_open_ended(column):
    frame = pd.DataFrame({"permno": [1], "gvkey": ["A"], "link_start": [None], "link_end": [None]})
    assert len(u._prepare_crosswalk(frame)) == 1
    frame[column] = "invalid"
    with pytest.raises(ValueError, match="invalid date"):
        u._prepare_crosswalk(frame)


def test_overlapping_history_is_resolved_before_lookup():
    frame = pd.DataFrame({"permno": [1, 1], "gvkey": ["A", "A"],
                          "effective_date": ["2020-01-01", "2020-06-01"],
                          "effective_end": ["2020-12-31", "2020-06-30"],
                          "gics_code": ["35201010", "35202010"]})
    with pytest.raises(ValueError, match="overlapping"):
        u._prepare_history(frame, "gics_code", "GICS")
    frame["gics_code"] = "35201010"
    cleaned = u._prepare_history(frame, "gics_code", "GICS")
    assert len(cleaned) == 1
    assert u._latest_value(u._history_lookup(cleaned, "gics_code"), 1, "A", pd.Timestamp("2020-07-01")) == "35201010"


def test_missing_comparison_returns_boolean():
    assert u._same_value(pd.NA, "x") is False
    assert u._same_value(pd.NA, None) is True


def test_large_exact_identifiers_are_not_rounded_when_missing_values_present():
    values = u._identifiers(pd.Series(["9007199254740993", None]), "identifier")
    assert values.iloc[0] == 9007199254740993
    assert pd.isna(values.iloc[1])


@pytest.mark.parametrize("value", [20200101, 0, True])
def test_numeric_inputs_are_not_calendar_dates(value):
    with pytest.raises(ValueError, match="calendar"):
        u._dates(pd.Series([value]), "date")
    with pytest.raises(ValueError, match="calendar"):
        u._as_timestamp(value, "date")


@pytest.mark.parametrize("mutation", ["missing_view", "lineage", "crosswalk", "history", "candidates"])
def test_output_validation_detects_incomplete_or_inconsistent_tables(mutation):
    tables = outputs()
    assert u.validate_pipeline_outputs(tables)
    if mutation == "missing_view":
        tables["biotech_universe_intervals"] = tables["biotech_universe_intervals"].iloc[:0]
    elif mutation == "lineage":
        tables["master_universe_intervals"]["source_membership_id"] = None
    elif mutation == "crosswalk":
        tables["identifier_crosswalk"]["cik"] = "0000000009"
    elif mutation == "history":
        tables["classification_history"] = pd.DataFrame({"unexpected": ["anything"]})
    else:
        del tables["identifier_link_candidates"]
    with pytest.raises(ValueError):
        u.validate_pipeline_outputs(tables)


@pytest.mark.parametrize("mutation", ["blank_lineage", "source_attribute", "missing_start", "internal_gap", "invalid_deletion_date", "deletion_flag"])
def test_lineage_and_retrospective_output_validation(mutation):
    tables = outputs()
    if mutation == "blank_lineage":
        for name in ("source_membership_intervals", "master_universe_intervals", "biotech_universe_intervals", "biopharma_universe_intervals", "identifier_crosswalk"):
            tables[name]["source_membership_id"] = " "
    elif mutation == "source_attribute":
        tables["source_membership_intervals"]["permco"] = 999
    elif mutation in ("missing_start", "internal_gap"):
        for name in ("master_universe_intervals", "biotech_universe_intervals", "biopharma_universe_intervals", "identifier_crosswalk"):
            if mutation == "missing_start":
                tables[name]["effective_start"] = pd.Timestamp("2020-06-01")
            else:
                first = tables[name].copy()
                first["effective_end"] = pd.Timestamp("2020-05-31")
                second = tables[name].copy()
                second["effective_start"] = pd.Timestamp("2020-07-01")
                second["interval_id"] = "second"
                tables[name] = pd.concat([first, second], ignore_index=True)
    else:
        deletion = u.prepare_compustat_deletion_history(pd.DataFrame({
            "permno": [1], "compustat_deletion_date": ["2020-12-31"],
        }))
        if mutation == "invalid_deletion_date":
            deletion["compustat_deletion_date"] = "not-a-date"
        else:
            deletion["retrospective_only"] = False
        tables["compustat_deletion_history"] = deletion
    with pytest.raises(ValueError):
        u.validate_pipeline_outputs(tables)


def test_deletion_preparation_rejects_malformed_nonempty_dates():
    with pytest.raises(ValueError, match="invalid date"):
        u.prepare_compustat_deletion_history(pd.DataFrame({
            "permno": [1], "compustat_deletion_date": ["not-a-date"],
        }))


def test_publisher_preserves_old_version_and_stable_public_paths(tmp_path):
    target = tmp_path / "universe"
    tables = {"biotech_universe_intervals": pd.DataFrame({"permno": [1]})}
    u.write_outputs_atomic(tables, target, {})
    old_version = target.resolve()
    assert target.is_symlink()
    tables["biotech_universe_intervals"]["permno"] = 2
    u.write_outputs_atomic(tables, target, {}, overwrite=True)
    assert old_version.is_dir()
    assert target.resolve() != old_version
    assert pd.read_csv(target / "biotech_universe_intervals.csv").permno.tolist() == [2]
    assert json.loads((target / "manifest.json").read_text())["rule_version"] == u.RULE_VERSION


def test_failed_pointer_swap_leaves_previous_version_readable(tmp_path):
    target = tmp_path / "universe"
    tables = {"table": pd.DataFrame({"id": [1]})}
    u.write_outputs_atomic(tables, target, {})
    previous = target.resolve()
    with patch.object(u.os, "replace", side_effect=OSError("simulated publish failure")):
        with pytest.raises(OSError, match="simulated"):
            u.write_outputs_atomic(tables, target, {}, overwrite=True)
    assert target.resolve() == previous
    assert pd.read_csv(target / "table.csv").id.tolist() == [1]
    assert len(list((tmp_path / ".universe.versions").iterdir())) == 1


def test_publisher_cannot_replace_existing_legacy_directory(tmp_path):
    target = tmp_path / "universe"
    target.mkdir()
    (target / "sentinel").write_text("old result")
    with pytest.raises(ValueError, match="never replaces"):
        u.write_outputs_atomic({}, target, {}, overwrite=True)
    assert (target / "sentinel").read_text() == "old result"


def test_dry_run_validates_dates_without_authentication(capsys, tmp_path):
    with patch.object(u, "open_wrds_connection", side_effect=AssertionError("network prohibited")):
        assert u.main(["--dry-run", "--output-dir", str(tmp_path / "new")]) == 0
        with pytest.raises(ValueError):
            u.main(["--dry-run", "--start", "2025-01-01", "--end", "2020-01-01"])
    assert json.loads(capsys.readouterr().out)["rule_version"] == u.RULE_VERSION
