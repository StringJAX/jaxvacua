# Copyright 2024 Andreas Schachner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
r"""
Tests for the ``jaxvacua.vacua_vault`` subpackage:

- Schema / parquet-level validation (``validate_parquet_file``).
- Row-level split for ``partial_failure="split"`` mode.
- Catalog regeneration from a synthetic repo tree.
- Community → curated promotion (file move + attribution preservation).
- CLI dispatch via ``python -m jaxvacua.vacua_vault``.

These tests build a fake local vault repo under a ``tempfile.mkdtemp()``
so no network access is required.  They do not test
``push_vacua_to_hub``'s upload step (which needs HF credentials).
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append("./../")

from jaxvacua.vacua_vault import (
    validate_parquet_file,
    split_by_validation,
    rebuild_catalog,
    curate_submission,
    SCHEMA_VERSION,
    RESERVED_NAMES,
    LABEL_SLUG_RE,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------

def _make_valid_df(n: int = 5, h12: int = 2) -> pd.DataFrame:
    n_fluxes = 2 * (h12 + 1)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "flux":      [[int(x) for x in rng.integers(-3, 4, 2 * n_fluxes)] for _ in range(n)],
        "moduli_re": [[1.0 + i * 0.1, 2.0] for i in range(n)],
        "moduli_im": [[3.0, 4.0 + i * 0.05] for i in range(n)],
        "tau_re":    [0.0] * n,
        "tau_im":    [1.0 + i * 0.01 for i in range(n)],
        "is_susy":   [False] * n,
    })


def _write_parquet_with_metadata(df: pd.DataFrame, path: Path,
                                 metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing = dict(table.schema.metadata or {})
    existing.update({k.encode(): v.encode() for k, v in metadata.items()})
    table = table.replace_schema_metadata(existing)
    pq.write_table(table, path)


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):
    r"""Tests for :func:`jaxvacua.vacua_vault.validate_parquet_file`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jvc_vault_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_label_slug_regex(self):
        r"""Accepted vs rejected dataset labels per the slug regex (§5.6)."""
        for ok in ["SUSY_Nmax34", "racetrack_small_W0", "a", "a_v2",
                   "a-b", "A1B2"]:
            self.assertTrue(LABEL_SLUG_RE.match(ok),
                            f"should accept {ok!r}")
        for bad in ["", "_leading_underscore", "has space",
                    "path/sep", "bad!char"]:
            self.assertFalse(LABEL_SLUG_RE.match(bad),
                             f"should reject {bad!r}")

    def test_valid_file_passes(self):
        r"""A well-formed parquet with correct metadata passes validation."""
        df = _make_valid_df()
        p = self.tmp / "tdf" / "h12_2" / "ks_29_tri_0" / "SUSY_Nmax34.parquet"
        _write_parquet_with_metadata(df, p, {
            "schema_version": str(SCHEMA_VERSION),
            "susy": "SUSY", "method": "enumerate", "nmax": "34",
        })
        result = validate_parquet_file(p)
        self.assertTrue(result["passed"], f"errors: {result['errors']}")
        self.assertEqual(result["n_rows"], 5)
        self.assertEqual(result["n_failed"], 0)
        self.assertEqual(result["metadata"]["susy"], "SUSY")

    def test_reserved_filename_rejected(self):
        r"""Reserved top-level names (schema.json, README.md, ...) fail
        even if their content is otherwise valid parquet."""
        df = _make_valid_df()
        p = self.tmp / "catalog.parquet"
        _write_parquet_with_metadata(df, p, {
            "schema_version": str(SCHEMA_VERSION),
        })
        result = validate_parquet_file(p)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Reserved" in e for e in result["errors"]))

    def test_schema_version_mismatch_flagged(self):
        r"""A parquet with a different schema_version in its metadata
        is rejected with a clear error."""
        df = _make_valid_df()
        p = self.tmp / "tdf" / "h12_2" / "ks_29_tri_0" / "mislabel.parquet"
        _write_parquet_with_metadata(df, p, {
            "schema_version": str(SCHEMA_VERSION + 99),
        })
        result = validate_parquet_file(p)
        self.assertFalse(result["passed"])
        self.assertTrue(any("schema_version" in e for e in result["errors"]))

    def test_non_integer_flux_flagged(self):
        r"""Rows with non-integer flux entries fail per-row validation."""
        df = _make_valid_df(3)
        df.at[1, "flux"] = [1.5, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]  # non-integer
        p = self.tmp / "tdf" / "h12_2" / "ks_29_tri_0" / "bad.parquet"
        _write_parquet_with_metadata(df, p, {
            "schema_version": str(SCHEMA_VERSION),
        })
        result = validate_parquet_file(p)
        self.assertGreaterEqual(result["n_failed"], 1)


class TestSplitByValidation(unittest.TestCase):
    r"""Tests for the ``partial_failure="split"`` row-splitting helper."""

    def test_split_preserves_counts(self):
        df = _make_valid_df(10)
        report = [
            {"index": i, "passed": (i % 2 == 0),
             "errors": [] if (i % 2 == 0) else [f"err-{i}"]}
            for i in range(len(df))
        ]
        valid, rejected = split_by_validation(df, report)
        self.assertEqual(len(valid), 5)
        self.assertEqual(len(rejected), 5)
        self.assertIn("error", rejected.columns)

    def test_all_valid_no_rejected(self):
        df = _make_valid_df(3)
        report = [{"index": i, "passed": True, "errors": []}
                  for i in range(3)]
        valid, rejected = split_by_validation(df, report)
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(rejected), 0)


class TestRebuildCatalog(unittest.TestCase):
    r"""Tests for :func:`rebuild_catalog`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jvc_vault_cat_"))
        # Layout: 2 curated + 1 community + 1 rejected
        self._mk(
            "tdf/h12_2/ks_29_tri_0/SUSY_Nmax34.parquet",
            {"susy": "SUSY", "method": "enumerate", "nmax": "34"},
        )
        self._mk(
            "tdf/h12_2/ks_29_tri_0/racetrack_small_W0.parquet",
            {"susy": "mixed", "method": "hybrid", "nmax": "200"},
        )
        self._mk(
            "tdf/h12_2/ks_29_tri_0/community/alice-hf_dS_candidates.parquet",
            {"susy": "nonSUSY", "method": "ISD-", "nmax": "200",
             "committed_by": "alice"},
        )
        self._mk(
            "tdf/h12_2/ks_29_tri_0/community/_rejected/bob-hf_bad.rejected.parquet",
            {"susy": "nonSUSY"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk(self, rel: str, metadata: dict):
        df = _make_valid_df(3)
        _write_parquet_with_metadata(
            df, self.tmp / rel,
            {**metadata, "schema_version": str(SCHEMA_VERSION)},
        )

    def test_catalog_covers_all_files(self):
        out = rebuild_catalog(self.tmp, verbose=False)
        self.assertTrue(out.exists())
        cat = pd.read_parquet(out)
        # All four files must show up.
        self.assertEqual(len(cat), 4)

    def test_catalog_status_inferred_from_path(self):
        rebuild_catalog(self.tmp, verbose=False)
        cat = pd.read_parquet(self.tmp / "catalog.parquet")
        status_by_name = dict(zip(cat["basename"], cat["status"]))
        self.assertEqual(status_by_name["SUSY_Nmax34.parquet"], "curated")
        self.assertEqual(
            status_by_name["alice-hf_dS_candidates.parquet"], "pending",
        )
        self.assertEqual(
            status_by_name["bob-hf_bad.rejected.parquet"], "rejected",
        )

    def test_catalog_classification_from_metadata(self):
        rebuild_catalog(self.tmp, verbose=False)
        cat = pd.read_parquet(self.tmp / "catalog.parquet")
        row = cat[cat["basename"] == "racetrack_small_W0.parquet"].iloc[0]
        self.assertEqual(row["susy"], "mixed")
        self.assertEqual(row["method"], "hybrid")
        self.assertEqual(row["nmax"], 200)
        self.assertEqual(row["classification_source"], "parquet_metadata")


class TestCurateSubmission(unittest.TestCase):
    r"""Tests for :func:`curate_submission`."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="jvc_vault_curate_"))
        self.community_file = (
            self.tmp / "tdf" / "h12_2" / "ks_29_tri_0" / "community"
            / "alice-hf_racetrack_candidates.parquet"
        )
        _write_parquet_with_metadata(
            _make_valid_df(3), self.community_file,
            {"schema_version": str(SCHEMA_VERSION),
             "susy": "nonSUSY", "method": "ISD-"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_moves_to_model_level(self):
        target = curate_submission(
            self.community_file, self.tmp, verbose=False,
        )
        self.assertFalse(self.community_file.exists(),
                         "community file should have been moved")
        self.assertTrue(target.exists())
        # New path: prefix stripped
        self.assertEqual(target.name, "racetrack_candidates.parquet")
        self.assertEqual(
            target.parent.name, "ks_29_tri_0",
            "file should sit at model-level directory",
        )

    def test_attribution_preserved_in_metadata(self):
        target = curate_submission(
            self.community_file, self.tmp, verbose=False,
        )
        md = pq.read_table(target).schema.metadata or {}
        md_dict = {k.decode(): v.decode() for k, v in md.items()}
        self.assertEqual(md_dict.get("original_contributor"), "alice-hf")
        self.assertEqual(md_dict.get("status"), "curated")

    def test_dry_run_does_not_move(self):
        projected = curate_submission(
            self.community_file, self.tmp, dry_run=True, verbose=False,
        )
        self.assertTrue(self.community_file.exists(),
                        "dry_run must not modify the filesystem")
        self.assertEqual(projected.name, "racetrack_candidates.parquet")

    def test_non_community_path_rejected(self):
        not_in_community = (
            self.tmp / "tdf" / "h12_2" / "ks_29_tri_0" / "at_model_level.parquet"
        )
        _write_parquet_with_metadata(
            _make_valid_df(1), not_in_community,
            {"schema_version": str(SCHEMA_VERSION)},
        )
        with self.assertRaises(ValueError):
            curate_submission(not_in_community, self.tmp, verbose=False)


class TestCLIEntryPoint(unittest.TestCase):
    r"""Smoke test that ``python -m jaxvacua.vacua_vault`` dispatches."""

    def test_help_exits_clean(self):
        res = subprocess.run(
            [sys.executable, "-m", "jaxvacua.vacua_vault", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(res.returncode, 0, f"stderr: {res.stderr}")
        self.assertIn("validate", res.stdout)
        self.assertIn("rebuild_catalog", res.stdout)
        self.assertIn("curate", res.stdout)


if __name__ == "__main__":
    unittest.main()
