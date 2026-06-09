"""Build the v3 endpoint codeset JSON via SNOMED descendant traversal in BigQuery.

Per docs/v3_spec.md §5.1: v2 used `condition_source_concept_id LIKE 'E11%'` —
that undercounts T2D and depression by 20-40% at SNOMED-shop sites. v3 uses
`concept_ancestor` to traverse a SNOMED root through all descendants, unions
with ICD-10 leaf codes, and writes one entry per endpoint per evidence type.

Multi-source phenotypes (Dx + Lab + Rx) for incident-disease endpoints follow
the eMERGE/PheKB pattern: a participant is "first-recorded" at the earliest
date of ANY evidence type. Baseline exclusion uses the same logic over the
lookback period.

Output: ~/workspace/phonefm-data/endpoint_concept_ids_v3.json
Schema:
  {
    "endpoints": {
      "mortality":     { "death_table_only": true },
      "t2d":           { "dx": [..ids..], "lab": [..ids..], "drug": [..ids..] },
      "dep":           { "dx": [..ids..], "drug": [..ids..] },
      "cv_composite":  { "dx_afib": [...], "dx_mi": [...], "dx_hf": [...] },
      "skin_neoplasm": { "dx": [..] },
      "refractive_errors": { "dx": [..] },
      "dental_caries": { "dx": [..] }
    },
    "baseline_codes": {
      "t2d": {"dx": [..], "lab": [..], "drug": [..]},
      "dep": {"dx": [..], "drug": [..]},
      "osa": {"dx": [..]},                                  # confounder (§2.1)
      "afib": {"dx": [..]},                                 # for prior_afib confounder
      "cad":  {"dx": [..]},                                 # baseline_cad confounder
      "cancer": {"dx": [..]},                               # baseline_cancer confounder
      "skin_neoplasm": {"dx": [..]},
      "refractive_errors": {"dx": [..]},
      "dental_caries": {"dx": [..]}
    },
    "_self_test": { ... per-endpoint hit counts on full cohort, per evidence type }
  }

Run from workbench:
    cd ~/repos/PhoneFM/workbench && python3 build_endpoint_concept_ids_v3.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-not-found]

# AoU env vars (populated by ~/load-env.sh on Verily Workbench)
CDR = os.environ["WORKSPACE_CDR"]  # e.g. "wb-silky-artichoke-2408.C2024Q3R8"
OUT_PATH = Path("/home/jupyter/workspace/phonefm-data/endpoint_concept_ids_v3.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# SNOMED root concepts and matching ICD-10 prefixes per docs/v3_spec.md §2 and §5.1.
# Sources: AthenaOHDSI concept IDs verified against AoU public concept lookup,
# 2026-06-09. If any of these become stale, the self-test at the end will flag
# zero hits and the file should be rebuilt.
ENDPOINT_DEFS: dict[str, dict[str, Any]] = {
    # Primary disease endpoints
    "t2d": {
        "dx": {
            "snomed_roots": [201826],          # Type 2 diabetes mellitus
            "icd10_prefixes": ["E11"],
        },
        "lab": {
            # HbA1c >=6.5, fasting glucose >=126, random glucose >=200. The
            # threshold logic lives in the tokenizer; here we just collect IDs.
            "loinc_codes": ["4548-4", "1558-6", "2345-7"],
        },
        "drug": {
            # RxNorm class "Antidiabetic Agents" — descendant of ATC A10
            "rxnorm_ancestors": [21600712],
        },
    },
    "dep": {
        "dx": {
            "snomed_roots": [192080],          # Depressive disorder
            "icd10_prefixes": ["F32", "F33"],
        },
        "drug": {
            "rxnorm_ancestors": [21604686],    # Antidepressants
        },
    },
    "cv_composite": {
        "dx_afib": {
            "snomed_roots": [313217],          # Atrial fibrillation
            "icd10_prefixes": ["I48"],
        },
        "dx_mi": {
            "snomed_roots": [4329847],         # Myocardial infarction
            "icd10_prefixes": ["I21", "I22"],
        },
        "dx_hf": {
            "snomed_roots": [316139],          # Heart failure
            "icd10_prefixes": ["I50"],
        },
    },
    # Negative controls (§11.2)
    "skin_neoplasm": {
        "dx": {
            "snomed_roots": [4112752],         # Malignant neoplasm of skin
            "icd10_prefixes": ["C43", "C44"],
        },
    },
    "refractive_errors": {
        "dx": {
            "snomed_roots": [4218554],         # Refractive disorder
            "icd10_prefixes": ["H52"],
        },
    },
    "dental_caries": {
        "dx": {
            "snomed_roots": [4210708],         # Dental caries
            "icd10_prefixes": ["K02"],
        },
    },
    # Baseline-only (confounders, not heads)
    "osa": {
        "dx": {
            "snomed_roots": [4154290],         # Obstructive sleep apnea
            "icd10_prefixes": ["G47.33", "G47.3"],
        },
    },
    "cad": {
        "dx": {
            "snomed_roots": [317576],          # Coronary arteriosclerosis
            "icd10_prefixes": ["I25"],
        },
    },
    "cancer": {
        "dx": {
            # Malignant neoplasm root (broad — captures all cancers for the
            # baseline_cancer confounder; skin separately handled above)
            "snomed_roots": [443392],
            "icd10_prefixes": ["C"],
        },
    },
}


def _bq_to_df(sql: str) -> pd.DataFrame:
    import pandas_gbq  # type: ignore[import-not-found]
    return pandas_gbq.read_gbq(sql, dialect="standard", progress_bar_type=None)


def fetch_dx_descendants(snomed_roots: list[int], icd10_prefixes: list[str]) -> list[int]:
    """Union of SNOMED descendants + ICD-10-CM concepts matching the given prefixes.

    Returns standard `concept_id`s suitable for filtering
    `condition_occurrence.condition_concept_id IN UNNEST(...)`.
    """
    if not snomed_roots and not icd10_prefixes:
        return []
    snomed_clause = (
        f"c.concept_id IN UNNEST({snomed_roots})" if snomed_roots else "FALSE"
    )
    icd_clause = (
        " OR ".join(
            f"(c.vocabulary_id = 'ICD10CM' AND c.concept_code LIKE '{p}%')"
            for p in icd10_prefixes
        )
        if icd10_prefixes else "FALSE"
    )
    sql = f"""
        SELECT DISTINCT ca.descendant_concept_id AS cid
        FROM `{CDR}.concept_ancestor` ca
        JOIN `{CDR}.concept` c
          ON ca.ancestor_concept_id = c.concept_id
        WHERE {snomed_clause} OR {icd_clause}
    """
    df = _bq_to_df(sql)
    return sorted(int(x) for x in df["cid"].tolist())


def fetch_measurement_concepts_by_loinc(loinc_codes: list[str]) -> list[int]:
    """Return standard concept_ids for the given LOINC codes."""
    if not loinc_codes:
        return []
    sql = f"""
        SELECT DISTINCT c.concept_id AS cid
        FROM `{CDR}.concept` c
        WHERE c.vocabulary_id = 'LOINC'
          AND c.concept_code IN UNNEST({loinc_codes})
    """
    df = _bq_to_df(sql)
    return sorted(int(x) for x in df["cid"].tolist())


def fetch_drug_descendants(rxnorm_ancestors: list[int]) -> list[int]:
    """Union of RxNorm class descendants — for drug_exposure.drug_concept_id."""
    if not rxnorm_ancestors:
        return []
    sql = f"""
        SELECT DISTINCT ca.descendant_concept_id AS cid
        FROM `{CDR}.concept_ancestor` ca
        WHERE ca.ancestor_concept_id IN UNNEST({rxnorm_ancestors})
    """
    df = _bq_to_df(sql)
    return sorted(int(x) for x in df["cid"].tolist())


def count_persons_with_evidence(concept_ids: list[int], table: str, id_col: str) -> int:
    """Self-test helper: count distinct persons with at least one row in the given
    table for the given concept_ids. Used to validate that codesets are nonempty
    and meaningful."""
    if not concept_ids:
        return 0
    sql = f"""
        SELECT COUNT(DISTINCT person_id) AS n
        FROM `{CDR}.{table}`
        WHERE {id_col} IN UNNEST({concept_ids})
    """
    df = _bq_to_df(sql)
    return int(df["n"].iloc[0])


def build_codesets() -> dict[str, Any]:
    """Resolve all SNOMED descendants + LOINC/RxNorm lists into concrete concept_ids."""
    resolved: dict[str, dict[str, list[int]]] = {}

    for ep, sources in ENDPOINT_DEFS.items():
        resolved[ep] = {}
        for src_key, src_spec in sources.items():
            if src_key.startswith("dx") or src_key == "dx":
                ids = fetch_dx_descendants(
                    src_spec.get("snomed_roots", []),
                    src_spec.get("icd10_prefixes", []),
                )
            elif src_key == "lab":
                ids = fetch_measurement_concepts_by_loinc(
                    src_spec.get("loinc_codes", [])
                )
            elif src_key == "drug":
                ids = fetch_drug_descendants(
                    src_spec.get("rxnorm_ancestors", [])
                )
            else:
                raise ValueError(f"unknown source key: {src_key}")
            resolved[ep][src_key] = ids
            print(f"  {ep}.{src_key}: {len(ids)} concept_ids", flush=True)

    return resolved


def self_test(resolved: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, int]]:
    """Count distinct persons in the FULL CDR (not yet cohort-filtered) with at least
    one row matching each (endpoint, source). Warns on any source contributing <5%
    of the endpoint's total hits — that source likely has a wrong concept ID.

    NB: This runs against the full CDR, not the PhoneFM cohort. Cohort-level
    sanity checking happens in the tokenizer.
    """
    out: dict[str, dict[str, int]] = {}
    table_map = {
        "dx": ("condition_occurrence", "condition_concept_id"),
        "lab": ("measurement", "measurement_concept_id"),
        "drug": ("drug_exposure", "drug_concept_id"),
    }
    for ep, sources in resolved.items():
        per_source = {}
        for src_key, ids in sources.items():
            # Resolve dx_* sub-keys to dx table
            base_key = "dx" if src_key.startswith("dx") else src_key
            if base_key not in table_map:
                continue
            table, id_col = table_map[base_key]
            n = count_persons_with_evidence(ids, table, id_col)
            per_source[src_key] = n
            print(f"  self_test  {ep}.{src_key}: {n:,} persons", flush=True)
        out[ep] = per_source
        # Warn if any single source is <5% of the union
        total = max(sum(per_source.values()), 1)
        for src_key, n in per_source.items():
            if n / total < 0.05 and n > 0:
                print(f"  WARNING  {ep}.{src_key} contributes <5% ({n}/{total} = "
                      f"{100*n/total:.1f}%) — codeset may be wrong", flush=True)
            elif n == 0:
                print(f"  WARNING  {ep}.{src_key} matched ZERO persons — likely broken",
                      flush=True)
    return out


def main() -> None:
    print(f"CDR: {CDR}", flush=True)
    print(f"OUT: {OUT_PATH}", flush=True)
    print("\nresolving codesets...", flush=True)
    resolved = build_codesets()

    print("\nself-test (cohort-wide person counts)...", flush=True)
    test_results = self_test(resolved)

    # Split into endpoint-vs-baseline view per the JSON schema in the module docstring
    endpoint_keys = [
        "t2d", "dep", "cv_composite", "skin_neoplasm",
        "refractive_errors", "dental_caries",
    ]
    # afib for prior_afib confounder: extract from cv_composite.dx_afib
    afib_for_baseline = resolved.get("cv_composite", {}).get("dx_afib", [])

    output = {
        "_generated_at": pd.Timestamp.now().isoformat(),
        "_cdr": CDR,
        "_spec_section": "docs/v3_spec.md §5.1",
        "endpoints": {
            "mortality": {"death_table_only": True},
            **{k: resolved.get(k, {}) for k in endpoint_keys},
        },
        "baseline_codes": {
            "t2d": resolved.get("t2d", {}),
            "dep": resolved.get("dep", {}),
            "osa": resolved.get("osa", {}),
            "cad": resolved.get("cad", {}),
            "cancer": resolved.get("cancer", {}),
            "afib": {"dx": afib_for_baseline},  # for prior_afib confounder
            "skin_neoplasm": resolved.get("skin_neoplasm", {}),
            "refractive_errors": resolved.get("refractive_errors", {}),
            "dental_caries": resolved.get("dental_caries", {}),
        },
        "_self_test": test_results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nwrote {OUT_PATH}", flush=True)
    print(f"size: {OUT_PATH.stat().st_size / 1024:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
