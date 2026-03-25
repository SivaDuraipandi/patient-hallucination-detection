"""
verify_downloads.py
Checks all 3 downloaded datasets are complete and correct.
Run after download_datasets.py completes.
"""

import os
import pandas as pd

RAW_DIR = "data/raw"

# ── Expected checks per file ──────────────────────────────────
CHECKS = {
    "pmc_patients.jsonl": {
        "min_rows"      : 100000,
        "min_size_mb"   : 200,
        "required_cols" : [
            "patient_uid", "patient",
            "age", "gender", "relevant_articles"
        ],
        "key_col"       : "patient",
        "label"         : "PMC-Patients V2"
    },
    "medhallu.jsonl": {
        "min_rows"      : 9000,
        "min_size_mb"   : 1,
        "required_cols" : [
            "question", "knowledge",
            "ground_truth", "hallucinated_ans",
            "difficulty", "hallucination_type",
            "split_source"
        ],
        "key_col"       : "question",
        "label"         : "MedHallu"
    },
    "pubmedqa.jsonl": {
        "min_rows"      : 900,
        "min_size_mb"   : 0.5,
        "required_cols" : [
            "pubid", "question",
            "context_text", "long_answer",
            "final_decision"
        ],
        "key_col"       : "question",
        "label"         : "PubMedQA"
    }
}

# ── Colour helpers (Windows-safe) ─────────────────────────────
def ok(msg):  return f"  [PASS] {msg}"
def err(msg): return f"  [FAIL] {msg}"
def wrn(msg): return f"  [WARN] {msg}"
def hdr(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

# ── Run all checks ────────────────────────────────────────────
all_passed = True

for filename, cfg in CHECKS.items():
    path = os.path.join(RAW_DIR, filename)
    hdr(f"Checking: {cfg['label']}")

    # 1. File exists?
    if not os.path.exists(path):
        print(err(f"File not found: {path}"))
        print(wrn("Re-run download_datasets.py"))
        all_passed = False
        continue

    # 2. File size
    size_mb = os.path.getsize(path) / 1e6
    if size_mb < cfg["min_size_mb"]:
        print(err(f"File too small: {size_mb:.1f} MB "
                  f"(expected >= {cfg['min_size_mb']} MB)"))
        all_passed = False
    else:
        print(ok(f"File size      : {size_mb:.1f} MB"))

    # 3. Load file
    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        print(err(f"Cannot read file: {e}"))
        all_passed = False
        continue

    # 4. Row count
    if len(df) < cfg["min_rows"]:
        print(err(f"Row count      : {len(df):,} "
                  f"(expected >= {cfg['min_rows']:,})"))
        all_passed = False
    else:
        print(ok(f"Row count      : {len(df):,}"))

    # 5. Required columns
    missing_cols = [
        c for c in cfg["required_cols"] if c not in df.columns
    ]
    if missing_cols:
        print(err(f"Missing columns: {missing_cols}"))
        all_passed = False
    else:
        print(ok(f"All columns    : present "
                 f"({len(cfg['required_cols'])} checked)"))

    # 6. Null check on key column
    null_count = df[cfg["key_col"]].isna().sum()
    if null_count > 0:
        print(wrn(f"Nulls in '{cfg['key_col']}': {null_count}"))
    else:
        print(ok(f"No nulls in    : '{cfg['key_col']}'"))

    # 7. Empty string check
    empty_count = (
        df[cfg["key_col"]].astype(str).str.strip() == ""
    ).sum()
    if empty_count > 0:
        print(wrn(f"Empty strings  : {empty_count} "
                  f"in '{cfg['key_col']}'"))
    else:
        print(ok(f"No empty strings in '{cfg['key_col']}'"))

    # 8. Dataset-specific checks
    if filename == "medhallu.jsonl":
        expert = (df["split_source"] == "expert").sum()
        artif  = (df["split_source"] == "artificial").sum()
        print(ok(f"Expert split   : {expert:,}"))
        print(ok(f"Artificial     : {artif:,}"))
        diff_counts = df["difficulty"].value_counts().to_dict()
        print(ok(f"Difficulty     : {diff_counts}"))

    if filename == "pubmedqa.jsonl":
        dec_counts = df["final_decision"].value_counts().to_dict()
        print(ok(f"Decisions      : {dec_counts}"))

    if filename == "pmc_patients.jsonl":
        avg_len = df["patient"].str.len().mean()
        print(ok(f"Avg patient text length : {avg_len:.0f} chars"))

    # 9. Sample row preview
    print(f"\n  Sample row preview ({cfg['label']}):")
    sample = df.iloc[0]
    for col in cfg["required_cols"][:3]:
        val = str(sample.get(col, "N/A"))
        val = val[:80] + "..." if len(val) > 80 else val
        print(f"    {col:25s}: {val}")

# ── Final result ──────────────────────────────────────────────
print(f"\n{'='*55}")
if all_passed:
    print("  ALL CHECKS PASSED")
    print("  Ready for: python src/dataset/preprocess.py")
else:
    print("  SOME CHECKS FAILED")
    print("  Re-run download_datasets.py before proceeding")
print(f"{'='*55}\n")