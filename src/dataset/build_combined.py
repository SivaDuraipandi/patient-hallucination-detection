"""
build_combined.py
=================
Takes matched_pairs.jsonl (output of match_datasets.py)
and builds the final custom dataset.

What this script does:
  1. For each matched pair creates TWO records:
       Type A — hallucinated answer  (is_hallucinated = 1)
       Type B — correct answer       (is_hallucinated = 0)
     This gives a perfectly balanced 50/50 dataset.

  2. Applies quality checks:
       - Remove nulls in critical columns
       - Remove duplicate (question + answer) pairs
       - Verify 50/50 label balance
       - Shuffle with fixed seed for reproducibility

  3. Stratified train / val / test split:
       - Expert-labeled rows  → test ONLY (never training)
       - Artificial rows      → train + val
       - Stratified by difficulty to preserve tier distribution

  4. Saves all outputs:
       data/combined/full_dataset.jsonl   (all pairs)
       data/splits/train.jsonl
       data/splits/val.jsonl
       data/splits/test.jsonl
       data/splits/pubmedqa_baseline.jsonl (500 rows)
       data/combined/dataset_report.json

  5. Prints a final summary with all stats you need
     for your paper's dataset section.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────
COMBINED_DIR = "data/combined"
SPLITS_DIR   = "data/splits"
RAW_DIR      = "data/raw"

os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR,   exist_ok=True)

# ── Config ────────────────────────────────────────────────────
RANDOM_SEED    = 42
VAL_SIZE       = 0.18   # 18% of artificial rows → val
MIN_TEXT_LEN   = 10     # minimum chars for question/answer

# Trust score mapping — used as regression target
TRUST_SCORE_MAP = {
    "easy"    : 0.9,
    "medium"  : 0.6,
    "hard"    : 0.3,
    "unknown" : 0.5,
}

np.random.seed(RANDOM_SEED)


def log(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD MATCHED PAIRS
# ─────────────────────────────────────────────────────────────
def load_matched_pairs():
    log("Loading matched_pairs.jsonl...")

    path = os.path.join(COMBINED_DIR, "matched_pairs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"matched_pairs.jsonl not found at {path}\n"
            f"Run match_datasets.py first."
        )

    df = pd.read_json(path, lines=True)
    print(f"  Loaded : {len(df):,} matched pairs")
    print(f"  Cols   : {list(df.columns)}")

    # Show match type distribution
    if "match_type" in df.columns:
        print(f"  Match types:")
        for mt, cnt in df["match_type"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {mt:30s}: {cnt:,}  ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — BUILD POSITIVE + NEGATIVE PAIRS
# ─────────────────────────────────────────────────────────────
def build_pairs(matched_df):
    """
    For every matched row create two records:
      Type A: answer = hallucinated_ans → is_hallucinated = 1
      Type B: answer = ground_truth     → is_hallucinated = 0

    This automatically gives a perfectly balanced dataset.
    trust_score = 1.0 for correct answers (full trust)
    trust_score = difficulty-based for hallucinated answers
    """
    log("Building positive + negative pairs...")

    records = []
    skipped = 0

    for _, row in matched_df.iterrows():

        # Shared fields for both record types
        base = {
            # Patient-aware context
            "matched_patient_uid" : str(
                row.get("matched_patient_uid", "")),
            "patient_context"     : str(
                row.get("patient_context", "")),
            "patient_age"         : str(
                row.get("patient_age", "unknown")),
            "patient_gender"      : str(
                row.get("patient_gender", "unknown")),

            # Question and knowledge context
            "question"            : str(
                row.get("question", "")),
            "knowledge"           : str(
                row.get("knowledge", "")),
            "ground_truth"        : str(
                row.get("ground_truth", "")),

            # MedHallu metadata
            "difficulty"          : str(
                row.get("difficulty", "unknown")),
            "hallucination_type"  : str(
                row.get("hallucination_type", "unknown")),
            "split_source"        : str(
                row.get("split_source", "artificial")),

            # Match metadata (kept for analysis)
            "match_type"          : str(
                row.get("match_type", "")),
            "cosine_similarity"   : float(
                row.get("cosine_similarity", 0.0)),
            "bridge_cosine"       : float(
                row.get("bridge_cosine", 0.0)),
        }

        # Validate minimum field lengths
        if (len(base["question"])        < MIN_TEXT_LEN or
            len(base["ground_truth"])    < MIN_TEXT_LEN or
            len(base["patient_context"]) < MIN_TEXT_LEN):
            skipped += 1
            continue

        hallucinated_ans = str(
            row.get("hallucinated_ans", ""))
        if len(hallucinated_ans) < MIN_TEXT_LEN:
            skipped += 1
            continue

        difficulty   = base["difficulty"]
        trust_hallu  = TRUST_SCORE_MAP.get(difficulty, 0.5)

        # ── TYPE A: Hallucinated answer ───────────────────────
        record_a = base.copy()
        record_a["answer"]           = hallucinated_ans
        record_a["is_hallucinated"]  = 1
        record_a["trust_score"]      = trust_hallu
        record_a["record_type"]      = "hallucinated"
        records.append(record_a)

        # ── TYPE B: Correct answer ────────────────────────────
        record_b = base.copy()
        record_b["answer"]           = base["ground_truth"]
        record_b["is_hallucinated"]  = 0
        record_b["trust_score"]      = 1.0
        record_b["record_type"]      = "correct"
        records.append(record_b)

    combined_df = pd.DataFrame(records)

    print(f"  Input rows      : {len(matched_df):,}")
    print(f"  Skipped         : {skipped:,} "
          f"(too short / missing fields)")
    print(f"  Output records  : {len(combined_df):,} "
          f"(2 per input row)")
    print(f"  Label split     :")
    print(f"    is_hallucinated=1 : "
          f"{(combined_df.is_hallucinated==1).sum():,}")
    print(f"    is_hallucinated=0 : "
          f"{(combined_df.is_hallucinated==0).sum():,}")

    return combined_df


# ─────────────────────────────────────────────────────────────
# STEP 3 — QUALITY CHECKS
# ─────────────────────────────────────────────────────────────
def quality_checks(df):
    log("Running quality checks...")

    before = len(df)

    # 1. Drop nulls in critical columns
    critical = ["question", "answer",
                "ground_truth", "patient_context",
                "is_hallucinated"]
    df = df.dropna(subset=critical)
    print(f"  After null removal     : {len(df):,} rows "
          f"(removed {before - len(df):,})")

    # 2. Drop empty strings in critical columns
    before2 = len(df)
    for col in ["question", "answer", "patient_context"]:
        df = df[df[col].str.strip().str.len()
                >= MIN_TEXT_LEN]
    print(f"  After empty removal    : {len(df):,} rows "
          f"(removed {before2 - len(df):,})")

    # 3. Remove duplicate (question + answer) pairs
    before3 = len(df)
    df = df.drop_duplicates(subset=["question", "answer"])
    print(f"  After deduplication   : {len(df):,} rows "
          f"(removed {before3 - len(df):,} duplicates)")

    # 4. Verify label balance
    label_counts = df["is_hallucinated"].value_counts()
    ratio = (label_counts.get(1, 0) /
             max(label_counts.get(0, 1), 1))
    print(f"  Label balance         :")
    print(f"    Hallucinated (1) : "
          f"{label_counts.get(1, 0):,}")
    print(f"    Correct      (0) : "
          f"{label_counts.get(0, 0):,}")
    print(f"    Ratio (1/0)      : {ratio:.3f} "
          f"(1.0 = perfect balance)")
    if abs(ratio - 1.0) > 0.05:
        print(f"  WARNING: ratio {ratio:.3f} deviates "
              f"from 1.0 — check pair building")

    # 5. Verify difficulty distribution
    print(f"  Difficulty distribution:")
    for d, cnt in df["difficulty"].value_counts().items():
        print(f"    {d:10s}: {cnt:,}")

    # 6. Verify trust score distribution
    print(f"  Trust score distribution:")
    for t, cnt in df["trust_score"].value_counts().items():
        print(f"    {t:.1f} : {cnt:,}")

    # 7. Shuffle with fixed seed for reproducibility
    df = df.sample(frac=1, random_state=RANDOM_SEED)\
           .reset_index(drop=True)
    print(f"  Shuffled with seed    : {RANDOM_SEED}")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────
def split_dataset(df):
    """
    Split rules:
      expert rows (split_source='expert')
        → TEST ONLY — human-annotated gold standard
      artificial rows
        → TRAIN (82%) + VAL (18%)
          stratified by difficulty to preserve tier ratio
    """
    log("Creating train / val / test splits...")

    # Separate expert vs artificial
    expert_mask  = df["split_source"] == "expert"
    test_df      = df[expert_mask].copy()
    trainval_df  = df[~expert_mask].copy()

    print(f"  Expert (test only)  : {len(test_df):,} rows")
    print(f"  Artificial (train+val): {len(trainval_df):,} rows")

    # Stratified split on difficulty
    # Handle edge case where a difficulty class is too small
    try:
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=VAL_SIZE,
            stratify=trainval_df["difficulty"],
            random_state=RANDOM_SEED
        )
    except ValueError:
        # Fallback: split without stratification
        print("  WARNING: stratified split failed "
              "— using random split")
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=VAL_SIZE,
            random_state=RANDOM_SEED
        )

    print(f"\n  Split sizes:")
    print(f"    Train : {len(train_df):,} rows  "
          f"({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val   : {len(val_df):,} rows  "
          f"({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test  : {len(test_df):,} rows  "
          f"({len(test_df)/len(df)*100:.1f}%)")

    # Verify difficulty preserved across splits
    print(f"\n  Difficulty in train:")
    for d, c in train_df["difficulty"]\
            .value_counts().items():
        print(f"    {d:10s}: {c:,}")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────
# STEP 5 — BUILD PUBMEDQA BASELINE TEST SET
# ─────────────────────────────────────────────────────────────
def build_pubmedqa_baseline():
    log("Building PubMedQA baseline test set...")

    path = os.path.join(RAW_DIR, "pubmedqa.jsonl")
    if not os.path.exists(path):
        print("  WARNING: pubmedqa.jsonl not found — skipping")
        return None

    df      = pd.read_json(path, lines=True)
    sample  = df.sample(
        n=min(500, len(df)),
        random_state=RANDOM_SEED
    )

    out = os.path.join(SPLITS_DIR, "pubmedqa_baseline.jsonl")
    sample.to_json(out, orient="records", lines=True)
    print(f"  Saved : {out}")
    print(f"  Rows  : {len(sample):,}")
    print(f"  Use   : compare patient-aware vs generic accuracy")
    return sample


# ─────────────────────────────────────────────────────────────
# STEP 6 — SAVE ALL FILES
# ─────────────────────────────────────────────────────────────
def save_all(full_df, train_df, val_df, test_df):
    log("Saving all dataset files...")

    files = {
        os.path.join(COMBINED_DIR, "full_dataset.jsonl") : full_df,
        os.path.join(SPLITS_DIR,   "train.jsonl")        : train_df,
        os.path.join(SPLITS_DIR,   "val.jsonl")          : val_df,
        os.path.join(SPLITS_DIR,   "test.jsonl")         : test_df,
    }

    for path, df in files.items():
        df.to_json(path, orient="records", lines=True)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {os.path.basename(path):30s}: "
              f"{len(df):,} rows  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────
# STEP 7 — DATASET REPORT
# ─────────────────────────────────────────────────────────────
def save_dataset_report(full_df, train_df,
                        val_df, test_df):
    log("Generating dataset report...")

    pmid_rows = full_df[
        full_df["match_type"] == "pmid_semantic_bridge"]
    sem_rows  = full_df[
        full_df["match_type"] == "semantic_fallback"]

    report = {
        "dataset_name"          : (
            "Patient-Aware Hallucination Detection Dataset"
        ),
        "version"               : "1.0",
        "created_by"            : "build_combined.py",
        "source_datasets"       : [
            "PMC-Patients V2 (zhengyun21/PMC-Patients)",
            "MedHallu (UTAustin-AIHealth/MedHallu)",
            "PubMedQA (qiaojin/PubMedQA pqa_labeled)"
        ],
        "matching_method"       : (
            "3-stage deterministic: semantic PMID bridge "
            "+ cosine-best patient + semantic fallback"
        ),
        "random_assignment_used": False,
        "total_records"         : len(full_df),
        "train_records"         : len(train_df),
        "val_records"           : len(val_df),
        "test_records"          : len(test_df),
        "label_distribution"    : full_df[
            "is_hallucinated"].value_counts().to_dict(),
        "difficulty_distribution": full_df[
            "difficulty"].value_counts().to_dict(),
        "trust_score_distribution": full_df[
            "trust_score"].value_counts().to_dict(),
        "hallucination_types"   : full_df[
            "hallucination_type"].value_counts().to_dict(),
        "match_type_distribution": full_df[
            "match_type"].value_counts().to_dict(),
        "pmid_bridge_records"   : len(pmid_rows),
        "semantic_records"      : len(sem_rows),
        "patient_gender_dist"   : full_df[
            "patient_gender"].value_counts().to_dict(),
        "split_strategy"        : (
            "Expert-labeled rows to test only. "
            "Artificial rows stratified by difficulty "
            "into train (82%) + val (18%)."
        ),
        "schema": {
            "matched_patient_uid" : "PMC-Patients record ID",
            "patient_context"     : "Full patient summary text",
            "patient_age"         : "Patient age (string)",
            "patient_gender"      : "M / F / unknown",
            "question"            : "Medical question (MedHallu)",
            "knowledge"           : "PubMed abstract context",
            "answer"              : "Answer being evaluated",
            "ground_truth"        : "Expert-verified correct answer",
            "is_hallucinated"     : "0=correct  1=hallucinated",
            "difficulty"          : "easy / medium / hard",
            "hallucination_type"  : "Category of hallucination",
            "trust_score"         : "Confidence target (0.3/0.6/0.9/1.0)",
            "match_type"          : "pmid_semantic_bridge / semantic_fallback",
            "cosine_similarity"   : "Patient match cosine score",
            "bridge_cosine"       : "PMID bridge cosine score (Stage 1 only)",
        }
    }

    out = os.path.join(COMBINED_DIR, "dataset_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved : {out}")

    # Print paper-ready summary
    print(f"\n  ── Dataset Summary (for paper) ─────────────")
    print(f"  Total records      : {report['total_records']:,}")
    print(f"  Train              : {report['train_records']:,}")
    print(f"  Val                : {report['val_records']:,}")
    print(f"  Test               : {report['test_records']:,}")
    print(f"  Hallucinated (1)   : "
          f"{report['label_distribution'].get(1, 0):,}")
    print(f"  Correct (0)        : "
          f"{report['label_distribution'].get(0, 0):,}")
    print(f"  Difficulty dist    : "
          f"{report['difficulty_distribution']}")
    print(f"  Trust scores       : "
          f"{report['trust_score_distribution']}")
    print(f"  Halluc types       : "
          f"{report['hallucination_types']}")
    print(f"  PMID-bridge pairs  : {report['pmid_bridge_records']:,}")
    print(f"  Semantic pairs     : {report['semantic_records']:,}")
    print(f"  Random assignment  : False")
    print(f"  ─────────────────────────────────────────────")

    return report


# ─────────────────────────────────────────────────────────────
# STEP 8 — FINAL FOLDER SUMMARY
# ─────────────────────────────────────────────────────────────
def print_folder_summary():
    log("DATASET BUILD COMPLETE")

    folders = {
        "data/combined" : [
            "full_dataset.jsonl",
            "matched_pairs.jsonl",
            "match_report.json",
            "dataset_report.json",
        ],
        "data/splits" : [
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "pubmedqa_baseline.jsonl",
        ]
    }

    for folder, files in folders.items():
        print(f"\n  {folder}/")
        for fname in files:
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / 1e6
                print(f"    {fname:35s} "
                      f"{size:.1f} MB")
            else:
                print(f"    {fname:35s} NOT FOUND")

    print(f"\n  Your custom patient-aware dataset is ready.")
    print(f"\n  Next steps:")
    print(f"    Phase 2 — Model training")
    print(f"      python src/model/train.py")
    print(f"    Phase 3 — Backend + Frontend")
    print(f"      python backend/main.py")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPatient-Aware Hallucination Detection")
    print("Build Combined Dataset Script")

    # Load matched pairs from match_datasets.py output
    matched_df = load_matched_pairs()

    # Build positive + negative pairs (2 records per input)
    combined_df = build_pairs(matched_df)

    # Quality checks — nulls, dedup, balance, shuffle
    combined_df = quality_checks(combined_df)

    # Train / val / test split
    train_df, val_df, test_df = split_dataset(combined_df)

    # PubMedQA baseline test set (for comparison)
    build_pubmedqa_baseline()

    # Save all files
    save_all(combined_df, train_df, val_df, test_df)

    # Generate full dataset report
    save_dataset_report(
        combined_df, train_df, val_df, test_df
    )

    # Print folder summary
    print_folder_summary()