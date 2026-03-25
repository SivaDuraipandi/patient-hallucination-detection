"""
preprocess.py
Cleans and normalises all 3 raw datasets.
Saves cleaned files to data/raw/ (overwrites with clean versions).
Run after verify_downloads.py passes.
"""

import os, re, ast
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"


def log(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


# ── Text cleaning helpers ─────────────────────────────────────
def clean_text(text):
    """Remove extra whitespace, fix encoding artifacts."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'\n+', ' ', text)          # remove newlines
    text = text.encode('ascii', 'ignore')\
               .decode('ascii')               # remove non-ASCII
    return text.strip()


def clean_age(age_val):
    """Normalise age to a readable string."""
    if pd.isna(age_val) or str(age_val).strip() in ("", "nan"):
        return "unknown"
    age_str = str(age_val).strip().lower()
    # Extract numeric age if present
    nums = re.findall(r'\d+', age_str)
    if nums:
        return nums[0] + " years"
    return age_str


def clean_gender(gender_val):
    """Normalise gender to M / F / unknown."""
    if pd.isna(gender_val):
        return "unknown"
    g = str(gender_val).strip().lower()
    if g in ("m", "male", "man", "boy"):
        return "M"
    if g in ("f", "female", "woman", "girl"):
        return "F"
    return "unknown"


def parse_relevant_articles(val):
    """Parse relevant_articles into a clean Python list."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "[]"):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except Exception:
        pass
    # Fallback: split by comma
    cleaned = re.sub(r"[\[\]'\"{}]", "", str(val))
    parts   = [p.strip() for p in cleaned.split(",") if p.strip()]
    return parts


# ── PREPROCESS 1: PMC-PATIENTS ───────────────────────────────
def preprocess_pmc_patients():
    log("Preprocessing PMC-Patients V2...")
    path = os.path.join(RAW_DIR, "pmc_patients.jsonl")
    df   = pd.read_json(path, lines=True)
    before = len(df)
    print(f"  Loaded : {before:,} rows")

    # Clean patient text
    print("  Cleaning patient text...")
    df["patient"] = df["patient"].apply(clean_text)

    # Remove rows where patient text is too short to be useful
    df = df[df["patient"].str.len() >= 50].copy()

    # Normalise age and gender
    df["age"]    = df["age"].apply(clean_age)
    df["gender"] = df["gender"].apply(clean_gender)

    # Parse relevant_articles into a real list
    print("  Parsing relevant_articles (PMID lists)...")
    df["relevant_articles"] = df["relevant_articles"].apply(
        parse_relevant_articles
    )

    # Add a short patient_summary field (first 512 chars)
    # Used later for faster semantic encoding
    df["patient_summary"] = df["patient"].str[:512]

    # Reset index
    df = df.reset_index(drop=True)
    after = len(df)

    # Save
    df.to_json(path, orient="records", lines=True)
    print(f"  Before : {before:,}  →  After : {after:,}")
    print(f"  Removed: {before - after:,} short/empty rows")
    print(f"  Gender distribution : "
          f"{df['gender'].value_counts().to_dict()}")
    print(f"  Avg text length     : "
          f"{df['patient'].str.len().mean():.0f} chars")
    print(f"  Saved  : {path}")
    return df


# ── PREPROCESS 2: MEDHALLU ────────────────────────────────────
def preprocess_medhallu():
    log("Preprocessing MedHallu...")
    path = os.path.join(RAW_DIR, "medhallu.jsonl")
    df   = pd.read_json(path, lines=True)
    before = len(df)
    print(f"  Loaded : {before:,} rows")

    # Clean all text fields
    text_cols = ["question", "knowledge",
                 "ground_truth", "hallucinated_ans"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Remove rows with empty critical fields
    df = df[df["question"].str.len() > 10].copy()
    df = df[df["ground_truth"].str.len() > 5].copy()
    df = df[df["hallucinated_ans"].str.len() > 5].copy()

    # Normalise difficulty to lowercase
    df["difficulty"] = (
        df["difficulty"].astype(str).str.lower().str.strip()
    )
    # Map any unexpected values to 'unknown'
    valid_difficulties = {"easy", "medium", "hard"}
    df["difficulty"] = df["difficulty"].apply(
        lambda x: x if x in valid_difficulties else "unknown"
    )

    # Normalise hallucination_type
    if "hallucination_type" in df.columns:
        df["hallucination_type"] = (
            df["hallucination_type"]
            .astype(str).str.strip().fillna("unknown")
        )
    else:
        df["hallucination_type"] = "unknown"

    # Add trust_score_target based on difficulty
    # Easy   → model should be very confident → 0.9
    # Medium → moderate confidence             → 0.6
    # Hard   → low confidence                  → 0.3
    trust_map = {"easy": 0.9, "medium": 0.6,
                 "hard": 0.3, "unknown": 0.5}
    df["trust_score_target"] = df["difficulty"].map(trust_map)

    # Verify split_source is present
    if "split_source" not in df.columns:
        df["split_source"] = "artificial"

    df = df.reset_index(drop=True)
    after = len(df)

    df.to_json(path, orient="records", lines=True)
    print(f"  Before : {before:,}  →  After : {after:,}")
    print(f"  Difficulty  : "
          f"{df['difficulty'].value_counts().to_dict()}")
    print(f"  Split source: "
          f"{df['split_source'].value_counts().to_dict()}")
    print(f"  Halluc types: "
          f"{df['hallucination_type'].value_counts().to_dict()}")
    print(f"  Trust scores: "
          f"{df['trust_score_target'].value_counts().to_dict()}")
    print(f"  Saved  : {path}")
    return df


# ── PREPROCESS 3: PUBMEDQA ────────────────────────────────────
def preprocess_pubmedqa():
    log("Preprocessing PubMedQA...")
    path = os.path.join(RAW_DIR, "pubmedqa.jsonl")
    df   = pd.read_json(path, lines=True)
    before = len(df)
    print(f"  Loaded : {before:,} rows")

    # Clean text fields
    for col in ["question", "context_text", "long_answer"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Remove rows with empty question or answer
    df = df[df["question"].str.len() > 10].copy()
    df = df[df["long_answer"].str.len() > 5].copy()

    # Normalise final_decision
    if "final_decision" in df.columns:
        df["final_decision"] = (
            df["final_decision"].astype(str)
            .str.lower().str.strip()
        )
        valid_decisions = {"yes", "no", "maybe"}
        df["final_decision"] = df["final_decision"].apply(
            lambda x: x if x in valid_decisions else "unknown"
        )

    # Convert pubid to string for consistent matching later
    df["pubid"] = df["pubid"].astype(str).str.strip()

    df = df.reset_index(drop=True)
    after = len(df)

    df.to_json(path, orient="records", lines=True)
    print(f"  Before : {before:,}  →  After : {after:,}")
    if "final_decision" in df.columns:
        print(f"  Decisions: "
              f"{df['final_decision'].value_counts().to_dict()}")
    print(f"  Avg question length  : "
          f"{df['question'].str.len().mean():.0f} chars")
    print(f"  Avg context length   : "
          f"{df['context_text'].str.len().mean():.0f} chars")
    print(f"  Saved  : {path}")
    return df


# ── SUMMARY REPORT ────────────────────────────────────────────
def print_summary(pmc_df, mh_df, pubmed_df):
    log("PREPROCESSING COMPLETE — SUMMARY")
    print(f"  PMC-Patients  : {len(pmc_df):,} rows")
    print(f"    Cols: {list(pmc_df.columns)}")
    print(f"\n  MedHallu      : {len(mh_df):,} rows")
    print(f"    Cols: {list(mh_df.columns)}")
    print(f"\n  PubMedQA      : {len(pubmed_df):,} rows")
    print(f"    Cols: {list(pubmed_df.columns)}")
    print(f"\n  All files saved to: {RAW_DIR}/")
    print(f"\n  Next step:")
    print(f"    python src/dataset/match_datasets.py")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPatient-Aware Hallucination Detection")
    print("Dataset Preprocessing Script")

    pmc_df    = preprocess_pmc_patients()
    mh_df     = preprocess_medhallu()
    pubmed_df = preprocess_pubmedqa()

    print_summary(pmc_df, mh_df, pubmed_df)