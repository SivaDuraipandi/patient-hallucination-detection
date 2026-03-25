"""
download_datasets.py
Downloads all 3 datasets using direct file download.
Bypasses HuggingFace dataset parser entirely to avoid
cache/schema errors on Windows.
"""

import os, io, json, requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.getenv("DATA_RAW_DIR", "data/raw")
os.makedirs(RAW_DIR, exist_ok=True)


def log(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


def download_file(url, desc):
    """Stream-download any file from a URL with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total     = int(response.headers.get("content-length", 0))
    data      = b""
    with tqdm(total=total, unit="B", unit_scale=True,
              desc=f"  {desc}") as pbar:
        for chunk in response.iter_content(1024 * 1024):
            data += chunk
            pbar.update(len(chunk))
    return data


# ─────────────────────────────────────────────────────────────
# DATASET 1 — PMC-PATIENTS V2
# ─────────────────────────────────────────────────────────────
def download_pmc_patients():
    out = os.path.join(RAW_DIR, "pmc_patients.jsonl")

    if os.path.exists(out):
        df = pd.read_json(out, lines=True)
        log("PMC-Patients V2 — already downloaded, skipping")
        print(f"  Found : {out}")
        print(f"  Rows  : {len(df):,}")
        return df

    log("Downloading PMC-Patients V2 (direct CSV)...")
    url  = ("https://huggingface.co/datasets/zhengyun21/"
            "PMC-Patients/resolve/main/PMC-Patients.csv")
    data = download_file(url, "PMC-Patients.csv")

    print("  Parsing CSV...")
    df = pd.read_csv(
        io.StringIO(data.decode("utf-8")),
        dtype=str, low_memory=False
    )

    needed    = ["patient_uid", "patient", "age",
                 "gender", "relevant_articles"]
    available = [c for c in needed if c in df.columns]
    df        = df[available].copy()
    df        = df.dropna(subset=["patient"])
    df        = df[df["patient"].str.strip().str.len() > 100]
    df["age"]    = df["age"].fillna("unknown")
    df["gender"] = df["gender"].fillna("unknown")
    df["relevant_articles"] = \
        df.get("relevant_articles", pd.Series("[]")).fillna("[]")

    df.to_json(out, orient="records", lines=True)
    print(f"  Saved : {out}  |  Rows: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# DATASET 2 — MEDHALLU
# ─────────────────────────────────────────────────────────────
def download_medhallu():
    out = os.path.join(RAW_DIR, "medhallu.jsonl")

    if os.path.exists(out):
        df = pd.read_json(out, lines=True)
        log("MedHallu — already downloaded, skipping")
        print(f"  Found : {out}")
        print(f"  Rows  : {len(df):,}")
        return df

    log("Downloading MedHallu (direct parquet)...")

    # HuggingFace parquet URLs for both splits
    urls = {
        "expert": (
            "https://huggingface.co/datasets/UTAustin-AIHealth/"
            "MedHallu/resolve/main/data/pqa_labeled/"
            "train-00000-of-00001.parquet"
        ),
        "artificial": (
            "https://huggingface.co/datasets/UTAustin-AIHealth/"
            "MedHallu/resolve/main/data/pqa_artificial/"
            "train-00000-of-00001.parquet"
        ),
    }

    frames = []
    for split_name, url in urls.items():
        print(f"\n  Downloading {split_name} split...")
        data = download_file(url, f"MedHallu-{split_name}.parquet")
        df_split = pd.read_parquet(io.BytesIO(data))
        df_split["split_source"] = split_name
        print(f"  {split_name}: {len(df_split):,} rows")
        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)

    # Standardise column names
    rename_map = {
        "Question"                  : "question",
        "Knowledge"                 : "knowledge",
        "Ground Truth"              : "ground_truth",
        "Hallucinated Answer"       : "hallucinated_ans",
        "Difficulty Level"          : "difficulty",
        "Category of Hallucination" : "hallucination_type",
    }
    df = df.rename(columns={
        k: v for k, v in rename_map.items() if k in df.columns
    })

    df = df.dropna(subset=["question", "ground_truth",
                            "hallucinated_ans"])
    df["difficulty"] = \
        df["difficulty"].astype(str).str.lower().fillna("unknown")
    if "hallucination_type" not in df.columns:
        df["hallucination_type"] = "unknown"
    else:
        df["hallucination_type"] = \
            df["hallucination_type"].fillna("unknown")

    df.to_json(out, orient="records", lines=True)
    print(f"\n  Saved  : {out}")
    print(f"  Rows   : {len(df):,}")
    print(f"  Expert : {(df.split_source=='expert').sum():,}")
    print(f"  Artif. : {(df.split_source=='artificial').sum():,}")
    print(f"  Difficulty:\n    " +
          df["difficulty"].value_counts()
          .to_string().replace("\n", "\n    "))
    return df


# ─────────────────────────────────────────────────────────────
# DATASET 3 — PUBMEDQA  (direct parquet — no cache issues)
# ─────────────────────────────────────────────────────────────
def download_pubmedqa():
    out = os.path.join(RAW_DIR, "pubmedqa.jsonl")

    if os.path.exists(out):
        df = pd.read_json(out, lines=True)
        log("PubMedQA — already downloaded, skipping")
        print(f"  Found : {out}")
        print(f"  Rows  : {len(df):,}")
        return df

    log("Downloading PubMedQA pqa_labeled (GitHub JSON)...")

    # Official source: raw JSON from the pubmedqa GitHub repo
    url = ("https://raw.githubusercontent.com/pubmedqa/"
           "pubmedqa/master/data/ori_pqal.json")

    print("  Fetching from GitHub...")
    response = requests.get(url)
    response.raise_for_status()

    raw_data = response.json()   # dict of {pubid: {...}, ...}
    print(f"  Raw records: {len(raw_data):,}")

    # Flatten nested JSON into a DataFrame
    records = []
    for pubid, entry in raw_data.items():
        # contexts is a list of abstract sentences
        contexts = entry.get("CONTEXTS", [])
        context_text = " ".join(contexts) if contexts else ""

        records.append({
            "pubid"          : pubid,
            "question"       : entry.get("QUESTION", ""),
            "context_text"   : context_text,
            "long_answer"    : entry.get("LONG_ANSWER", ""),
            "final_decision" : entry.get("final_decision", ""),
            "reasoning_required": entry.get(
                "reasoning_required_pred", ""),
        })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["question"])
    df = df[df["question"].str.strip().str.len() > 5]

    df.to_json(out, orient="records", lines=True)
    print(f"  Saved : {out}")
    print(f"  Rows  : {len(df):,}")
    print(f"  Cols  : {list(df.columns)}")
    if "final_decision" in df.columns:
        print(f"  Decisions: "
              f"{df['final_decision'].value_counts().to_dict()}")
    return df

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPatient-Aware Hallucination Detection")
    print("Dataset Download Script")

    pmc_df    = download_pmc_patients()
    mh_df     = download_medhallu()
    pubmed_df = download_pubmedqa()

    log("ALL DOWNLOADS COMPLETE")
    print(f"  PMC-Patients : {len(pmc_df):,} rows")
    print(f"  MedHallu     : {len(mh_df):,} rows")
    print(f"  PubMedQA     : {len(pubmed_df):,} rows")
    print(f"\n  Files in: {RAW_DIR}/")
    print(f"    pmc_patients.jsonl")
    print(f"    medhallu.jsonl")
    print(f"    pubmedqa.jsonl")
    print(f"\n  Next: python src/dataset/verify_downloads.py")