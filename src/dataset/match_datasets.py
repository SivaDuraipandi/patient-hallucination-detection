"""
match_datasets.py  —  V3
========================
Identical to V2 but with the colon-score PMID fix.

Root cause confirmed by debug:
  PMC relevant_articles format : '32320506: 1'
  PubMedQA pubid format        : 21645374  (int64)

Fix: extract_pmids() now splits on ':' and takes the
left part only, stripping the relevance score.
All PMIDs normalised to clean digit string before
any comparison or index lookup.

Three-stage fully deterministic pipeline:
  Stage 1A — semantic PMID bridge
              (MedHallu → PubMedQA cosine >= 0.75)
  Stage 1B — PMID → cosine-best PMC patient
  Stage 2  — semantic fallback for unmatched rows

No random assignment at any stage.
Fixed seed 42 only for PMC sample selection.

Output:
  data/combined/matched_pairs.jsonl
  data/combined/review_pairs.jsonl
  data/combined/match_report.json
"""

import os
import json
import ast
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# ── Paths ─────────────────────────────────────────────────────
RAW_DIR      = "data/raw"
COMBINED_DIR = "data/combined"
os.makedirs(COMBINED_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
SEMANTIC_SAMPLE       = 15000
PMID_BRIDGE_THRESHOLD = 0.75
SEMANTIC_ACCEPT_THRESHOLD = 0.50
SEMANTIC_REVIEW_THRESHOLD = 0.40
RANDOM_SEED           = 42
np.random.seed(RANDOM_SEED)


def log(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")


def classify_match_confidence(match_type, patient_score, bridge_score=0.0):
    """Assign a simple confidence tier for downstream filtering/reporting."""
    if match_type == "pmid_semantic_bridge":
        if patient_score >= 0.75 and bridge_score >= 0.90:
            return "high"
        return "medium"

    if patient_score >= 0.65:
        return "high"
    if patient_score >= 0.55:
        return "medium"
    if patient_score >= SEMANTIC_ACCEPT_THRESHOLD:
        return "low"
    if patient_score >= SEMANTIC_REVIEW_THRESHOLD:
        return "review"
    return "reject"


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASETS
# ─────────────────────────────────────────────────────────────
def load_datasets():
    log("Loading preprocessed datasets...")

    pmc_df = pd.read_json(
        os.path.join(RAW_DIR, "pmc_patients.jsonl"),
        lines=True
    )
    mh_df = pd.read_json(
        os.path.join(RAW_DIR, "medhallu.jsonl"),
        lines=True
    )
    pubmed_df = pd.read_json(
        os.path.join(RAW_DIR, "pubmedqa.jsonl"),
        lines=True
    )

    print(f"  PMC-Patients : {len(pmc_df):,} rows")
    print(f"  MedHallu     : {len(mh_df):,} rows")
    print(f"  PubMedQA     : {len(pubmed_df):,} rows")
    return pmc_df, mh_df, pubmed_df


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD ENCODER
# ─────────────────────────────────────────────────────────────
def load_encoder():
    log("Loading sentence encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Status : Loaded successfully")
    return encoder


# ─────────────────────────────────────────────────────────────
# STEP 3 — PMID NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────
def normalise_pmid(raw):
    """
    Convert any PMID format to a clean digit string.

    Handles:
      '32320506: 1'  → '32320506'   (colon-score format)
      '32320506.0'   → '32320506'   (float string)
      32320506       → '32320506'   (integer)
      32320506.0     → '32320506'   (float)
      '32320506'     → '32320506'   (already clean)
    """
    if raw is None:
        return None

    # Convert to string first
    s = str(raw).strip()

    # ── KEY FIX: strip colon-score suffix '32320506: 1' ──────
    if ':' in s:
        s = s.split(':')[0].strip()

    # Strip float decimal suffix
    if s.endswith('.0'):
        s = s[:-2]

    # Strip any surrounding quotes or spaces
    s = s.strip("'\" ")

    # Valid PMID must be all digits
    return s if s.isdigit() else None


def extract_pmids_from_field(val):
    """
    Extract a list of normalised PMIDs from the
    relevant_articles field regardless of its storage format.

    Confirmed format from debug:
      list of strings: ['32320506: 1', '23219649: 1', ...]
    Also handles: stringified lists, plain strings, ints.
    """
    if val is None:
        return []

    # ── Already a list (confirmed format from debug) ──────────
    if isinstance(val, list):
        result = []
        for item in val:
            pmid = normalise_pmid(item)
            if pmid:
                result.append(pmid)
        return result

    # ── Integer or float ──────────────────────────────────────
    if isinstance(val, (int, float)):
        pmid = normalise_pmid(val)
        return [pmid] if pmid else []

    # ── String — try to parse as Python list first ────────────
    if isinstance(val, str):
        val = val.strip()
        if val in ("", "[]", "nan"):
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return [p for item in parsed
                        for p in [normalise_pmid(item)]
                        if p]
            pmid = normalise_pmid(parsed)
            return [pmid] if pmid else []
        except Exception:
            pass
        # Plain string PMID
        pmid = normalise_pmid(val)
        return [pmid] if pmid else []

    return []


# ─────────────────────────────────────────────────────────────
# STEP 4 — BUILD PMC PMID INDEX
# ─────────────────────────────────────────────────────────────
def build_pmc_pmid_index(pmc_df, pubmed_df):
    """
    Builds pmid_to_pmc_rows:
      key   = normalised PMID string
      value = list of pmc_df row indices

    Also prints cross-check against PubMedQA pubids
    so we can verify matches exist before running Stage 1B.
    """
    log("Building PMC-Patients PMID index (V3 fix)...")

    pmid_to_pmc_rows = defaultdict(list)

    for idx, row in tqdm(
        pmc_df.iterrows(),
        total=len(pmc_df),
        desc="  Indexing relevant_articles"
    ):
        for pmid in extract_pmids_from_field(
            row.get("relevant_articles", [])
        ):
            pmid_to_pmc_rows[pmid].append(idx)

    print(f"  Unique PMIDs indexed    : "
          f"{len(pmid_to_pmc_rows):,}")

    # ── Cross-check: verify overlap with PubMedQA ────────────
    pubmed_ids = set(
        str(int(v)) for v in pubmed_df["pubid"].tolist()
        if str(v).strip() not in ("", "nan")
    )
    overlap = pubmed_ids & set(pmid_to_pmc_rows.keys())

    print(f"  PubMedQA unique pubids  : {len(pubmed_ids):,}")
    print(f"  Overlap with PMC index  : {len(overlap):,}")
    print(f"  PMC patients reachable  : "
          f"{sum(len(pmid_to_pmc_rows[p]) for p in overlap):,}")

    if len(overlap) == 0:
        print("\n  WARNING: Still no overlap detected.")
        print("  Sample PMC PMIDs from index:")
        for k in list(pmid_to_pmc_rows.keys())[:5]:
            print(f"    '{k}'")
        print("  Sample PubMedQA pubids:")
        for v in list(pubmed_ids)[:5]:
            print(f"    '{v}'")
    else:
        print(f"\n  PMID bridge confirmed — "
              f"{len(overlap):,} shared PMIDs found")
        print(f"  Sample overlap PMIDs: "
              f"{list(overlap)[:5]}")

    return pmid_to_pmc_rows


# ─────────────────────────────────────────────────────────────
# STEP 5 — STAGE 1A: SEMANTIC PMID BRIDGE
# ─────────────────────────────────────────────────────────────
def build_semantic_pmid_bridge(mh_df, pubmed_df, encoder):
    """
    Encodes MedHallu and PubMedQA questions.
    Matches each MedHallu question to the most similar
    PubMedQA question (cosine >= 0.75).
    Returns: mh_question_to_pmid
      key   = MedHallu question string
      value = (normalised_pmid, bridge_cosine_score)
    """
    log("Stage 1A — Semantic PMID bridge...")
    print(f"  Threshold : cosine >= {PMID_BRIDGE_THRESHOLD}")

    # Normalise PubMedQA pubids to clean digit strings
    pubmed_questions = pubmed_df["question"].tolist()
    pubmed_pmids = [
        str(int(v)) for v in pubmed_df["pubid"].tolist()
    ]

    print(f"\n  Encoding {len(pubmed_questions):,} "
          f"PubMedQA questions...")
    pubmed_embs = encoder.encode(
        pubmed_questions,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    mh_questions = mh_df["question"].tolist()
    print(f"\n  Encoding {len(mh_questions):,} "
          f"MedHallu questions...")
    mh_embs = encoder.encode(
        mh_questions,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    print("\n  Computing MedHallu × PubMedQA cosine matrix...")
    scores_matrix = util.cos_sim(mh_embs, pubmed_embs)

    mh_question_to_pmid = {}
    bridge_scores_list  = []

    for i, q in enumerate(mh_questions):
        best_idx   = int(scores_matrix[i].argmax())
        best_score = float(scores_matrix[i][best_idx])

        if best_score >= PMID_BRIDGE_THRESHOLD:
            pmid = pubmed_pmids[best_idx]
            mh_question_to_pmid[q] = (pmid, round(best_score, 4))
            bridge_scores_list.append(best_score)

    accepted  = len(mh_question_to_pmid)
    rejected  = len(mh_questions) - accepted
    mean_sc   = round(float(np.mean(bridge_scores_list)), 4) \
                if bridge_scores_list else 0.0

    print(f"\n  Accepted (>= {PMID_BRIDGE_THRESHOLD}) : "
          f"{accepted:,}  "
          f"({accepted / len(mh_questions) * 100:.1f}%)")
    print(f"  Rejected (<  {PMID_BRIDGE_THRESHOLD}) : "
          f"{rejected:,}  "
          f"({rejected / len(mh_questions) * 100:.1f}%)")
    print(f"  Mean bridge cosine         : {mean_sc}")

    return mh_question_to_pmid


# ─────────────────────────────────────────────────────────────
# STEP 6 — STAGE 1B: PMID → COSINE-BEST PATIENT
# ─────────────────────────────────────────────────────────────
def pmid_to_best_patient(question, pmid,
                         pmid_to_pmc_rows, pmc_df, encoder):
    """
    Given a question and normalised pubid, find the best
    matching PMC-Patients record.
    Single candidate  → return directly
    Multiple          → cosine-best selection
    None found        → return (None, 0.0)
    """
    candidates = pmid_to_pmc_rows.get(str(pmid).strip(), [])

    if not candidates:
        return None, 0.0

    if len(candidates) == 1:
        return pmc_df.iloc[candidates[0]], 1.0

    # Multiple candidates — encode and pick best
    q_emb = encoder.encode(
        [question],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    candidate_texts = []
    for idx in candidates:
        row  = pmc_df.iloc[idx]
        text = str(row.get(
            "patient_summary",
            row.get("patient", "")
        ))[:256]
        candidate_texts.append(text)

    c_embs = encoder.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores    = util.cos_sim(q_emb, c_embs)[0]
    best_pos  = int(scores.argmax())
    best_score = float(scores[best_pos])

    return (pmc_df.iloc[candidates[best_pos]],
            round(best_score, 4))


# ─────────────────────────────────────────────────────────────
# STEP 7 — STAGE 2: SEMANTIC FALLBACK
# ─────────────────────────────────────────────────────────────
def build_semantic_index(pmc_df, encoder):
    log("Stage 2 — Building semantic fallback index...")

    n = min(SEMANTIC_SAMPLE, len(pmc_df))
    pmc_sample = pmc_df.sample(
        n=n, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    if "patient_summary" in pmc_sample.columns:
        texts = pmc_sample["patient_summary"].fillna("").tolist()
    else:
        texts = pmc_sample["patient"].str[:512].fillna("").tolist()

    print(f"  Encoding {n:,} patient summaries...")
    print(f"  (~3-5 min on CPU — please wait)")

    pmc_embeddings = encoder.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    print(f"  Embedding shape : {pmc_embeddings.shape}")
    return pmc_sample, pmc_embeddings


def semantic_match_batch(questions, encoder,
                         pmc_sample, pmc_embeddings):
    print(f"  Encoding {len(questions):,} fallback questions...")

    q_embs = encoder.encode(
        questions,
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    print("  Computing cosine similarity matrix...")
    scores = util.cos_sim(q_embs, pmc_embeddings)

    results = []
    for i in range(len(questions)):
        best_idx   = int(scores[i].argmax())
        best_score = float(scores[i][best_idx])
        results.append(
            (pmc_sample.iloc[best_idx], round(best_score, 4))
        )
    return results


# ─────────────────────────────────────────────────────────────
# STEP 8 — RUN FULL 3-STAGE PIPELINE
# ─────────────────────────────────────────────────────────────
def run_matching(pmc_df, mh_df, pubmed_df, encoder):

    # Build PMC PMID index with colon-score fix
    pmid_to_pmc_rows = build_pmc_pmid_index(pmc_df, pubmed_df)

    # Stage 1A — semantic PMID bridge
    mh_question_to_pmid = build_semantic_pmid_bridge(
        mh_df, pubmed_df, encoder
    )

    # Initialise result arrays
    n             = len(mh_df)
    matched_rows  = [None]  * n
    match_types   = [""]    * n
    match_scores  = [0.0]   * n
    match_pmids   = [""]    * n
    bridge_scores = [0.0]   * n
    unmatched_idx = []

    # Stage 1B — PMID → best patient
    log("Stage 1B — PMID → best patient lookup...")

    stage1_success = 0
    stage1_no_pmc  = 0

    for i, (_, mh_row) in enumerate(tqdm(
        mh_df.iterrows(),
        total=n,
        desc="  Stage 1B PMID→patient"
    )):
        question = str(mh_row["question"])
        bridge   = mh_question_to_pmid.get(question)

        if bridge is None:
            unmatched_idx.append(i)
            continue

        pmid, b_score = bridge

        pmc_row, p_score = pmid_to_best_patient(
            question, pmid,
            pmid_to_pmc_rows, pmc_df, encoder
        )

        if pmc_row is not None:
            matched_rows[i]  = pmc_row
            match_types[i]   = "pmid_semantic_bridge"
            match_scores[i]  = p_score
            match_pmids[i]   = pmid
            bridge_scores[i] = b_score
            stage1_success  += 1
        else:
            unmatched_idx.append(i)
            stage1_no_pmc += 1

    print(f"\n  Stage 1 results:")
    print(f"    PMID bridge + patient found : {stage1_success:,}")
    print(f"    PMID bridge, no PMC match   : {stage1_no_pmc:,}")
    print(f"    No PMID bridge              : "
          f"{n - stage1_success - stage1_no_pmc:,}")
    print(f"    Needs Stage 2 fallback      : "
          f"{len(unmatched_idx):,}")

    # Stage 2 — semantic fallback
    if unmatched_idx:
        pmc_sample, pmc_embeddings = build_semantic_index(
            pmc_df, encoder
        )
        fallback_questions = [
            str(mh_df.iloc[i]["question"])
            for i in unmatched_idx
        ]
        sem_results = semantic_match_batch(
            fallback_questions,
            encoder, pmc_sample, pmc_embeddings
        )
        print("  Assigning semantic fallback matches...")
        for j, idx in enumerate(unmatched_idx):
            pmc_row, score    = sem_results[j]
            matched_rows[idx] = pmc_row
            match_types[idx]  = "semantic_fallback"
            match_scores[idx] = score
            match_pmids[idx]  = ""

        print(f"  Stage 2 assigned : {len(unmatched_idx):,}")

    return (matched_rows, match_types,
            match_scores, match_pmids, bridge_scores)


# ─────────────────────────────────────────────────────────────
# STEP 9 — BUILD matched_pairs.jsonl
# ─────────────────────────────────────────────────────────────
def build_matched_pairs(mh_df, matched_rows, match_types,
                        match_scores, match_pmids,
                        bridge_scores):
    log("Building matched_pairs.jsonl...")

    records = []
    review_records = []
    skipped = 0

    for i, (_, mh_row) in enumerate(mh_df.iterrows()):
        pmc_row = matched_rows[i]
        if pmc_row is None:
            skipped += 1
            continue

        match_confidence = classify_match_confidence(
            match_types[i],
            match_scores[i],
            bridge_scores[i],
        )

        # Keep borderline semantic matches for manual review instead
        # of mixing them into the training pool.
        if match_confidence == "review":
            review_records.append({
                "matched_patient_uid": str(
                    pmc_row.get("patient_uid", "")),
                "patient_context": str(
                    pmc_row.get("patient", "")),
                "question": str(mh_row.get("question", "")),
                "ground_truth": str(
                    mh_row.get("ground_truth", "")),
                "hallucinated_ans": str(
                    mh_row.get("hallucinated_ans", "")),
                "difficulty": str(
                    mh_row.get("difficulty", "unknown")),
                "match_type": match_types[i],
                "match_pmid": match_pmids[i],
                "cosine_similarity": match_scores[i],
                "bridge_cosine": bridge_scores[i],
                "match_confidence": match_confidence,
                "review_recommendation": "manual_review",
            })
            skipped += 1
            continue

        if match_confidence == "reject":
            skipped += 1
            continue

        record = {
            # Patient context (PMC-Patients)
            "matched_patient_uid" : str(
                pmc_row.get("patient_uid", "")),
            "patient_context"     : str(
                pmc_row.get("patient", "")),
            "patient_age"         : str(
                pmc_row.get("age", "unknown")),
            "patient_gender"      : str(
                pmc_row.get("gender", "unknown")),

            # QA pair (MedHallu)
            "question"            : str(
                mh_row.get("question", "")),
            "knowledge"           : str(
                mh_row.get("knowledge", "")),
            "ground_truth"        : str(
                mh_row.get("ground_truth", "")),
            "hallucinated_ans"    : str(
                mh_row.get("hallucinated_ans", "")),
            "difficulty"          : str(
                mh_row.get("difficulty", "unknown")),
            "hallucination_type"  : str(
                mh_row.get("hallucination_type", "unknown")),
            "trust_score_target"  : float(
                mh_row.get("trust_score_target", 0.5)),
            "split_source"        : str(
                mh_row.get("split_source", "artificial")),

            # Match metadata (cite in paper)
            "match_type"        : match_types[i],
            "match_pmid"        : match_pmids[i],
            "cosine_similarity" : match_scores[i],
            "bridge_cosine"     : bridge_scores[i],
            "match_confidence"  : match_confidence,
        }
        records.append(record)

    matched_df = pd.DataFrame(records)
    review_df  = pd.DataFrame(review_records)
    out_path   = os.path.join(
        COMBINED_DIR, "matched_pairs.jsonl"
    )
    review_path = os.path.join(
        COMBINED_DIR, "review_pairs.jsonl"
    )
    matched_df.to_json(out_path, orient="records", lines=True)
    review_df.to_json(review_path, orient="records", lines=True)

    print(f"  Saved   : {out_path}")
    print(f"  Review  : {review_path}")
    print(f"  Rows    : {len(matched_df):,}")
    print(f"  Review  : {len(review_df):,}")
    if skipped > 0:
        print(f"  Skipped : {skipped} "
              f"(review/rejected semantic matches)")
    return matched_df


# ─────────────────────────────────────────────────────────────
# STEP 10 — MATCH QUALITY REPORT
# ─────────────────────────────────────────────────────────────
def save_match_report(matched_df):
    log("Generating match quality report...")

    pmid_rows = matched_df[
        matched_df["match_type"] == "pmid_semantic_bridge"]
    sem_rows  = matched_df[
        matched_df["match_type"] == "semantic_fallback"]

    def safe_stats(series):
        if len(series) == 0:
            return 0.0, 0.0, 0.0
        return (round(float(series.mean()), 4),
                round(float(series.min()),  4),
                round(float(series.max()),  4))

    pm_mean, pm_min, pm_max = safe_stats(
        pmid_rows["cosine_similarity"])
    sm_mean, sm_min, sm_max = safe_stats(
        sem_rows["cosine_similarity"])
    br_mean, _, _           = safe_stats(
        pmid_rows["bridge_cosine"])

    report = {
        "total_pairs"                  : len(matched_df),
        "matching_version"             : "V4_confidence_filtering",
        "random_assignment_used"       : False,
        "matching_method"              : (
            "3-stage: semantic PMID bridge "
            "(MedHallu→PubMedQA cosine>=0.75, "
            "colon-score PMID parsing) → "
            "PMID cosine-best patient → "
            "semantic fallback (all-MiniLM-L6-v2) "
            f"with accept>={SEMANTIC_ACCEPT_THRESHOLD} "
            f"and review>={SEMANTIC_REVIEW_THRESHOLD}"
        ),
        "semantic_accept_threshold"    : SEMANTIC_ACCEPT_THRESHOLD,
        "semantic_review_threshold"    : SEMANTIC_REVIEW_THRESHOLD,
        "pmid_bridge_count"            : len(pmid_rows),
        "pmid_bridge_pct"              : round(
            len(pmid_rows) / len(matched_df) * 100, 2),
        "pmid_mean_bridge_cosine"      : br_mean,
        "pmid_mean_patient_cosine"     : pm_mean,
        "pmid_min_patient_cosine"      : pm_min,
        "pmid_max_patient_cosine"      : pm_max,
        "semantic_fallback_count"      : len(sem_rows),
        "semantic_fallback_pct"        : round(
            len(sem_rows) / len(matched_df) * 100, 2),
        "semantic_mean_cosine"         : sm_mean,
        "semantic_min_cosine"          : sm_min,
        "semantic_max_cosine"          : sm_max,
        "difficulty_distribution"      : matched_df[
            "difficulty"].value_counts().to_dict(),
        "match_confidence_distribution": matched_df[
            "match_confidence"].value_counts().to_dict(),
        "split_source_distribution"    : matched_df[
            "split_source"].value_counts().to_dict(),
        "trust_score_distribution"     : matched_df[
            "trust_score_target"].value_counts().to_dict(),
    }

    report_path = os.path.join(
        COMBINED_DIR, "match_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Saved  : {report_path}")
    print(f"\n  ── Match Quality Report (V3) ───────────────")
    print(f"  Total pairs              : "
          f"{report['total_pairs']:,}")
    print(f"\n  PMID semantic bridge     : "
          f"{report['pmid_bridge_count']:,} "
          f"({report['pmid_bridge_pct']}%)")
    print(f"    Mean bridge cosine     : "
          f"{report['pmid_mean_bridge_cosine']}")
    print(f"    Mean patient cosine    : "
          f"{report['pmid_mean_patient_cosine']}")
    print(f"\n  Semantic fallback        : "
          f"{report['semantic_fallback_count']:,} "
          f"({report['semantic_fallback_pct']}%)")
    print(f"    Mean patient cosine    : "
          f"{report['semantic_mean_cosine']}")
    print(f"    Min  patient cosine    : "
          f"{report['semantic_min_cosine']}")
    print(f"    Accept threshold       : "
          f"{report['semantic_accept_threshold']}")
    print(f"    Review threshold       : "
          f"{report['semantic_review_threshold']}")
    print(f"\n  Confidence tiers        : "
          f"{report['match_confidence_distribution']}")
    print(f"\n  Random assignment        : False")
    print(f"\n  ── Paper methodology statement ─────────────")
    print(f"  Patient records matched using 3-stage")
    print(f"  deterministic pipeline. Stage 1: semantic")
    print(f"  PMID bridge (MedHallu→PubMedQA cosine>=0.75,")
    print(f"  mean={report['pmid_mean_bridge_cosine']})")
    print(f"  with colon-score PMID parsing, followed")
    print(f"  by cosine-best patient selection.")
    print(f"  Stage 2: semantic similarity fallback")
    print(f"  (mean cosine={report['semantic_mean_cosine']}).")
    print(f"  No random assignment used.")
    print(f"  ─────────────────────────────────────────────")

    return report


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPatient-Aware Hallucination Detection")
    print("Dataset Matching — V4 (Confidence Filtering)")
    print("Strategy : 3-stage deterministic + stricter filtering")

    pmc_df, mh_df, pubmed_df = load_datasets()
    encoder                  = load_encoder()

    (matched_rows, match_types,
     match_scores, match_pmids,
     bridge_scores) = run_matching(
        pmc_df, mh_df, pubmed_df, encoder
    )

    matched_df = build_matched_pairs(
        mh_df, matched_rows, match_types,
        match_scores, match_pmids, bridge_scores
    )

    report = save_match_report(matched_df)

    log("MATCHING V3 COMPLETE")
    print(f"  data/combined/matched_pairs.jsonl")
    print(f"  data/combined/review_pairs.jsonl")
    print(f"  data/combined/match_report.json")
    print(f"\n  Next step:")
    print(f"    python src/dataset/build_combined.py")
