import argparse
import os
from pathlib import Path  # NEW
import torch
from tqdm import tqdm
from language_quality import extract_good_candidates_by_LQ
from utils import (
    read_candidates,
    initialize_train_test_dataset,
    to_method_object,
    convert_to_contexts_responses,
)

# -----------------------------------------------------------------------------
# Device selection (identical behaviour, modern torch still supports this)
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Helper ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _resolve_candidates_file(dataset: str, explicit_file: str | None) -> Path:
    """Return Path to <dataset>_candidates.txt unless user overrides.

    ### CHANGED:  allow `--candidates_file` CLI flag so we can point to the
    sample file without renaming or copying.  Falls back to the legacy naming
    scheme for full backward‑compatibility.
    """
    if explicit_file:
        return Path(explicit_file).expanduser().resolve()
    return Path("./data") / f"{dataset}_candidates.txt"


# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    print("Start Main…")

    # ---------- load synthetic candidates from Module 1 ----------------------
    cand_path: Path = _resolve_candidates_file(args.dataset, args.candidates_file)
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {cand_path}")
    candidates = read_candidates(cand_path)

    # ---------- load human train / test split --------------------------------
    train_x_text, train_y_text, test_x_text, test_y_text = initialize_train_test_dataset(
        args.dataset
    )
    contexts_train, responses_train = convert_to_contexts_responses(
        train_x_text, train_y_text
    )

    # -------------------------------------------------------------------------
    # Module 2 – grammaticality pruning (Language‑Quality model)
    # -------------------------------------------------------------------------
    candidates = extract_good_candidates_by_LQ(
        candidates, LQ_thres=0.52, num_of_generation=args.num_gen
    )  # CHANGED: num_gen becomes CLI flag (default 30000)

    tfidf = to_method_object("TF_IDF")
    tfidf.train(contexts_train, responses_train)
    good_idx = tfidf.sort_responses(
        test_x_text, candidates, k=min(args.kpq, len(candidates))
    )
    good_cands = [[candidates[j] for j in idx_row] for idx_row in good_idx]

    # -------------------------------------------------------------------------
    # Module 3 – response selection with ranking models
    # -------------------------------------------------------------------------
    METHODS = [
        "TF_IDF",
        "BM25",
        "USE_SIM",
        "USE_MAP",
        "USE_LARGE_SIM",
        "USE_LARGE_MAP",
        "ELMO_SIM",
        "ELMO_MAP",
        "BERT_SMALL_SIM",
        "BERT_SMALL_MAP",
        "BERT_LARGE_SIM",
        "BERT_LARGE_MAP",
        "USE_QA_SIM",
        "USE_QA_MAP",
        "CONVERT_SIM",
        "CONVERT_MAP",
    ]

    for method_name in METHODS[args.method_start : args.method_end]:  # CHANGED
        print("\n", "=" * 20, method_name, "=" * 20)
        method = to_method_object(method_name)
        method.train(contexts_train, responses_train)
        outputs = []
        for query, cand_subset in tqdm(zip(test_x_text, good_cands), total=len(test_x_text)):
            pred_idx = method.rank_responses([query], cand_subset)
            outputs.append(cand_subset[pred_idx.item()])
        print(outputs[:5], "… (total", len(outputs), ")")

    print("*" * 80)
    print(f"After LQ filtering, {len(candidates)} candidates remain.\n")


# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Main.py",
        description="Modules 2 & 3 of pipeline (candidate pruning + ranking)",
    )
    parser.add_argument(
        "--kpq",
        type=int,
        default=100,
        help="Top‑k candidates per query after TF‑IDF pre‑selection.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        choices=["reddit", "gab", "conan"],
        help="Dataset prefix (expects <dataset>_candidates.txt unless --candidates_file given).",
    )
    parser.add_argument(
        "--candidates_file",
        type=str,
        default=None,
        help="Optional explicit path to candidates txt file (overrides --dataset).",
    )  # NEW
    parser.add_argument(
        "--num_gen",
        type=int,
        default=30000,
        help="num_of_generation arg for Language‑Quality pruning (was hard‑coded).",
    )  # NEW
    parser.add_argument(
        "--method_start",
        type=int,
        default=14,
        help="Start index to slice METHODS list (enables quick test).",
    )  # NEW
    parser.add_argument(
        "--method_end",
        type=int,
        default=16,
        help="End index (non‑inclusive) for METHODS slice.",
    )  # NEW

    cli_args = parser.parse_args()
    main(cli_args)
