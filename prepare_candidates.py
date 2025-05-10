#!/usr/bin/env python3
"""
Convert reddit.csv → reddit_for_VAE.txt

Usage:
    python prepare_candidates.py --csv ./data/reddit.csv \
                                 --out ./data/reddit_for_VAE.txt \
                                 --col text # adjust if needed
"""

import re
import argparse
import pandas as pd
from pathlib import Path

# ---------- the same cleaners you use in dataset.py ----------
MY_PUNC   = "!\"#$%&\()*+?_/:;[]{}|~,`"
TRANS_TAB = str.maketrans({c: " " for c in MY_PUNC})

def clean_str(text: str) -> str:
    text = re.sub(r"\'s ",  " ",   text)
    text = re.sub(r"\'m ",  " ",   text)
    text = re.sub(r"\'ve ", " ",   text)
    text = re.sub(r"n\'t ", " not ", text)
    text = re.sub(r"\'re ", " ",   text)
    text = re.sub(r"\'d ",  " ",   text)
    text = re.sub(r"\'ll ", " ",   text)
    text = re.sub(r"-",     " ",   text)
    text = re.sub(r"@",     " ",   text)
    text = re.sub("'",      "",    text)
    text = text.translate(TRANS_TAB)
    text = text.replace("..", "").strip()
    return text

# -------------------------------------------------------------

def main(csv_path: Path, out_path: Path, text_col: str):
    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in {csv_path.name}. "
            f"Available columns: {', '.join(df.columns.tolist())}"
        )

    cleaned = (
        df[text_col]
        .astype(str)
        .map(clean_str)          # apply cleaning
        .dropna()
        .loc[lambda s: s.str.len() > 0]  # drop empty after cleaning
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False, header=False)
    print(f"Wrote {len(cleaned):,} lines → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default="./data/reddit.csv",
                        help="Input CSV file")
    parser.add_argument("--out", type=Path, default="./data/reddit_for_VAE.txt",
                        help="Output text file (one cleaned comment per line)")
    parser.add_argument("--col", type=str, default="body",
                        help="Column name containing comment text "
                             "(e.g., body, comment, text)")
    args = parser.parse_args()
    main(args.csv, args.out, args.col)


