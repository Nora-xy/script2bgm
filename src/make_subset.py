from pathlib import Path
import pandas as pd
import argparse

INPUT = Path("data/processed/mtg_moodtheme_manifest.csv")
OUTPUT = Path("data/processed/mtg_subset.csv")

def main(n_train, n_val, n_test):
    df = pd.read_csv(INPUT)

    parts = []

    for split, n in [("train", n_train), ("validation", n_val), ("test", n_test)]:
        sub = df[df["split"] == split]
        sub = sub.sample(n=min(n, len(sub)), random_state=42)
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print(out["split"].value_counts())
    print("\nExample:")
    print(out.head(5).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=700)
    parser.add_argument("--n_val", type=int, default=150)
    parser.add_argument("--n_test", type=int, default=150)
    args = parser.parse_args()

    main(args.n_train, args.n_val, args.n_test)
