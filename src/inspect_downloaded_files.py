from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

def inspect_tsv_files():
    print("\n=== Inspecting TSV files ===")
    tsv_files = list(RAW_DIR.rglob("*.tsv"))

    for path in tsv_files:
        print(f"\nFile: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for i in range(5):
                line = f.readline()
                if not line:
                    break
                print(f"Line {i + 1}: {line[:800].rstrip()}")

if __name__ == "__main__":
    inspect_tsv_files()
