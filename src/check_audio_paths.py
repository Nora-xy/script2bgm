from pathlib import Path
import argparse
import pandas as pd


def find_audio_file(audio_root: Path, rel_path: str):
    direct = audio_root / rel_path
    if direct.exists():
        return direct

    rel = Path(rel_path)
    stem = rel.stem
    parent = rel.parent

    # MTG audio-low files are named like 1021915.low.mp3
    low_direct = audio_root / parent / f"{stem}.low.mp3"
    if low_direct.exists():
        return low_direct

    # Fallback search by exact filename
    filename = rel.name
    matches = list(audio_root.rglob(filename))
    if matches:
        return matches[0]

    # Fallback search by low-quality filename
    low_filename = f"{stem}.low.mp3"
    low_matches = list(audio_root.rglob(low_filename))
    if low_matches:
        return low_matches[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/mtg_subset.csv")
    parser.add_argument("--audio_root", required=True)
    parser.add_argument("--output", default="data/processed/mtg_subset_with_audio.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    audio_root = Path(args.audio_root)

    audio_paths = []
    exists = []

    for rel_path in df["path"].astype(str):
        found = find_audio_file(audio_root, rel_path)
        if found is None:
            audio_paths.append("")
            exists.append(False)
        else:
            audio_paths.append(str(found))
            exists.append(True)

    df["audio_path"] = audio_paths
    df["audio_exists"] = exists

    df.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Total:", len(df))
    print("Found:", sum(exists))
    print("Missing:", len(df) - sum(exists))
    print(df[["track_id", "path", "audio_path", "audio_exists"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
