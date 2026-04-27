import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

INPUT = "data/processed/debug_subset.csv"
OUT_DIR = Path("data/audio_debug_manual")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://cdn.jamendo.com/?trackid={}&format=mp31&from=app-97dab294"

def main():
    df = pd.read_csv(INPUT)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        track_id = str(row["track_id"]).replace("track_", "")
        url = BASE_URL.format(track_id)
        out_path = OUT_DIR / f"{track_id}.mp3"

        if out_path.exists():
            continue

        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 10000:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Skip {track_id}, status={r.status_code}, size={len(r.content)}")
        except Exception as e:
            print(f"Error {track_id}: {e}")

if __name__ == "__main__":
    main()
