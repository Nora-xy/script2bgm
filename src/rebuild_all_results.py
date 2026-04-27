from pathlib import Path
import pandas as pd
import math

OUT = Path("experiments/all_results.csv")

COLUMNS = [
    "experiment",
    "date",
    "dataset",
    "model",
    "query",
    "target",
    "ExactRecall@1",
    "ExactRecall@5",
    "ExactRecall@10",
    "MRR",
    "MoodHit@5",
    "GenreHit@5",
    "notes"
]

def safe_read(path):
    path = Path(path)
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return pd.read_csv(path).iloc[0].to_dict()

def val(d, *keys):
    if d is None:
        return "NA"
    for k in keys:
        if k in d:
            return d[k]
    return "NA"

rows = []

# 1. MiniLM text baseline v1
d = safe_read("experiments/text_baseline_v1/metrics.csv")
if d:
    rows.append({
        "experiment": "text_baseline_v1",
        "date": "2026-04-24",
        "dataset": "mtg_subset.csv",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "query": "scene_text",
        "target": "music_description",
        "ExactRecall@1": val(d, "Recall@1", "ExactRecall@1"),
        "ExactRecall@5": val(d, "Recall@5", "ExactRecall@5"),
        "ExactRecall@10": val(d, "Recall@10", "ExactRecall@10"),
        "MRR": val(d, "MRR"),
        "MoodHit@5": "NA",
        "GenreHit@5": "NA",
        "notes": "first text-only retrieval baseline"
    })

# 2. MiniLM text evaluation v2
d = safe_read("experiments/text_eval_v2/metrics_v2.csv")
if d:
    rows.append({
        "experiment": "text_eval_v2",
        "date": "2026-04-24",
        "dataset": "mtg_subset.csv",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "query": "scene_text",
        "target": "music_description",
        "ExactRecall@1": val(d, "ExactRecall@1"),
        "ExactRecall@5": val(d, "ExactRecall@5"),
        "ExactRecall@10": val(d, "ExactRecall@10"),
        "MRR": val(d, "MRR"),
        "MoodHit@5": val(d, "MoodHit@5"),
        "GenreHit@5": val(d, "GenreHit@5"),
        "notes": "text-only retrieval with tag-level metrics"
    })

# 3. CLAP text-only retrieval
d = safe_read("experiments/clap_text_retrieval_v1/metrics.csv")
if d:
    rows.append({
        "experiment": "clap_text_retrieval_v1",
        "date": "2026-04-24",
        "dataset": "mtg_subset.csv",
        "model": "laion/clap-htsat-fused",
        "query": "scene_text",
        "target": "music_description",
        "ExactRecall@1": val(d, "ExactRecall@1"),
        "ExactRecall@5": val(d, "ExactRecall@5"),
        "ExactRecall@10": val(d, "ExactRecall@10"),
        "MRR": val(d, "MRR"),
        "MoodHit@5": val(d, "MoodHit@5"),
        "GenreHit@5": val(d, "GenreHit@5"),
        "notes": "CLAP text encoder baseline; text-to-text only"
    })

# 4. CLAP audio debug retrieval
d = safe_read("experiments/clap_audio_debug_v1/metrics.csv")
if d:
    # Try to compute MoodHit@5 / GenreHit@5 from saved retrieval examples.
    ex_path = Path("experiments/clap_audio_debug_v1/retrieval_examples.csv")
    mood_hit5 = "NA"
    genre_hit5 = "NA"

    if ex_path.exists():
        ex = pd.read_csv(ex_path)

        def split_tags(x):
            if pd.isna(x) or not str(x).strip():
                return set()
            return set(t.strip() for t in str(x).split("|") if t.strip())

        mood_hits = []
        genre_hits = []

        for qid, group in ex.groupby("query_index"):
            q_mood = split_tags(group.iloc[0].get("query_mood", ""))
            q_genre = split_tags(group.iloc[0].get("query_genre", ""))

            mood_found = False
            genre_found = False

            for _, r in group.iterrows():
                if q_mood and (q_mood & split_tags(r.get("retrieved_mood", ""))):
                    mood_found = True
                if q_genre and (q_genre & split_tags(r.get("retrieved_genre", ""))):
                    genre_found = True

            if q_mood:
                mood_hits.append(mood_found)
            if q_genre:
                genre_hits.append(genre_found)

        if mood_hits:
            mood_hit5 = sum(mood_hits) / len(mood_hits)
        if genre_hits:
            genre_hit5 = sum(genre_hits) / len(genre_hits)

    rows.append({
        "experiment": "clap_audio_debug_v1",
        "date": "2026-04-25",
        "dataset": "debug_audio_subset.csv",
        "model": "laion/clap-htsat-fused",
        "query": "scene_text",
        "target": "MTG-Jamendo audio-low",
        "ExactRecall@1": val(d, "ExactRecall@1"),
        "ExactRecall@5": val(d, "ExactRecall@5"),
        "ExactRecall@10": "NA",
        "MRR": "NA",
        "MoodHit@5": mood_hit5,
        "GenreHit@5": genre_hit5,
        "notes": "first successful CLAP text-to-audio retrieval debug run; 20 examples only"
    })

out = pd.DataFrame(rows, columns=COLUMNS)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("Saved:", OUT)
print(out.to_string(index=False))
