from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT = Path("data/processed/mtg_subset.csv")
OUT_DIR = Path("data/processed/text_retrieval_eval_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_tags(x):
    if pd.isna(x) or not str(x).strip():
        return set()
    return set(t.strip() for t in str(x).split("|") if t.strip())


def recall_at_k(scores, k):
    topk = np.argsort(-scores, axis=1)[:, :k]
    targets = np.arange(scores.shape[0])
    return float(np.mean([targets[i] in topk[i] for i in range(scores.shape[0])]))


def mrr(scores):
    ranked = np.argsort(-scores, axis=1)
    targets = np.arange(scores.shape[0])
    vals = []
    for i in range(scores.shape[0]):
        rank = np.where(ranked[i] == targets[i])[0][0] + 1
        vals.append(1 / rank)
    return float(np.mean(vals))


def hit_at_k(scores, df, tag_col, k):
    topk = np.argsort(-scores, axis=1)[:, :k]
    hits = []

    for i in range(scores.shape[0]):
        query_tags = split_tags(df.loc[i, tag_col])
        if not query_tags:
            continue

        found = False
        for j in topk[i]:
            cand_tags = split_tags(df.loc[j, tag_col])
            if query_tags & cand_tags:
                found = True
                break

        hits.append(found)

    return float(np.mean(hits)) if hits else 0.0


def avg_best_jaccard_at_k(scores, df, tag_col, k):
    topk = np.argsort(-scores, axis=1)[:, :k]
    vals = []

    for i in range(scores.shape[0]):
        query_tags = split_tags(df.loc[i, tag_col])
        if not query_tags:
            continue

        best = 0.0
        for j in topk[i]:
            cand_tags = split_tags(df.loc[j, tag_col])
            union = query_tags | cand_tags
            inter = query_tags & cand_tags
            if union:
                best = max(best, len(inter) / len(union))
        vals.append(best)

    return float(np.mean(vals)) if vals else 0.0


def main():
    df = pd.read_csv(INPUT)
    df = df[df["split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    scene_texts = df["scene_text"].fillna("").tolist()
    music_texts = df["music_description"].fillna("").tolist()

    print("Loaded examples:", len(df))
    print("Loading model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Encoding scene_text...")
    scene_emb = model.encode(
        scene_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Encoding music_description...")
    music_emb = model.encode(
        music_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    scores = cosine_similarity(scene_emb, music_emb)

    results = {
        "num_examples": len(df),
        "model": MODEL_NAME,
        "ExactRecall@1": recall_at_k(scores, 1),
        "ExactRecall@5": recall_at_k(scores, 5),
        "ExactRecall@10": recall_at_k(scores, 10),
        "MRR": mrr(scores),
        "MoodHit@1": hit_at_k(scores, df, "moodtheme_tags", 1),
        "MoodHit@5": hit_at_k(scores, df, "moodtheme_tags", 5),
        "MoodHit@10": hit_at_k(scores, df, "moodtheme_tags", 10),
        "GenreHit@1": hit_at_k(scores, df, "genre_tags", 1),
        "GenreHit@5": hit_at_k(scores, df, "genre_tags", 5),
        "GenreHit@10": hit_at_k(scores, df, "genre_tags", 10),
        "BestMoodJaccard@5": avg_best_jaccard_at_k(scores, df, "moodtheme_tags", 5),
        "BestGenreJaccard@5": avg_best_jaccard_at_k(scores, df, "genre_tags", 5)
    }

    print("\n=== Evaluation v2 Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    pd.DataFrame([results]).to_csv(OUT_DIR / "metrics_v2.csv", index=False)

    topk = np.argsort(-scores, axis=1)[:, :5]
    rows = []

    for i in range(min(30, len(df))):
        for rank, j in enumerate(topk[i], start=1):
            rows.append({
                "query_index": i,
                "rank": rank,
                "query_track_id": df.loc[i, "track_id"],
                "retrieved_track_id": df.loc[j, "track_id"],
                "exact_match": int(i == j),
                "score": float(scores[i, j]),
                "query_scene": df.loc[i, "scene_text"],
                "query_mood": df.loc[i, "moodtheme_tags"],
                "query_genre": df.loc[i, "genre_tags"],
                "retrieved_description": df.loc[j, "music_description"],
                "retrieved_mood": df.loc[j, "moodtheme_tags"],
                "retrieved_genre": df.loc[j, "genre_tags"],
                "mood_overlap": "|".join(sorted(split_tags(df.loc[i, "moodtheme_tags"]) & split_tags(df.loc[j, "moodtheme_tags"]))),
                "genre_overlap": "|".join(sorted(split_tags(df.loc[i, "genre_tags"]) & split_tags(df.loc[j, "genre_tags"])))
            })

    pd.DataFrame(rows).to_csv(OUT_DIR / "retrieval_examples_v2.csv", index=False)

    print("\nSaved:")
    print(OUT_DIR / "metrics_v2.csv")
    print(OUT_DIR / "retrieval_examples_v2.csv")


if __name__ == "__main__":
    main()
