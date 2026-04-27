from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT = Path("data/processed/mtg_subset.csv")
OUT_DIR = Path("data/processed/text_retrieval_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def recall_at_k(scores, k):
    topk = np.argsort(-scores, axis=1)[:, :k]
    targets = np.arange(scores.shape[0])
    hits = [(targets[i] in topk[i]) for i in range(scores.shape[0])]
    return float(np.mean(hits))


def mean_reciprocal_rank(scores):
    ranked = np.argsort(-scores, axis=1)
    targets = np.arange(scores.shape[0])
    rr = []
    for i in range(scores.shape[0]):
        rank = np.where(ranked[i] == targets[i])[0][0] + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def tag_overlap(a, b):
    set_a = set(str(a).split("|")) if pd.notna(a) and str(a).strip() else set()
    set_b = set(str(b).split("|")) if pd.notna(b) and str(b).strip() else set()
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def avg_tag_overlap_at_k(scores, df, k=5):
    topk = np.argsort(-scores, axis=1)[:, :k]
    overlaps = []

    for i in range(scores.shape[0]):
        query_tags = df.loc[i, "moodtheme_tags"]
        vals = []
        for j in topk[i]:
            vals.append(tag_overlap(query_tags, df.loc[j, "moodtheme_tags"]))
        overlaps.append(np.mean(vals))

    return float(np.mean(overlaps))


def main():
    df = pd.read_csv(INPUT)

    # Use only rows with known split.
    df = df[df["split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    scene_texts = df["scene_text"].fillna("").tolist()
    music_texts = df["music_description"].fillna("").tolist()

    print("Loaded examples:", len(df))
    print("Model:", MODEL_NAME)

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

    print("Computing cosine similarity...")
    scores = cosine_similarity(scene_emb, music_emb)

    r1 = recall_at_k(scores, 1)
    r5 = recall_at_k(scores, 5)
    r10 = recall_at_k(scores, 10)
    mrr = mean_reciprocal_rank(scores)
    tag5 = avg_tag_overlap_at_k(scores, df, k=5)

    results = {
        "num_examples": len(df),
        "model": MODEL_NAME,
        "Recall@1": r1,
        "Recall@5": r5,
        "Recall@10": r10,
        "MRR": mrr,
        "Avg_MoodTheme_TagOverlap@5": tag5,
    }

    print("\n=== Retrieval Baseline Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    pd.DataFrame([results]).to_csv(OUT_DIR / "metrics.csv", index=False)

    # Save example retrievals
    topk = np.argsort(-scores, axis=1)[:, :5]
    rows = []

    for i in range(min(20, len(df))):
        for rank, j in enumerate(topk[i], start=1):
            rows.append({
                "query_index": i,
                "rank": rank,
                "query_track_id": df.loc[i, "track_id"],
                "retrieved_track_id": df.loc[j, "track_id"],
                "is_exact_match": int(i == j),
                "score": float(scores[i, j]),
                "query_scene": df.loc[i, "scene_text"],
                "query_tags": df.loc[i, "moodtheme_tags"],
                "retrieved_music_description": df.loc[j, "music_description"],
                "retrieved_tags": df.loc[j, "moodtheme_tags"],
            })

    examples = pd.DataFrame(rows)
    examples.to_csv(OUT_DIR / "example_retrievals.csv", index=False)

    print("\nSaved metrics to:", OUT_DIR / "metrics.csv")
    print("Saved examples to:", OUT_DIR / "example_retrievals.csv")

    print("\n=== Example Retrievals Preview ===")
    for i in range(min(3, len(df))):
        print("\nQUERY:")
        print(df.loc[i, "scene_text"])
        print("TRUE TRACK:", df.loc[i, "track_id"], "| TAGS:", df.loc[i, "moodtheme_tags"])
        print("TOP 5:")
        for rank, j in enumerate(topk[i], start=1):
            print(
                rank,
                df.loc[j, "track_id"],
                "score=",
                round(float(scores[i, j]), 4),
                "| tags=",
                df.loc[j, "moodtheme_tags"],
                "| exact=",
                i == j,
            )


if __name__ == "__main__":
    main()
