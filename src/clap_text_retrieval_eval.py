from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

INPUT = Path("data/processed/mtg_subset.csv")
OUT_DIR = Path("experiments/clap_text_retrieval_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "laion/clap-htsat-fused"


def split_tags(x):
    if pd.isna(x) or not str(x).strip():
        return set()
    return set(t.strip() for t in str(x).split("|") if t.strip())


def unwrap_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected model output type: {type(output)}")


def encode_texts(model, processor, texts, device, batch_size=16):
    all_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = processor(
            text=batch,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_text_features(**inputs)
            emb = unwrap_embedding(emb)
            emb = torch.nn.functional.normalize(emb, dim=-1)

        all_embeds.append(emb.cpu())

    return torch.cat(all_embeds, dim=0)


def recall_at_k(scores, k):
    topk = torch.topk(scores, k=k, dim=1).indices
    targets = torch.arange(scores.size(0)).unsqueeze(1)
    return (topk == targets).any(dim=1).float().mean().item()


def mrr(scores):
    ranked = torch.argsort(scores, dim=1, descending=True)
    vals = []
    for i in range(scores.size(0)):
        rank = (ranked[i] == i).nonzero(as_tuple=True)[0].item() + 1
        vals.append(1.0 / rank)
    return float(np.mean(vals))


def hit_at_k(scores, df, tag_col, k):
    topk = torch.topk(scores, k=k, dim=1).indices.numpy()
    hits = []

    for i in range(scores.size(0)):
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


def main():
    df = pd.read_csv(INPUT)
    df = df[df["split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    scene_texts = df["scene_text"].fillna("").astype(str).tolist()
    music_texts = df["music_description"].fillna("").astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Model:", MODEL_NAME)
    print("Examples:", len(df))

    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print("Encoding scene_text with CLAP text encoder...")
    scene_emb = encode_texts(model, processor, scene_texts, device)

    print("Encoding music_description with CLAP text encoder...")
    music_emb = encode_texts(model, processor, music_texts, device)

    scores = scene_emb @ music_emb.T

    results = {
        "experiment": "clap_text_retrieval_v1",
        "num_examples": len(df),
        "model": MODEL_NAME,
        "query": "scene_text",
        "target": "music_description",
        "ExactRecall@1": recall_at_k(scores, 1),
        "ExactRecall@5": recall_at_k(scores, 5),
        "ExactRecall@10": recall_at_k(scores, 10),
        "MRR": mrr(scores),
        "MoodHit@5": hit_at_k(scores, df, "moodtheme_tags", 5),
        "GenreHit@5": hit_at_k(scores, df, "genre_tags", 5)
    }

    print("\n=== CLAP Text Retrieval Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    pd.DataFrame([results]).to_csv(OUT_DIR / "metrics.csv", index=False)

    topk = torch.topk(scores, k=5, dim=1).indices.numpy()
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
                "retrieved_genre": df.loc[j, "genre_tags"]
            })

    pd.DataFrame(rows).to_csv(OUT_DIR / "examples.csv", index=False)

    with open(OUT_DIR / "config.json", "w") as f:
        f.write('''{
  "experiment_name": "clap_text_retrieval_v1",
  "date": "2026-04-24",
  "dataset": "mtg_subset.csv",
  "num_examples": 1000,
  "model": "laion/clap-htsat-fused",
  "query_field": "scene_text",
  "target_field": "music_description",
  "similarity": "cosine_similarity",
  "description": "Text-only retrieval baseline using the CLAP text encoder for both scene descriptions and music descriptions. This compares CLAP text embeddings against the earlier MiniLM baseline before adding audio embeddings."
}''')

    print("\nSaved results to:", OUT_DIR)


if __name__ == "__main__":
    main()
