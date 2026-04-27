from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

AUDIO_EMBED_DIR = Path("data/embeddings/clap_audio_subset")
PRED_PATH = Path("experiments/flan_t5_scene_to_music_prompt_leakage_reduced_v1/test_predictions.csv")
OUT_DIR = Path("experiments/flan_t5_to_clap_audio_retrieval_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "laion/clap-htsat-fused"

QUERY_COLUMNS = {
    "direct_screenplay": "input_screenplay",
    "oracle_target_music_description": "target_text",
    "finetuned_generated_music_description": "predicted_music_description",
}

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
    raise TypeError(f"Unexpected output type: {type(output)}")

def encode_texts(texts, model, processor, device, batch_size=16):
    all_embeds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="CLAP text encoding"):
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

def recall_at_k(scores, target_indices, k):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices
    targets = torch.tensor(target_indices).unsqueeze(1)
    return (topk == targets).any(dim=1).float().mean().item()

def mean_reciprocal_rank(scores, target_indices):
    ranked = torch.argsort(scores, dim=1, descending=True)
    rr = []
    for i, target_idx in enumerate(target_indices):
        rank = (ranked[i] == target_idx).nonzero(as_tuple=True)[0].item() + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))

def tag_hit_at_k(scores, query_df, audio_meta, target_col, k):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices.numpy()
    hits = []

    for i in range(scores.shape[0]):
        q_tags = split_tags(query_df.loc[i, target_col])
        if not q_tags:
            continue

        found = False
        for j in topk[i]:
            cand_tags = split_tags(audio_meta.loc[j, target_col])
            if q_tags & cand_tags:
                found = True
                break

        hits.append(found)

    return float(np.mean(hits)) if hits else 0.0

def save_examples(method_name, scores, query_df, audio_meta, k=5):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices.numpy()
    rows = []

    for i in range(min(50, scores.shape[0])):
        for rank, j in enumerate(topk[i], start=1):
            rows.append({
                "method": method_name,
                "query_index": i,
                "rank": rank,
                "query_track_id": query_df.loc[i, "track_id"],
                "retrieved_track_id": audio_meta.loc[j, "track_id"],
                "exact_match": int(query_df.loc[i, "track_id"] == audio_meta.loc[j, "track_id"]),
                "score": float(scores[i, j]),
                "input_screenplay": query_df.loc[i, "input_screenplay"],
                "target_text": query_df.loc[i, "target_text"],
                "predicted_music_description": query_df.loc[i, "predicted_music_description"],
                "query_mood": query_df.loc[i, "moodtheme_tags"],
                "query_genre": query_df.loc[i, "genre_tags"],
                "retrieved_mood": audio_meta.loc[j, "moodtheme_tags"],
                "retrieved_genre": audio_meta.loc[j, "genre_tags"],
                "retrieved_audio_path": audio_meta.loc[j, "audio_path"],
            })

    pd.DataFrame(rows).to_csv(OUT_DIR / f"examples_{method_name}.csv", index=False)

def main():
    print("Loading audio embeddings...")
    audio_emb = torch.load(AUDIO_EMBED_DIR / "clap_audio_embeds.pt")
    audio_meta = pd.read_csv(AUDIO_EMBED_DIR / "metadata.csv")

    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)

    print("Audio embeddings:", tuple(audio_emb.shape))
    print("Audio metadata:", len(audio_meta))

    query_df = pd.read_csv(PRED_PATH)
    query_df = query_df[query_df["split"] == "test"].reset_index(drop=True)

    # Keep only test examples whose correct track exists in the audio candidate pool.
    track_to_idx = {tid: i for i, tid in enumerate(audio_meta["track_id"].astype(str).tolist())}

    keep_rows = []
    target_indices = []

    for i, row in query_df.iterrows():
        tid = str(row["track_id"])
        if tid in track_to_idx:
            keep_rows.append(i)
            target_indices.append(track_to_idx[tid])

    query_df = query_df.loc[keep_rows].reset_index(drop=True)

    print("Queries kept:", len(query_df))
    print("Candidate audio pool:", len(audio_meta))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    all_metrics = []

    for method_name, col in QUERY_COLUMNS.items():
        print("\n=== Method:", method_name, "column:", col, "===")

        texts = query_df[col].fillna("").astype(str).tolist()
        text_emb = encode_texts(texts, model, processor, device, batch_size=16)

        scores = text_emb @ audio_emb.T

        metrics = {
            "method": method_name,
            "query_column": col,
            "num_queries": len(query_df),
            "num_audio_candidates": len(audio_meta),
            "ExactRecall@1": recall_at_k(scores, target_indices, 1),
            "ExactRecall@5": recall_at_k(scores, target_indices, 5),
            "ExactRecall@10": recall_at_k(scores, target_indices, 10),
            "MRR": mean_reciprocal_rank(scores, target_indices),
            "MoodHit@5": tag_hit_at_k(scores, query_df, audio_meta, "moodtheme_tags", 5),
            "GenreHit@5": tag_hit_at_k(scores, query_df, audio_meta, "genre_tags", 5),
        }

        print(metrics)
        all_metrics.append(metrics)

        save_examples(method_name, scores, query_df, audio_meta, k=5)

    pd.DataFrame(all_metrics).to_csv(OUT_DIR / "metrics.csv", index=False)

    with open(OUT_DIR / "config.json", "w") as f:
        f.write('''{
  "experiment_name": "flan_t5_to_clap_audio_retrieval_v1",
  "description": "Compares three CLAP text-to-audio retrieval query types: direct screenplay input, oracle target music description, and fine-tuned FLAN-T5 generated music description.",
  "audio_pool": "1000 MTG-Jamendo audio-low clips from mtg_subset_with_audio.csv",
  "clap_model": "laion/clap-htsat-fused",
  "generation_model": "fine-tuned google/flan-t5-small",
  "metrics": ["ExactRecall@1", "ExactRecall@5", "ExactRecall@10", "MRR", "MoodHit@5", "GenreHit@5"]
}''')

    print("\nSaved metrics:", OUT_DIR / "metrics.csv")
    print("Saved examples to:", OUT_DIR)

if __name__ == "__main__":
    main()
