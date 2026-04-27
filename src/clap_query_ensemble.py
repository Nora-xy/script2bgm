from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

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
    raise TypeError(f"Unexpected output type: {type(output)}")

def encode_texts(texts, model, processor, device, batch_size=16):
    all_embeds = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
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

def mrr(scores, target_indices):
    ranked = torch.argsort(scores, dim=1, descending=True)
    vals = []
    for i, target in enumerate(target_indices):
        rank = (ranked[i] == target).nonzero(as_tuple=True)[0].item() + 1
        vals.append(1.0 / rank)
    return float(np.mean(vals))

def tag_hit_at_k(scores, query_df, audio_meta, tag_col, k):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices.numpy()
    hits = []

    for i in range(scores.shape[0]):
        q_tags = split_tags(query_df.loc[i, tag_col])
        if not q_tags:
            continue

        found = False
        for j in topk[i]:
            cand_tags = split_tags(audio_meta.loc[j, tag_col])
            if q_tags & cand_tags:
                found = True
                break
        hits.append(found)

    return float(np.mean(hits)) if hits else 0.0

def compute_metrics(name, scores, query_df, audio_meta, target_indices):
    return {
        "method": name,
        "num_queries": len(query_df),
        "num_audio_candidates": len(audio_meta),
        "ExactRecall@1": recall_at_k(scores, target_indices, 1),
        "ExactRecall@5": recall_at_k(scores, target_indices, 5),
        "ExactRecall@10": recall_at_k(scores, target_indices, 10),
        "MRR": mrr(scores, target_indices),
        "MoodHit@5": tag_hit_at_k(scores, query_df, audio_meta, "moodtheme_tags", 5),
        "GenreHit@5": tag_hit_at_k(scores, query_df, audio_meta, "genre_tags", 5),
    }

def save_examples(name, scores, query_df, audio_meta, out_dir, k=5):
    topk = torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices.numpy()
    rows = []

    for i in range(min(50, scores.shape[0])):
        for rank, j in enumerate(topk[i], start=1):
            rows.append({
                "method": name,
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

    pd.DataFrame(rows).to_csv(out_dir / f"examples_{name}.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", default="experiments/flan_t5_scene_to_music_prompt_leakage_reduced_v1/test_predictions.csv")
    parser.add_argument("--audio_embed_dir", default="data/embeddings/clap_audio_subset")
    parser.add_argument("--out_dir", default="experiments/clap_query_ensemble_v1")
    args = parser.parse_args()

    pred_path = Path(args.pred_path)
    audio_embed_dir = Path(args.audio_embed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_emb = torch.load(audio_embed_dir / "clap_audio_embeds.pt")
    audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)
    audio_meta = pd.read_csv(audio_embed_dir / "metadata.csv")

    query_df = pd.read_csv(pred_path)
    query_df = query_df[query_df["split"] == "test"].reset_index(drop=True)

    track_to_idx = {tid: i for i, tid in enumerate(audio_meta["track_id"].astype(str).tolist())}

    keep = []
    target_indices = []

    for i, row in query_df.iterrows():
        tid = str(row["track_id"])
        if tid in track_to_idx:
            keep.append(i)
            target_indices.append(track_to_idx[tid])

    query_df = query_df.loc[keep].reset_index(drop=True)

    print("Queries:", len(query_df))
    print("Audio candidates:", len(audio_meta))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    direct_texts = query_df["input_screenplay"].fillna("").astype(str).tolist()
    pred_texts = query_df["predicted_music_description"].fillna("").astype(str).tolist()
    target_texts = query_df["target_text"].fillna("").astype(str).tolist()

    print("\nEncoding direct screenplay...")
    direct_emb = encode_texts(direct_texts, model, processor, device)

    print("\nEncoding predicted descriptions...")
    pred_emb = encode_texts(pred_texts, model, processor, device)

    print("\nEncoding oracle target descriptions...")
    target_emb = encode_texts(target_texts, model, processor, device)

    direct_scores = direct_emb @ audio_emb.T
    pred_scores = pred_emb @ audio_emb.T
    target_scores = target_emb @ audio_emb.T

    rows = []

    base_scores = {
        "direct_screenplay": direct_scores,
        "finetuned_generated_description": pred_scores,
        "oracle_target_description": target_scores,
    }

    for name, scores in base_scores.items():
        metrics = compute_metrics(name, scores, query_df, audio_meta, target_indices)
        rows.append(metrics)
        save_examples(name, scores, query_df, audio_meta, out_dir)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scores = alpha * direct_scores + (1.0 - alpha) * pred_scores
        name = f"score_fusion_alpha_{alpha:.2f}_direct"
        metrics = compute_metrics(name, scores, query_df, audio_meta, target_indices)
        rows.append(metrics)
        save_examples(name, scores, query_df, audio_meta, out_dir)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "metrics.csv", index=False)

    with open(out_dir / "config.json", "w") as f:
        f.write('''{
  "experiment_name": "clap_query_ensemble_v1",
  "description": "Tests CLAP retrieval using direct screenplay queries, fine-tuned FLAN-T5 generated music descriptions, oracle target music descriptions, and score-level ensembles between direct screenplay and generated descriptions.",
  "clap_model": "laion/clap-htsat-fused",
  "audio_pool": "1000 MTG-Jamendo audio-low CLAP embeddings",
  "fusion": "score = alpha * direct_screenplay_score + (1 - alpha) * generated_description_score"
}''')

    print("\nSaved:", out_dir / "metrics.csv")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()
