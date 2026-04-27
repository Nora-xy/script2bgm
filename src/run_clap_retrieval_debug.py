import torch
import pandas as pd
from pathlib import Path

TEXT_EMB_PATH = "data/embeddings/clap_text_debug/clap_text_embeds.pt"
AUDIO_EMB_PATH = "data/embeddings/clap_audio_debug.pt"
META_PATH = "data/processed/debug_audio_subset.csv"

OUT_DIR = Path("experiments/clap_audio_debug_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

text_emb = torch.load(TEXT_EMB_PATH)
audio_emb = torch.load(AUDIO_EMB_PATH)
df = pd.read_csv(META_PATH)

text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
audio_emb = torch.nn.functional.normalize(audio_emb, dim=-1)

scores = text_emb @ audio_emb.T
topk = torch.topk(scores, k=min(5, len(df)), dim=1).indices

rows = []
for i in range(len(df)):
    for rank, j in enumerate(topk[i].tolist(), start=1):
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
            "retrieved_mood": df.loc[j, "moodtheme_tags"],
            "retrieved_genre": df.loc[j, "genre_tags"],
            "retrieved_audio_path": df.loc[j, "audio_path"]
        })

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / "retrieval_examples.csv", index=False)

exact_r1 = (topk[:, 0] == torch.arange(len(df))).float().mean().item()
exact_r5 = (topk == torch.arange(len(df)).unsqueeze(1)).any(dim=1).float().mean().item()

metrics = pd.DataFrame([{
    "experiment": "clap_audio_debug_v1",
    "num_examples": len(df),
    "model": "laion/clap-htsat-fused",
    "query": "scene_text",
    "target": "audio",
    "ExactRecall@1": exact_r1,
    "ExactRecall@5": exact_r5
}])
metrics.to_csv(OUT_DIR / "metrics.csv", index=False)

with open(OUT_DIR / "config.json", "w") as f:
    f.write('''{
  "experiment_name": "clap_audio_debug_v1",
  "date": "2026-04-25",
  "dataset": "debug_audio_subset.csv",
  "num_examples": 20,
  "model": "laion/clap-htsat-fused",
  "query_field": "scene_text",
  "target": "MTG-Jamendo audio-low mp3",
  "similarity": "cosine_similarity",
  "description": "First successful CLAP text-to-audio retrieval debug run. Uses generated scene descriptions as text queries and CLAP audio embeddings from matched MTG-Jamendo audio files.",
  "limitations": "Small 20-example debug subset only. Main full-subset run should use generated detailed music descriptions and/or 1000-example subset."
}''')

print("Saved:")
print(OUT_DIR / "metrics.csv")
print(OUT_DIR / "retrieval_examples.csv")
print(OUT_DIR / "config.json")
print(metrics.to_string(index=False))