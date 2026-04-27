from pathlib import Path
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_PATH = Path("data/processed/scene_to_music_prompt_v2_structured.csv")
MODEL_DIR = Path("models/flan_t5_scene_to_music_prompt_v2_structured")
EXP_DIR = Path("experiments/flan_t5_scene_to_music_prompt_v2_structured")

PRED_PATH = EXP_DIR / "test_predictions.csv"
METRICS_PATH = EXP_DIR / "generation_metrics.csv"

MAX_INPUT_LEN = 384
MAX_GEN_LEN = 192

def split_tags(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [t.strip().lower().replace("_", " ").replace("-", " ") for t in str(x).split("|") if t.strip()]

def tag_recovery(pred, tags):
    pred = str(pred).lower()
    tags = split_tags(tags)
    if not tags:
        return None
    return sum(1 for t in tags if t in pred) / len(tags)

def any_tag_hit(pred, tags):
    pred = str(pred).lower()
    tags = split_tags(tags)
    if not tags:
        return None
    return int(any(t in pred for t in tags))

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    preds = []

    for text in tqdm(test_df["input_text"].tolist(), desc="Generating"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_GEN_LEN,
                num_beams=4,
                early_stopping=True,
            )

        preds.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

    test_df["predicted_music_description"] = preds

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=test_df["predicted_music_description"].tolist(),
        references=test_df["target_text"].tolist(),
        use_stemmer=True,
    )

    test_df["mood_recovery"] = test_df.apply(
        lambda r: tag_recovery(r["predicted_music_description"], r["moodtheme_tags"]),
        axis=1,
    )
    test_df["genre_recovery"] = test_df.apply(
        lambda r: tag_recovery(r["predicted_music_description"], r["genre_tags"]),
        axis=1,
    )
    test_df["mood_any_hit"] = test_df.apply(
        lambda r: any_tag_hit(r["predicted_music_description"], r["moodtheme_tags"]),
        axis=1,
    )
    test_df["genre_any_hit"] = test_df.apply(
        lambda r: any_tag_hit(r["predicted_music_description"], r["genre_tags"]),
        axis=1,
    )

    metrics = {
        "num_test_examples": len(test_df),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "avg_mood_recovery": test_df["mood_recovery"].mean(),
        "avg_genre_recovery": test_df["genre_recovery"].mean(),
        "mood_any_hit_rate": test_df["mood_any_hit"].mean(),
        "genre_any_hit_rate": test_df["genre_any_hit"].mean(),
    }

    test_df.to_csv(PRED_PATH, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    print("Saved predictions:", PRED_PATH)
    print("Saved metrics:", METRICS_PATH)
    print(metrics)
    print(test_df[["input_screenplay", "target_text", "predicted_music_description"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
