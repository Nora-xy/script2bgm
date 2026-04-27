from pathlib import Path
import json
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATA_PATH = Path("data/processed/scene_to_music_prompt_leakage_reduced.csv")
MODEL_DIR = Path("models/flan_t5_scene_to_music_prompt_leakage_reduced")
EXP_DIR = Path("experiments/flan_t5_scene_to_music_prompt_leakage_reduced_v1")

PRED_PATH = EXP_DIR / "test_predictions.csv"
TEST_METRICS_PATH = EXP_DIR / "test_metrics_recovered.csv"
TAG_METRICS_PATH = EXP_DIR / "tag_recovery_metrics.csv"
TRAIN_LOG_PATH = EXP_DIR / "training_log_recovered.csv"
CONFIG_PATH = EXP_DIR / "config.json"

MAX_INPUT_LEN = 320
MAX_GEN_LEN = 128

def split_tags(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [t.strip().lower().replace("_", " ").replace("-", " ") for t in str(x).split("|") if t.strip()]

def tag_recovery(pred, tags):
    pred = str(pred).lower()
    tags = split_tags(tags)
    if not tags:
        return None
    hits = sum(1 for t in tags if t in pred)
    return hits / len(tags)

def any_tag_hit(pred, tags):
    pred = str(pred).lower()
    tags = split_tags(tags)
    if not tags:
        return None
    return int(any(t in pred for t in tags))

def recover_training_log():
    state_files = sorted(MODEL_DIR.glob("checkpoint-*/trainer_state.json"))
    if not state_files:
        state_file = MODEL_DIR / "trainer_state.json"
        state_files = [state_file] if state_file.exists() else []

    if not state_files:
        print("No trainer_state.json found. Skipping training log recovery.")
        return

    latest = state_files[-1]
    with open(latest, "r") as f:
        state = json.load(f)

    logs = pd.DataFrame(state.get("log_history", []))
    if len(logs) > 0:
        logs.to_csv(TRAIN_LOG_PATH, index=False)
        print("Saved training log:", TRAIN_LOG_PATH)
    else:
        print("trainer_state.json found but log_history is empty.")

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test data...")
    df = pd.read_csv(DATA_PATH)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print("Test examples:", len(test_df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading fine-tuned model...")
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

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(pred)

    test_df["predicted_music_description"] = preds
    test_df.to_csv(PRED_PATH, index=False)
    print("Saved predictions:", PRED_PATH)

    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=test_df["predicted_music_description"].tolist(),
        references=test_df["target_text"].tolist(),
        use_stemmer=True,
    )

    test_metrics = {
        "num_test_examples": len(test_df),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
    }
    pd.DataFrame([test_metrics]).to_csv(TEST_METRICS_PATH, index=False)
    print("Saved test metrics:", TEST_METRICS_PATH)
    print(test_metrics)

    print("Computing tag recovery...")
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

    tag_metrics = {
        "num_test_examples": len(test_df),
        "avg_mood_recovery": test_df["mood_recovery"].mean(),
        "avg_genre_recovery": test_df["genre_recovery"].mean(),
        "mood_any_hit_rate": test_df["mood_any_hit"].mean(),
        "genre_any_hit_rate": test_df["genre_any_hit"].mean(),
    }
    pd.DataFrame([tag_metrics]).to_csv(TAG_METRICS_PATH, index=False)
    print("Saved tag recovery metrics:", TAG_METRICS_PATH)
    print(tag_metrics)

    # Save predictions again with tag-recovery columns included.
    test_df.to_csv(PRED_PATH, index=False)

    recover_training_log()

    config = {
        "experiment_name": "flan_t5_scene_to_music_prompt_leakage_reduced_v1",
        "model": "google/flan-t5-small",
        "saved_model_dir": str(MODEL_DIR),
        "dataset": "scene_to_music_prompt_leakage_reduced.csv",
        "task": "script_like_scene_to_detailed_music_description",
        "input": "leakage-reduced screenplay-style scene text",
        "target": "detailed music description generated from MTG-Jamendo mood/theme and genre tags",
        "train_examples": int((df["split"] == "train").sum()),
        "validation_examples": int((df["split"] == "validation").sum()),
        "test_examples": int((df["split"] == "test").sum()),
        "notes": "Training completed successfully. Original Trainer evaluation crashed during token decoding, so test generation and metrics were recovered with manual model.generate evaluation."
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print("Saved config:", CONFIG_PATH)

    print("\n=== Example Predictions ===")
    cols = ["input_screenplay", "target_text", "predicted_music_description", "moodtheme_tags", "genre_tags"]
    print(test_df[cols].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
