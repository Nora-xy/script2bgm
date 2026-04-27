from pathlib import Path
import pandas as pd
import numpy as np
import evaluate
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_PATH = Path("data/processed/scene_to_music_prompt_leakage_reduced.csv")
MODEL_NAME = "google/flan-t5-small"
OUT_DIR = Path("models/flan_t5_scene_to_music_prompt_leakage_reduced")
EXP_DIR = Path("experiments/flan_t5_scene_to_music_prompt_leakage_reduced_v1")

MAX_INPUT_LEN = 320
MAX_TARGET_LEN = 128

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df),
    })

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading data...")
    datasets = load_data()
    print(datasets)

    print("Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = datasets.map(
        preprocess,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {k: round(v, 4) for k, v in result.items()}

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        weight_decay=0.01,

        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,

        report_to="none",
        fp16=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving best model...")
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    pd.DataFrame([test_metrics]).to_csv(EXP_DIR / "test_metrics.csv", index=False)

    logs = pd.DataFrame(trainer.state.log_history)
    logs.to_csv(EXP_DIR / "training_log.csv", index=False)

    with open(EXP_DIR / "config.json", "w") as f:
        f.write(f'''{{
  "experiment_name": "flan_t5_scene_to_music_prompt_leakage_reduced_v1",
  "model": "{MODEL_NAME}",
  "dataset": "scene_to_music_prompt_leakage_reduced.csv",
  "task": "script_like_scene_to_detailed_music_description",
  "train_examples": {len(datasets["train"])},
  "validation_examples": {len(datasets["validation"])},
  "test_examples": {len(datasets["test"])},
  "epochs": 4,
  "learning_rate": 5e-5,
  "batch_size": 4,
  "gradient_accumulation_steps": 2,
  "max_input_length": {MAX_INPUT_LEN},
  "max_target_length": {MAX_TARGET_LEN},
  "leakage_control": "Input screenplay-style text is generated without explicitly exposing target mood/genre tags. Leakage count is measured and saved in leakage_report.csv.",
  "description": "Fine-tunes FLAN-T5-small to translate script-like scene descriptions into detailed background music descriptions."
}}''')

    print("Saved experiment to:", EXP_DIR)

if __name__ == "__main__":
    main()
