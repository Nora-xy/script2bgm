from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_PATH = Path("data/processed/scene_to_music_prompt_v2_structured.csv")
MODEL_NAME = "google/flan-t5-small"
OUT_DIR = Path("models/flan_t5_scene_to_music_prompt_v2_structured")
EXP_DIR = Path("experiments/flan_t5_scene_to_music_prompt_v2_structured")

MAX_INPUT_LEN = 384
MAX_TARGET_LEN = 192

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["split"].isin(["train", "validation", "test"])].reset_index(drop=True)

    return DatasetDict({
        "train": Dataset.from_pandas(df[df["split"] == "train"].reset_index(drop=True)),
        "validation": Dataset.from_pandas(df[df["split"] == "validation"].reset_index(drop=True)),
        "test": Dataset.from_pandas(df[df["split"] == "test"].reset_index(drop=True)),
    })

def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    datasets = load_data()
    print(datasets)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = datasets.map(
        preprocess,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        learning_rate=7e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        weight_decay=0.01,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

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
    )

    trainer.train()

    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    pd.DataFrame([test_metrics]).to_csv(EXP_DIR / "test_loss_metrics.csv", index=False)
    pd.DataFrame(trainer.state.log_history).to_csv(EXP_DIR / "training_log.csv", index=False)

    with open(EXP_DIR / "config.json", "w") as f:
        f.write('''{
  "experiment_name": "flan_t5_scene_to_music_prompt_v2_structured",
  "model": "google/flan-t5-small",
  "dataset": "scene_to_music_prompt_v2_structured.csv",
  "task": "screenplay_to_structured_music_description",
  "main_change_from_v1": "Structured output format plus stronger non-music cinematic cues for genre/style inference.",
  "epochs": 5,
  "learning_rate": 7e-5,
  "batch_size": 4,
  "gradient_accumulation_steps": 2
}''')

    print("Saved:", EXP_DIR)

if __name__ == "__main__":
    main()
