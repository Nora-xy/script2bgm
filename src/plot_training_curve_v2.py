from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG = Path("experiments/flan_t5_scene_to_music_prompt_v2_structured/training_log.csv")
OUT = Path("experiments/flan_t5_scene_to_music_prompt_v2_structured/training_curve.png")

df = pd.read_csv(LOG)

plt.figure()
if "loss" in df.columns:
    train = df.dropna(subset=["loss"])
    plt.plot(train["step"], train["loss"], label="train loss")

if "eval_loss" in df.columns:
    val = df.dropna(subset=["eval_loss"])
    plt.plot(val["step"], val["eval_loss"], label="validation loss")

plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("FLAN-T5 v2 training curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT, dpi=200)
print("Saved:", OUT)
