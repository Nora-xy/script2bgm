import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import librosa
from transformers import ClapProcessor, ClapModel

INPUT = "data/processed/debug_audio_subset.csv"
OUTPUT = "data/embeddings/clap_audio_debug.pt"

def unwrap_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected output type: {type(output)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model.eval()

df = pd.read_csv(INPUT)

audio_embeds = []
failed = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    path = row["audio_path"]

    try:
        audio, sr = librosa.load(path, sr=48000, mono=True, duration=30.0)

        inputs = processor(
            audio=audio,
            sampling_rate=48000,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_audio_features(**inputs)
            emb = unwrap_embedding(emb)
            emb = torch.nn.functional.normalize(emb, dim=-1)

        audio_embeds.append(emb.cpu())

    except Exception as e:
        failed.append((idx, path, str(e)))
        print(f"\nFAILED idx={idx} path={path}\n{e}\n")

if failed:
    print("\nFailed files:")
    for item in failed:
        print(item)
    raise RuntimeError(f"{len(failed)} audio files failed. Not saving fake embeddings.")

audio_embeds = torch.cat(audio_embeds, dim=0)
torch.save(audio_embeds, OUTPUT)

print("Saved:", OUTPUT)
print("Shape:", tuple(audio_embeds.shape))
print("Norms:", torch.norm(audio_embeds, dim=1))