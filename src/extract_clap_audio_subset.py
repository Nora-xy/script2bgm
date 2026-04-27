from pathlib import Path
import argparse
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel

def unwrap_embedding(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected output type: {type(output)}")

def load_audio(path, sr=48000, duration=30.0):
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio

def batch_items(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/mtg_subset_with_audio.csv")
    parser.add_argument("--output_dir", default="data/embeddings/clap_audio_subset")
    parser.add_argument("--model_name", default="laion/clap-htsat-fused")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sampling_rate", type=int, default=48000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    if "audio_exists" in df.columns:
        df = df[df["audio_exists"] == True].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    print("Rows:", len(df))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = ClapModel.from_pretrained(args.model_name).to(device)
    processor = ClapProcessor.from_pretrained(args.model_name)
    model.eval()

    all_embeds = []
    failed = []

    audio_paths = df["audio_path"].astype(str).tolist()

    for start in tqdm(range(0, len(audio_paths), args.batch_size), desc="CLAP audio embedding"):
        batch_paths = audio_paths[start:start + args.batch_size]
        audio_arrays = []
        valid_indices = []

        for local_idx, path in enumerate(batch_paths):
            try:
                audio = load_audio(path, sr=args.sampling_rate, duration=args.duration)
                audio_arrays.append(audio)
                valid_indices.append(local_idx)
            except Exception as e:
                failed.append((start + local_idx, path, str(e)))
                print(f"\nFAILED load idx={start + local_idx} path={path}\n{e}\n")

        if not audio_arrays:
            continue

        try:
            inputs = processor(
                audio=audio_arrays,
                sampling_rate=args.sampling_rate,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_audio_features(**inputs)
                emb = unwrap_embedding(emb)
                emb = torch.nn.functional.normalize(emb, dim=-1)

            all_embeds.append(emb.cpu())

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nCUDA OOM. Re-run with --batch_size 1.\n")
            raise e

    if failed:
        failed_path = out_dir / "failed_audio_files.csv"
        pd.DataFrame(failed, columns=["index", "audio_path", "error"]).to_csv(failed_path, index=False)
        print("Saved failed file list:", failed_path)

    audio_embeds = torch.cat(all_embeds, dim=0)
    torch.save(audio_embeds, out_dir / "clap_audio_embeds.pt")

    df.to_csv(out_dir / "metadata.csv", index=False)

    print("Saved audio embeddings:", out_dir / "clap_audio_embeds.pt")
    print("Saved metadata:", out_dir / "metadata.csv")
    print("Shape:", tuple(audio_embeds.shape))
    print("Zero rows:", int((audio_embeds.abs().sum(dim=1) == 0).sum().item()))
    print("Norm min/max:", torch.norm(audio_embeds, dim=1).min().item(), torch.norm(audio_embeds, dim=1).max().item())

if __name__ == "__main__":
    main()
