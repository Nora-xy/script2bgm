from pathlib import Path
import argparse
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


def unwrap_embedding(output):
    """Handle different Transformers versions where CLAP may return a tensor or a ModelOutput."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected model output type: {type(output)}")



def batch_items(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_audio(path, sr=48000, duration=30.0):
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio


def extract_text_embeddings(model, processor, texts, device, batch_size):
    all_embeds = []

    for batch in tqdm(list(batch_items(texts, batch_size)), desc="CLAP text"):
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


def extract_audio_embeddings(model, processor, audio_paths, device, batch_size, sr, duration):
    all_embeds = []
    failed = []

    for batch_paths in tqdm(list(batch_items(audio_paths, batch_size)), desc="CLAP audio"):
        audio_arrays = []
        valid_positions = []

        for pos, path in enumerate(batch_paths):
            try:
                audio_arrays.append(load_audio(path, sr=sr, duration=duration))
                valid_positions.append(pos)
            except Exception as e:
                print(f"Failed to load {path}: {e}")

        if not audio_arrays:
            all_embeds.extend([None for _ in batch_paths])
            continue

        inputs = processor(
            audios=audio_arrays,
            sampling_rate=sr,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = model.get_audio_features(**inputs)
            emb = unwrap_embedding(emb)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            emb = emb.cpu()

        dim = emb.shape[1]
        batch_out = [torch.full((dim,), float("nan")) for _ in batch_paths]

        for local_pos, e in zip(valid_positions, emb):
            batch_out[local_pos] = e

        all_embeds.extend(batch_out)

    return torch.stack(all_embeds, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/mtg_subset_with_audio.csv")
    parser.add_argument("--output_dir", default="data/embeddings/clap_subset")
    parser.add_argument("--model_name", default="laion/clap-htsat-fused")
    parser.add_argument("--text_col", default="scene_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--audio_batch_size", type=int, default=2)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sampling_rate", type=int, default=48000)
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. "
            f"If audio is not ready yet, run with --manifest data/processed/mtg_subset.csv --text_only"
        )

    df = pd.read_csv(manifest)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if not args.text_only:
        if "audio_path" not in df.columns or "audio_exists" not in df.columns:
            raise ValueError(
                "Manifest must contain audio_path and audio_exists. "
                "Run src/check_audio_paths.py first, or use --text_only."
            )

        df = df[df["audio_exists"] == True].reset_index(drop=True)

    print("Rows:", len(df))
    print("Model:", args.model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ClapModel.from_pretrained(args.model_name).to(device)
    processor = ClapProcessor.from_pretrained(args.model_name)
    model.eval()

    texts = df[args.text_col].fillna("").astype(str).tolist()

    text_emb = extract_text_embeddings(
        model=model,
        processor=processor,
        texts=texts,
        device=device,
        batch_size=args.text_batch_size
    )

    torch.save(text_emb, out_dir / "clap_text_embeds.pt")
    print("Saved text embeddings:", out_dir / "clap_text_embeds.pt", tuple(text_emb.shape))

    if not args.text_only:
        audio_paths = df["audio_path"].astype(str).tolist()

        audio_emb = extract_audio_embeddings(
            model=model,
            processor=processor,
            audio_paths=audio_paths,
            device=device,
            batch_size=args.audio_batch_size,
            sr=args.sampling_rate,
            duration=args.duration
        )

        torch.save(audio_emb, out_dir / "clap_audio_embeds.pt")
        print("Saved audio embeddings:", out_dir / "clap_audio_embeds.pt", tuple(audio_emb.shape))

    keep_cols = [c for c in [
        "track_id",
        "path",
        "split",
        "moodtheme_tags",
        "genre_tags",
        "scene_text",
        "music_description",
        "audio_path",
        "audio_exists"
    ] if c in df.columns]

    df[keep_cols].to_csv(out_dir / "metadata.csv", index=False)
    print("Saved metadata:", out_dir / "metadata.csv")


if __name__ == "__main__":
    main()
