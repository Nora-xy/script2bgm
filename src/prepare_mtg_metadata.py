from pathlib import Path
import pandas as pd
import random

RAW_DIR = Path("data/raw/mtg_metadata")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

def clean_tag(tag):
    tag = str(tag).strip()
    if "---" in tag:
        tag = tag.split("---", 1)[1]
    return tag.replace("_", " ").replace("-", " ").strip()

def read_mtg_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue

            track_id = parts[0]
            artist_id = parts[1]
            album_id = parts[2]
            rel_path = parts[3]

            try:
                duration = float(parts[4])
            except ValueError:
                duration = None

            tags = [clean_tag(t) for t in parts[5:] if clean_tag(t)]

            rows.append({
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "path": rel_path,
                "duration": duration,
                "tags_list": tags,
            })

    return pd.DataFrame(rows)

def load_split_file(split_name):
    path = RAW_DIR / f"autotagging_moodtheme-{split_name}.tsv"
    df = read_mtg_file(path)
    df["split"] = split_name
    return df[["track_id", "split"]]

def split_tags(tag_string):
    if pd.isna(tag_string) or not str(tag_string).strip():
        return []
    return [t.strip() for t in str(tag_string).split("|") if t.strip()]

SETTING_TEMPLATES = [
    "a character walking alone through an empty street at night",
    "two friends sitting in a quiet room after an argument",
    "a person looking at old photos and remembering the past",
    "a tense scene before an important decision",
    "a slow montage of a character traveling through unfamiliar places",
    "a quiet conversation where something remains unsaid",
    "a suspenseful scene where the character feels watched",
    "a bittersweet ending scene after a long emotional journey",
    "a lonely interior scene with very little dialogue",
    "a night-time scene where the mood slowly becomes uncertain",
]

def choose_setting(mood_tags):
    lower = set(t.lower() for t in mood_tags)

    if lower & {"dark", "scary", "suspenseful", "mysterious"}:
        return random.choice([
            "a character walking alone through an empty street at night",
            "a suspenseful scene where the character feels watched",
            "a night-time scene where the mood slowly becomes uncertain",
        ])
    if lower & {"sad", "melancholic", "emotional"}:
        return random.choice([
            "a person looking at old photos and remembering the past",
            "two friends sitting in a quiet room after an argument",
            "a lonely interior scene with very little dialogue",
        ])
    if lower & {"happy", "fun", "uplifting"}:
        return random.choice([
            "a warm coming-of-age scene with friends walking through the city",
            "a bright outdoor scene where the character starts to feel hopeful",
            "a light montage showing small moments of everyday happiness",
        ])
    if lower & {"energetic", "powerful", "aggressive", "action"}:
        return random.choice([
            "a fast-paced chase scene through a crowded city",
            "a training montage where the character pushes past their limit",
            "a tense action scene with quick cuts and rising pressure",
        ])
    return random.choice(SETTING_TEMPLATES)

def infer_pacing(tags):
    lower = set(t.lower() for t in tags)
    if lower & {"calm", "relaxing", "meditative", "sad", "melancholic", "dreamy"}:
        return "slow"
    if lower & {"energetic", "action", "powerful", "aggressive"}:
        return "fast"
    return "medium"

def infer_tension(tags):
    lower = set(t.lower() for t in tags)
    if lower & {"dark", "scary", "suspenseful", "aggressive", "epic", "powerful"}:
        return "high"
    if lower & {"sad", "melancholic", "emotional", "mysterious", "dramatic"}:
        return "medium"
    return "low"

def make_scene(row):
    mood_tags = split_tags(row["moodtheme_tags"])
    genre_tags = split_tags(row["genre_tags"])

    mood_text = ", ".join(mood_tags[:5]) if mood_tags else "emotionally expressive"
    genre_text = ", ".join(genre_tags[:3]) if genre_tags else "cinematic"

    setting = choose_setting(mood_tags)
    pacing = infer_pacing(mood_tags + genre_tags)
    tension = infer_tension(mood_tags + genre_tags)

    return (
        f"A short film scene showing {setting}. "
        f"The scene needs background music that feels {mood_text}. "
        f"The musical atmosphere should lean toward {genre_text}. "
        f"The music should work as emotional underscore without overpowering dialogue. "
        f"The pacing should feel {pacing}, and the emotional tension is {tension}."
    )

def main():
    mood_all = read_mtg_file(RAW_DIR / "autotagging_moodtheme.tsv")
    mood_all["moodtheme_tags"] = mood_all["tags_list"].apply(lambda xs: "|".join(xs))
    mood_all = mood_all.drop(columns=["tags_list"])

    genre = read_mtg_file(RAW_DIR / "autotagging_genre.tsv")
    genre["genre_tags"] = genre["tags_list"].apply(lambda xs: "|".join(xs))
    genre = genre[["track_id", "genre_tags"]].drop_duplicates("track_id")

    split_df = pd.concat([
        load_split_file("train"),
        load_split_file("validation"),
        load_split_file("test"),
    ], ignore_index=True).drop_duplicates("track_id")

    df = mood_all.merge(genre, on="track_id", how="left")
    df = df.merge(split_df, on="track_id", how="left")
    df["genre_tags"] = df["genre_tags"].fillna("")
    df["split"] = df["split"].fillna("unknown")

    df["scene_text"] = df.apply(make_scene, axis=1)
    df["music_description"] = (
        "mood/theme: " + df["moodtheme_tags"].fillna("").str.replace("|", ", ", regex=False)
        + "; genre: " + df["genre_tags"].fillna("").str.replace("|", ", ", regex=False)
    )

    out_path = OUT_DIR / "mtg_moodtheme_manifest.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", df.shape)
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nExample rows:")
    print(df[["track_id", "path", "split", "moodtheme_tags", "genre_tags", "scene_text"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
