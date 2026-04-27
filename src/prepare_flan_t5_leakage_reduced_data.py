from pathlib import Path
import pandas as pd
import random
import re

INPUT = Path("data/processed/mtg_subset.csv")
OUTPUT = Path("data/processed/scene_to_music_prompt_leakage_reduced.csv")
REPORT = Path("data/processed/leakage_report.csv")

random.seed(42)

GENERIC_TAGS = {
    "background", "film", "movie", "soundtrack", "advertising",
    "music", "theme", "sound"
}

def split_tags(x):
    if pd.isna(x) or not str(x).strip():
        return []
    return [t.strip().lower().replace("_", " ").replace("-", " ") for t in str(x).split("|") if t.strip()]

def tag_set(row):
    return split_tags(row["moodtheme_tags"]) + split_tags(row["genre_tags"])

def contains_tag_leak(text, tags):
    text_l = str(text).lower()
    leaked = []
    for tag in tags:
        tag = tag.strip().lower()
        if not tag or tag in GENERIC_TAGS:
            continue

        # exact phrase/word match
        pattern = r"\b" + re.escape(tag) + r"\b"
        if re.search(pattern, text_l):
            leaked.append(tag)

    return leaked

def sanitize_leaks(text, tags):
    fixed = text
    replacements = {
        "sad": "quietly withdrawn",
        "happy": "newly hopeful",
        "dark": "dimly lit",
        "dramatic": "weighted with consequence",
        "romantic": "intimate",
        "calm": "still",
        "relaxing": "unhurried",
        "melancholic": "caught in memory",
        "emotional": "personally charged",
        "mysterious": "uncertain",
        "scary": "unsettling",
        "suspenseful": "uneasy",
        "epic": "large in scale",
        "powerful": "forceful",
        "energetic": "full of movement",
        "positive": "hopeful",
        "uplifting": "rising toward relief",
        "fun": "playful",
        "classical": "formal",
        "orchestral": "grand",
        "electronic": "neon-lit",
        "ambient": "atmospheric",
        "rock": "restless",
        "pop": "bright",
        "jazz": "late-night",
        "blues": "weary",
        "metal": "harsh",
        "punkrock": "rebellious",
        "newage": "dreamlike",
        "chillout": "unhurried",
        "downtempo": "slow-moving",
        "easylistening": "gentle"
    }

    for tag in tags:
        if tag in GENERIC_TAGS:
            continue
        repl = replacements.get(tag, "cinematic")
        fixed = re.sub(r"\b" + re.escape(tag) + r"\b", repl, fixed, flags=re.IGNORECASE)

    return fixed

def choose_scene_family(tags):
    t = set(tags)

    if t & {"dark", "scary", "suspenseful", "mysterious", "dramatic"}:
        return "thriller"
    if t & {"sad", "melancholic", "emotional"}:
        return "memory"
    if t & {"happy", "positive", "uplifting", "fun"}:
        return "hopeful"
    if t & {"epic", "powerful", "trailer", "action", "energetic", "adventure"}:
        return "montage"
    if t & {"calm", "relaxing", "meditative", "dreamy"}:
        return "quiet"
    if t & {"romantic", "love"}:
        return "intimate"
    return "neutral"

SCENE_BANK = {
    "thriller": [
        "INT. EMPTY SUBWAY PLATFORM - NIGHT.\nA young woman crosses the platform alone. The lights flicker above her. She stops when she hears footsteps behind her, but the platform is empty. No one speaks.",
        "INT. APARTMENT HALLWAY - NIGHT.\nA man stands outside a half-open door. The corridor is still. Somewhere inside the apartment, a floorboard creaks. He hesitates before stepping forward.",
        "EXT. PARKING LOT - NIGHT.\nA character searches for their car under weak streetlights. Their phone battery dies. A distant engine starts, then cuts off abruptly."
    ],
    "memory": [
        "INT. BEDROOM - LATE AFTERNOON.\nA character sits on the floor beside an open suitcase, looking through old photographs. They pause over one picture and do not move for a long time.",
        "EXT. TRAIN STATION - EVENING.\nTwo friends stand apart after a difficult goodbye. One wants to say more, but the train arrives before either of them can speak.",
        "INT. DORM ROOM - NIGHT.\nA student reads an old message on their phone. The room is quiet except for the hum of the light. They place the phone face down and stare at the wall."
    ],
    "hopeful": [
        "EXT. CITY STREET - MORNING.\nA character steps out of an apartment building after a long night. Sunlight hits the windows across the street. They take a breath and begin walking.",
        "EXT. SCHOOL COURTYARD - DAY.\nA small group of friends run across the courtyard after finishing their final exam. One of them turns back for a moment, smiling before joining the others.",
        "INT. SMALL KITCHEN - MORNING.\nA family prepares breakfast in a cramped apartment. The conversation is simple and ordinary, but the atmosphere begins to feel lighter."
    ],
    "montage": [
        "EXT. HILLSIDE ROAD - SUNSET.\nA character runs uphill, exhausted but refusing to stop. The sequence cuts between their feet, their breathing, and the open road ahead.",
        "INT. TRAINING ROOM - NIGHT.\nA montage shows a character repeating the same difficult task again and again. Each failure is shorter than the last. Their focus sharpens.",
        "EXT. WIDE LANDSCAPE - DAWN.\nA group reaches the top of a ridge after a long journey. They look out over the distance, aware that the hardest part is still ahead."
    ],
    "quiet": [
        "INT. LIVING ROOM - EARLY MORNING.\nA person sits beside a window with a cup of tea. Outside, the street is almost empty. Nothing urgent happens, but the moment feels important.",
        "EXT. LAKESIDE - AFTERNOON.\nA character walks slowly along the water. The conversation from the previous scene lingers in their mind, though no words are spoken now.",
        "INT. LIBRARY - DAY.\nTwo characters sit across from each other at a table. They exchange brief glances over their books, each waiting for the other to speak first."
    ],
    "intimate": [
        "INT. APARTMENT KITCHEN - NIGHT.\nTwo people clean up after dinner in near silence. Their hands almost touch at the sink. One begins to say something, then stops.",
        "EXT. BUS STOP - RAINY EVENING.\nTwo characters wait under the same shelter. The conversation is ordinary, but their pauses reveal more than their words.",
        "INT. HOTEL ROOM - DAWN.\nA character watches someone sleep before quietly packing a bag. They leave a note on the table and stand at the door for a long moment."
    ],
    "neutral": [
        "INT. SMALL OFFICE - DAY.\nA character waits for an important phone call. They rearrange papers on the desk, check the clock, and try to appear composed.",
        "EXT. CITY SIDEWALK - EVENING.\nA character moves through a crowd while replaying a conversation in their head. The city continues around them, indifferent.",
        "INT. CAR - NIGHT.\nTwo people drive without speaking. The road lights pass across their faces. The silence between them slowly changes shape."
    ],
}

def dialogue_density_for_family(family):
    if family in {"thriller", "quiet", "montage"}:
        return "low"
    if family in {"intimate", "memory"}:
        return "medium"
    return "medium"

def make_script_like_input(row, variant_idx):
    tags = tag_set(row)
    family = choose_scene_family(tags)
    scenes = SCENE_BANK[family]
    scene = scenes[(variant_idx + abs(hash(row["track_id"]))) % len(scenes)]

    dialogue_density = dialogue_density_for_family(family)

    add_on = random.choice([
        f"\nThe scene relies on visual detail and has {dialogue_density} dialogue density.",
        f"\nThe emotional shift happens mostly through gesture, silence, and blocking. Dialogue density is {dialogue_density}.",
        f"\nThe scene is meant to be read through movement, facial expression, and editing rhythm. Dialogue density is {dialogue_density}."
    ])

    screenplay = scene + add_on
    screenplay = sanitize_leaks(screenplay, tags)

    return screenplay

def infer_pacing(tags):
    t = set(tags)
    if t & {"calm", "relaxing", "meditative", "sad", "melancholic", "dreamy"}:
        return "slow"
    if t & {"energetic", "action", "powerful", "aggressive", "trailer"}:
        return "fast"
    return "medium"

def infer_tension(tags):
    t = set(tags)
    if t & {"dark", "scary", "suspenseful", "dramatic", "aggressive", "epic", "powerful", "trailer"}:
        return "high"
    if t & {"sad", "melancholic", "emotional", "mysterious", "romantic"}:
        return "medium"
    return "low"

def infer_texture(genre_tags):
    g = set(genre_tags)
    if g & {"ambient", "newage", "downtempo", "chillout"}:
        return "soft ambient textures"
    if g & {"classical", "orchestral", "soundtrack"}:
        return "orchestral cinematic textures"
    if g & {"electronic", "techno", "house", "trance"}:
        return "electronic synth textures"
    if g & {"rock", "metal", "punkrock", "grunge", "poprock"}:
        return "guitar-driven textures"
    if g & {"jazz", "blues"}:
        return "warm acoustic and improvisational textures"
    if g & {"pop", "easylistening"}:
        return "accessible melodic textures"
    if g & {"folk", "country"}:
        return "acoustic folk textures"
    return "cinematic musical textures"

def make_target(row):
    mood = split_tags(row["moodtheme_tags"])
    genre = split_tags(row["genre_tags"])
    all_tags = mood + genre

    mood_text = ", ".join(mood) if mood else "cinematic"
    genre_text = ", ".join(genre) if genre else "background"
    pacing = infer_pacing(all_tags)
    tension = infer_tension(all_tags)
    texture = infer_texture(genre)

    return (
        f"{mood_text} {genre_text} background music with {texture}. "
        f"The music should have a {pacing} pace and {tension} emotional tension. "
        f"It should function as a cinematic underscore, supporting the scene without overpowering dialogue, editing, or visual storytelling."
    )

def main():
    df = pd.read_csv(INPUT)
    rows = []

    # two script-like variations per track, while preserving original split by track
    for _, row in df.iterrows():
        for variant_idx in range(2):
            screenplay = make_script_like_input(row, variant_idx)
            tags = tag_set(row)
            leaked = contains_tag_leak(screenplay, tags)

            rows.append({
                "track_id": row["track_id"],
                "split": row["split"],
                "moodtheme_tags": row["moodtheme_tags"],
                "genre_tags": row["genre_tags"],
                "input_screenplay": screenplay,
                "input_text": "Generate a detailed background music description for this screenplay excerpt:\n" + screenplay,
                "target_text": make_target(row),
                "leak_count": len(leaked),
                "leaked_terms": "|".join(leaked),
            })

    out = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    report = out.groupby("split").agg(
        rows=("track_id", "count"),
        avg_leak_count=("leak_count", "mean"),
        rows_with_leak=("leak_count", lambda x: int((x > 0).sum()))
    ).reset_index()
    report.to_csv(REPORT, index=False)

    print("Saved:", OUTPUT)
    print("Shape:", out.shape)
    print("\nSplit counts:")
    print(out["split"].value_counts())
    print("\nLeakage report:")
    print(report.to_string(index=False))
    print("\nExamples:")
    print(out[["input_screenplay", "target_text", "leak_count", "leaked_terms"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
