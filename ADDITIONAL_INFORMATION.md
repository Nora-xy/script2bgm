## Project Goal

The main research question is:

- Can a learned intermediate music-description generator improve text-to-audio background music retrieval compared with directly giving screenplay text to CLAP?

Final pipeline:

```text
screenplay-like scene text
        ↓
fine-tuned FLAN-T5-small
        ↓
structured music description
        ↓
CLAP text encoder
        ↓
CLAP audio embedding retrieval
        ↓
recommended MTG-Jamendo BGM track
```

## Motivation

Choosing background music for short films is difficult because directors usually think in cinematic terms rather than music-catalog terms. A director may know that a scene is quiet, tense, visually sparse, dialogue-light, or emotionally unresolved, but may not know the best music genre, texture, or instrumentation to search for.

This project explores whether machine learning can help bridge that gap. The system translates a screenplay-like scene into a more retrieval-friendly music description, then uses CLAP to search a music library.

## Dataset

The project uses MTG-Jamendo mood/theme and genre metadata, together with MTG-Jamendo audio-low MP3 files.

Because large-scale paired datasets of film scripts and background music descriptions are difficult to obtain, I construct a weakly supervised dataset:

- Original human annotations: MTG-Jamendo mood/theme and genre tags
- Input side: screenplay-style scene descriptions
- Target side: structured background music descriptions
- Audio side: MTG-Jamendo audio-low tracks

The tags are used as weak supervision, but the model input is designed to look like a screenplay or scene cue rather than a list of music tags.

## Leakage Reduction

A major concern is that if both input and output are generated from the same tags, the model could simply learn to copy tag words. To reduce this problem, the input screenplay-style text avoids directly exposing explicit mood, genre, or instrumentation labels.

For example, the input does not simply say:

```text
The scene needs dark orchestral music.
```

Instead, it uses cinematic cues:

```text
INT. EMPTY SUBWAY PLATFORM - NIGHT.
A young woman crosses the platform alone. The lights flicker above her. She stops when she hears footsteps behind her, but the platform is empty.
The scene is carried by blocking, editing rhythm, and facial expression. Dialogue density is low.
```

The target may then be:

```text
Mood: dark, suspenseful
Genre: orchestral, soundtrack
Pacing: medium
Tension: high
Texture: orchestral cinematic textures
Description: dark suspenseful orchestral soundtrack background music with high emotional tension.
```

The v2 leakage report shows that direct tag leakage in the generated screenplay-style inputs is close to zero:

```text
train: avg_leak_count = 0.0021
validation: avg_leak_count = 0.0067
test: avg_leak_count = 0.0100
```

## Data Split

The project uses a 1000-track subset for fast iteration:

```text
Train: 700 tracks
Validation: 150 tracks
Test: 150 tracks
```

For FLAN-T5 training, I generate two screenplay-style variants per track:

```text
Train: 1400 examples
Validation: 300 examples
Test: 300 examples
```

The final CLAP retrieval experiment uses:

```text
300 test queries
1000 candidate audio tracks
```

## Models Used

### MiniLM Text Retrieval Baseline

I first used `sentence-transformers/all-MiniLM-L6-v2` as a text-to-text retrieval baseline. This helped test whether scene descriptions and music descriptions could be matched in a general sentence embedding space.

### CLAP

I use `laion/clap-htsat-fused`, a pretrained Contrastive Language-Audio Pretraining model. CLAP has a text encoder and an audio encoder trained to align language and audio in a shared embedding space.

In this project, CLAP is used for:

- text embedding
- audio embedding
- text-to-audio retrieval
- comparison between raw screenplay queries, generated music-description queries, oracle music-description queries, and score-level fusion

### Fine-tuned FLAN-T5-small

The main trainable model is `google/flan-t5-small`, fine-tuned as a sequence-to-sequence model.

Task:

```text
input: screenplay-like scene description
output: structured background music description
```

The v2 output format is:

```text
Mood:
Genre:
Pacing:
Tension:
Texture:
Description:
```

## Main Findings

### 1. CLAP benefits from explicit music-language descriptions

The oracle target-description condition performs very strongly on mood and genre matching. This supports the design choice of translating screenplay language into music-description language before retrieval.

### 2. Fine-tuned FLAN-T5 v2 improves retrieval over direct screenplay input

Compared with direct screenplay queries, FLAN-T5 v2 generated descriptions improve:

```text
Recall@5: 0.0100 → 0.0167
MoodHit@5: 0.3433 → 0.4100
GenreHit@5: 0.6678 → 0.6779
```

### 3. Fusion performs best

The best hybrid method improves substantially over direct screenplay input:

```text
MoodHit@5: 0.3433 → 0.4667
GenreHit@5: 0.6678 → 0.7852
```

This suggests that screenplay text and generated music descriptions contain complementary information.

## Error Analysis

The v1 FLAN-T5 model learned the general format of music descriptions but did not reliably recover explicit mood and genre attributes. This hurt CLAP retrieval because CLAP responds strongly to explicit musical descriptors such as genre, instrumentation, and mood.

The v2 model improves on v1 by using:

- structured target format
- stronger non-music cinematic cues
- leakage-reduced screenplay-style inputs
- explicit fields for mood, genre, pacing, tension, and texture

The final results show that v2 improves retrieval, but the generated descriptions still do not fully match the oracle condition. This means the intermediate generation model is helpful, but not yet perfect.

More detailed error analysis is saved in:

```text
experiments/error_analysis.md
```

## Design Decisions

### Why not directly use screenplay text with CLAP?

Direct screenplay text contains useful visual and narrative information, but CLAP is trained on audio-language alignment and responds better to music-language descriptions. The oracle target-description result confirms that CLAP performs better when the query contains explicit music descriptors.

### Why use weak supervision?

There is no convenient large-scale dataset of paired film scripts and background music descriptions. MTG-Jamendo provides mood/theme and genre annotations, so I use those annotations to construct weakly supervised training pairs.

### Why use FLAN-T5?

FLAN-T5-small is lightweight enough to fine-tune on the cluster GPU while still being a transformer language model capable of instruction-style sequence-to-sequence generation.

### Why use score fusion?

Direct screenplay queries preserve visual and narrative context. Generated music descriptions provide explicit musical attributes. The score fusion experiment shows that combining both signals improves retrieval.

