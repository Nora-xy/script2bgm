# Error Analysis

## Main failure in v1
The v1 FLAN-T5 model learned the general format of music descriptions, but it did not reliably recover explicit mood and genre attributes. This hurt CLAP retrieval because CLAP responds strongly to explicit musical descriptors such as genre, instrumentation, and mood.

## Why v1 failed
The input screenplay text was leakage-reduced, so direct words such as "jazz", "rock", or "orchestral" were intentionally removed. This made the task more realistic but also harder. The model often generated generic cinematic descriptions rather than specific music attributes.

## v2 improvement
For v2, I changed the target format to a structured music plan:
Mood, Genre, Pacing, Tension, Texture, and Description.
I also added stronger non-music cinematic cues in the screenplay input, such as setting, editing rhythm, visual scale, and location type.

## Quantitative impact
V1 generated-description retrieval:
MoodHit@5 = 0.2433
GenreHit@5 = 0.6443

V2 generated-description retrieval:
MoodHit@5 = 0.4100
GenreHit@5 = 0.6779

Best v2 score fusion:
MoodHit@5 = 0.4667
GenreHit@5 = 0.7852

## Interpretation
The results suggest that raw screenplay text and generated music descriptions provide complementary retrieval signals. Direct screenplay text preserves narrative and visual information, while generated music descriptions translate part of that information into CLAP-friendly music-language attributes.
