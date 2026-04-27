# CLAP Query Ensemble v1

This experiment compares CLAP audio retrieval using direct screenplay input, fine-tuned FLAN-T5 generated music descriptions, oracle target music descriptions, and score-level fusion between direct screenplay and generated descriptions.

## Key Results

- Direct screenplay:
  - ExactRecall@5: 0.0067
  - MoodHit@5: 0.3333
  - GenreHit@5: 0.6745

- Fine-tuned generated description:
  - ExactRecall@5: 0.0067
  - MoodHit@5: 0.2433
  - GenreHit@5: 0.6443

- Oracle target description:
  - ExactRecall@5: 0.0200
  - MoodHit@5: 0.4267
  - GenreHit@5: 0.8456

- Best fusion observations:
  - alpha=0.50 gives the best ExactRecall@5 among fusion variants: 0.0133
  - alpha=0.25 gives the best GenreHit@5 among fusion variants: 0.6879
  - alpha=0.75 gives the best MoodHit@5 among fusion variants: 0.3500

## Interpretation

The oracle music-description query strongly outperforms both direct screenplay input and generated descriptions, especially on genre-level matching. This supports the design decision that CLAP should receive music-language descriptions rather than raw screenplay text.

The v1 fine-tuned FLAN-T5 generator learns a description format, but it does not consistently recover explicit musical attributes needed by CLAP. Score fusion provides small improvements, suggesting that narrative cues and generated music descriptors contain partially complementary information.
