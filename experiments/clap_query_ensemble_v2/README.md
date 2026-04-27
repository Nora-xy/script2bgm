# CLAP Query Ensemble v2

## Goal
This experiment evaluates whether fine-tuned FLAN-T5 v2 generated music descriptions improve CLAP text-to-audio retrieval over direct screenplay queries.

## Methods Compared
1. `direct_screenplay`: raw screenplay-style input is used directly as the CLAP text query.
2. `finetuned_generated_description`: FLAN-T5 v2 generates a structured music description, which is used as the CLAP query.
3. `oracle_target_description`: ground-truth music description derived from MTG-Jamendo tags, used as an upper-bound condition.
4. `score_fusion`: score-level ensemble between direct screenplay and generated music description.

Fusion formula:
score = alpha * direct_screenplay_score + (1 - alpha) * generated_description_score

## Key Results

Direct screenplay:
- ExactRecall@5: 0.0100
- MoodHit@5: 0.3433
- GenreHit@5: 0.6678

Fine-tuned FLAN-T5 v2 generated description:
- ExactRecall@5: 0.0167
- MoodHit@5: 0.4100
- GenreHit@5: 0.6779

Oracle target description:
- ExactRecall@5: 0.0133
- MoodHit@5: 0.4733
- GenreHit@5: 0.8322

Best fusion result:
- alpha=0.75
- ExactRecall@5: 0.0233
- ExactRecall@10: 0.0333
- MoodHit@5: 0.4667
- GenreHit@5: 0.7852

## Interpretation
The v2 fine-tuned FLAN-T5 model improves CLAP retrieval over direct screenplay input. The best result comes from combining the original screenplay signal with the generated music-description signal. This suggests that narrative scene cues and explicit music-language descriptions provide complementary information for text-to-audio retrieval.

The oracle condition remains strongest for genre-level retrieval, showing that CLAP benefits from explicit music descriptors. However, the fusion method approaches oracle-level mood matching and substantially improves genre matching compared with direct screenplay input.
