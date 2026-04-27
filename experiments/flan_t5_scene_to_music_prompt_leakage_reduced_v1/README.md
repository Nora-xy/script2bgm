# FLAN-T5 Scene-to-Music Prompt Fine-tuning v1

## Goal
Fine-tune `google/flan-t5-small` to convert screenplay-like scene descriptions into detailed background music descriptions.

## Dataset
The training data is weakly supervised and constructed from MTG-Jamendo metadata. Mood/theme and genre tags are used to generate target music descriptions. The input side is converted into screenplay-style scene descriptions with leakage reduction, so explicit mood/genre/music words are not directly exposed in the input.

## Data Split
- Train: 1400 examples
- Validation: 300 examples
- Test: 300 examples

There are two screenplay-style input variants per original MTG-Jamendo track.

## Model
- Base model: `google/flan-t5-small`
- Task: sequence-to-sequence generation
- Input: screenplay-like scene text
- Output: detailed background music description

## Main Results
Recovered test metrics:
- ROUGE-1: 0.7455
- ROUGE-2: 0.7023
- ROUGE-L: 0.7443
- ROUGE-Lsum: 0.7443

Tag recovery:
- Average mood recovery: 0.1479
- Average genre recovery: 0.0921
- Mood any-hit rate: 0.2367
- Genre any-hit rate: 0.2181

## Notes
Training completed successfully. The original Trainer evaluation crashed during final test decoding, so evaluation was recovered by loading the saved fine-tuned model and manually generating predictions on the test set.

## Evidence Files
- `config.json`
- `training_log_recovered.csv`
- `test_metrics_recovered.csv`
- `tag_recovery_metrics.csv`
- `test_predictions.csv`
