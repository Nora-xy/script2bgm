# Script2BGM: Fine-tuned Music Prompt Generation and CLAP-based Background Music Retrieval


## What it Does

Script2BGM is a multi-stage applied machine learning system for recommending background music for short film scenes. Given a screenplay-like scene description, the system first uses a fine-tuned FLAN-T5 language model to translate the scene into a structured background-music description. That generated description is then used as a CLAP text query to retrieve matching audio tracks from the MTG-Jamendo dataset.The central idea is that screenplay language and music-description language are different. A filmmaker may describe setting, blocking, tension, dialogue density, and visual rhythm, while an audio retrieval model such as CLAP performs better when the query contains explicit musical attributes such as mood, genre, pacing, texture, and instrumentation. This project learns an intermediate translation layer between those two forms of language.


## Quick Start

### 1. Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare MTG-Jamendo metadata

```bash
python src/prepare_mtg_metadata.py
python src/make_subset.py
```

### 3. Match audio paths

After downloading and unpacking MTG-Jamendo audio-low files, run:

```bash
python src/check_audio_paths.py \
  --manifest data/processed/mtg_subset.csv \
  --audio_root ~/datasets/mtg_jamendo/audio_low \
  --output data/processed/mtg_subset_with_audio.csv
```

Expected result for the 1000-track subset:

```text
COVERAGE: 100%
```

### 4. Prepare FLAN-T5 v2 training data

```bash
python src/prepare_flan_t5_v2_structured_data.py
```

This creates:

```text
data/processed/scene_to_music_prompt_v2_structured.csv
data/processed/leakage_report_v2.csv
```

### 5. Fine-tune FLAN-T5-small

```bash
python src/finetune_flan_t5_v2.py
```

This saves the fine-tuned model and training logs.

### 6. Generate and evaluate music descriptions

```bash
python src/generate_and_eval_flan_t5_v2.py
```

This creates:

```text
experiments/flan_t5_scene_to_music_prompt_v2_structured/test_predictions.csv
experiments/flan_t5_scene_to_music_prompt_v2_structured/generation_metrics.csv
```

### 7. Extract CLAP audio embeddings

```bash
python src/extract_clap_audio_subset.py \
  --manifest data/processed/mtg_subset_with_audio.csv \
  --output_dir data/embeddings/clap_audio_subset \
  --batch_size 1
```

Expected sanity check:

```text
emb shape: torch.Size([1000, 512])
metadata rows: 1000
zero rows: 0
norm min/max: approximately 1.0
```

### 8. Run final CLAP retrieval comparison

```bash
python src/clap_query_ensemble.py \
  --pred_path experiments/flan_t5_scene_to_music_prompt_v2_structured/test_predictions.csv \
  --out_dir experiments/clap_query_ensemble_v2
```

This creates:

```text
experiments/clap_query_ensemble_v2/metrics.csv
experiments/clap_query_ensemble_v2/README.md
experiments/clap_query_ensemble_v2/examples_direct_screenplay.csv
experiments/clap_query_ensemble_v2/examples_finetuned_generated_description.csv
experiments/clap_query_ensemble_v2/examples_oracle_target_description.csv
```

## Video Links

- Demo video: [https://duke.box.com/s/byq5e2fx64dr277jryw4vkd8gvimvk3o]
- Technical walkthrough video: [https://duke.box.com/s/b9x3le4esyw29d2e4t5j62wunkqt8me3]


## Evaluation

### Main Retrieval Metrics

The final experiment compares four query strategies over a 1000-track CLAP audio retrieval pool.

| Method | Recall@5 | MoodHit@5 | GenreHit@5 |
|---|---:|---:|---:|
| Direct screenplay → CLAP | 0.0100 | 0.3433 | 0.6678 |
| FLAN-T5 v2 generated description → CLAP | 0.0167 | 0.4100 | 0.6779 |
| Oracle target music description → CLAP | 0.0133 | 0.4733 | 0.8322 |
| Best score fusion, alpha=0.75 | 0.0233 | 0.4667 | 0.7852 |

The generated music-description query improves over direct screenplay input. The best result comes from score-level fusion between direct screenplay and generated-description scores, suggesting that narrative screenplay cues and explicit music-language cues provide complementary retrieval signals.

Full retrieval results are saved in:

```text
experiments/clap_query_ensemble_v2/metrics.csv
experiments/final_retrieval_results.csv
```

### Generation Metrics

FLAN-T5 v2 generation is evaluated using ROUGE and tag-recovery metrics.

The complete generation metrics are saved in:

```text
experiments/flan_t5_scene_to_music_prompt_v2_structured/generation_metrics.csv
experiments/generation_results.csv
```

### Retrieval Conditions

The final CLAP retrieval comparison uses four conditions:

1. `direct_screenplay`  
   The raw screenplay-style scene text is used directly as the CLAP text query.

2. `finetuned_generated_description`  
   The fine-tuned FLAN-T5 v2 model generates a structured music description, which is used as the CLAP query.

3. `oracle_target_description`  
   The ground-truth structured music description derived from MTG-Jamendo tags is used as an upper-bound query.

4. `score_fusion_alpha_0.75_direct`  
   A score-level ensemble combines direct screenplay CLAP scores and generated-description CLAP scores.

Fusion formula:

```text
score = alpha * direct_screenplay_score + (1 - alpha) * generated_description_score
```

The best fusion result uses:

```text
alpha = 0.75
```