# Attribution

## Datasets

This project uses MTG-Jamendo metadata and audio-low files. Mood/theme and genre tags are used as weak supervision to construct screenplay-to-music-description training pairs. The audio files are used for CLAP audio embedding extraction and retrieval evaluation.

I also investigated LP-MusicCaps-MC as a possible source of richer music captions, but I did not use it as the main paired training dataset because I could not reliably align its captions with my selected MTG-Jamendo track IDs within the project timeline.


## Pretrained Models

### google/flan-t5-small

I fine-tuned `google/flan-t5-small` on my custom instruction-style dataset. The task is:

```text
input: screenplay-like scene description
output: structured background music description The v2 output format includes mood, genre, pacing, tension, texture, and a full description.
```

### laion/clap-htsat-fused

I used laion/clap-htsat-fused as a pretrained audio-language retrieval model. CLAP is used for text embedding, audio embedding, and text-to-audio retrieval. I did not fine-tune CLAP.

### sentence-transformers/all-MiniLM-L6-v2

I used this model for an initial text-to-text retrieval baseline.


## Libraries

Major libraries used include PyTorch, Hugging Face Transformers, Hugging Face Datasets, Hugging Face Evaluate, SentenceTransformers, scikit-learn, pandas, numpy, librosa, matplotlib, and tqdm.


## AI Development Tool Use

I used ChatGPT as a coding and debugging assistant. It helped draft and debug preprocessing, fine-tuning, CLAP embedding, retrieval, evaluation, and documentation scripts.

I reviewed, modified, executed, and evaluated the code myself. I made the main project decisions, including:

- changing the design from direct screenplay-to-CLAP retrieval to screenplay-to-music-description-to-CLAP retrieval
- using MTG-Jamendo tags as weak supervision
- avoiding unreliable LP-MusicCaps-MC alignment
- adding leakage-reduced screenplay-style inputs
- creating the v2 structured FLAN-T5 target format
- comparing direct screenplay, generated description, oracle description, and score fusion retrieval
- interpreting v1 failure and using it to motivate v2

Important debugging work included:

- fixing PyTorch CUDA compatibility on the cluster GPU
- matching MTG-Jamendo metadata paths to .low.mp3 audio files
- fixing CLAP audio embedding extraction after early zero-vector outputs
- recovering FLAN-T5 evaluation by manually generating predictions from the saved model
- redesigning v2 after v1 generated descriptions failed to improve retrieval enough


## Summary

My main contribution is the full applied ML pipeline:

weakly supervised dataset construction
→ fine-tuned transformer generation
→ CLAP audio embedding extraction
→ text-to-audio retrieval
→ score-level ensemble
→ quantitative evaluation and error analysis

