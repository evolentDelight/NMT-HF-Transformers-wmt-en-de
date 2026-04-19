## Our Baseline Datasets

Training datasets:
- Statmt-europarl-10-deu-eng
- Statmt-news_commentary-18.1-deu-eng

Validation dataset:
- Statmt-newstest_deen-2019-deu-eng

Test dataset:
- Statmt-newstest_deen-2020-deu-eng

### Step 1 - Local dataset loader
Built a Hugging Face-compatible local dataset loader using the repaired EN-DE files in `data/deu-eng-fixed/`.

Output format:
- each example is stored as:
  - `{"translation": {"en": ..., "de": ...}}`

Purpose:
- verify train/validation/test files load correctly
- confirm alignment after the earlier Unicode line separator repair
- prepare the data for tokenization and seq2seq training

### Step 2 - Baseline tokenizer/model sanity check
Checkpoint: `Helsinki-NLP/opus-mt-en-de`

Purpose:
- confirm that a translation-oriented seq2seq model and tokenizer load correctly
- verify that the repaired local EN-DE dataset can be tokenized in Hugging Face format
- inspect the resulting fields (`input_ids`, `attention_mask`, `labels`) before building the trainer

### Step 3 - Baseline training script
Built the first Hugging Face fine-tuning script using:
- local repaired EN-DE data
- `Helsinki-NLP/opus-mt-en-de`
- `Seq2SeqTrainer`
- `DataCollatorForSeq2Seq`
- SacreBLEU evaluation

Purpose:
- verify end-to-end training and evaluation
- produce a baseline before hyperparameter tuning

### Hugging Face proof-of-concept baseline
- Framework: Hugging Face Transformers
- Model: Helsinki-NLP/opus-mt-en-de
- Dataset: repaired local WMT-based EN-DE files
- Train subset: 20,000
- Validation subset: 1,000
- Test subset: 785
- Epochs: 1
- Purpose: verify end-to-end translation fine-tuning pipeline

### Proof-of-concept baseline result
- Framework: Hugging Face Transformers
- Model: Helsinki-NLP/opus-mt-en-de
- Dataset: repaired local EN-DE WMT-based files
- Train subset: 20,000
- Validation subset: 1,000
- Test subset: 785
- Epochs: 1
- Validation loss: 1.4933
- Validation BLEU: 29.07
- Test loss: 1.4003
- Test BLEU: 29.29

```
Validation metrics: {'eval_loss': 1.493253469467163, 'eval_bleu': 29.073198819729775, 'eval_runtime': 40.4334, 'eval_samples_per_second': 24.732, 'eval_steps_per_second': 3.092, 'epoch': 1.0}
Test metrics: {'eval_loss': 1.400292158126831, 'eval_bleu': 29.288774711079068, 'eval_runtime': 67.5751, 'eval_samples_per_second': 11.617, 'eval_steps_per_second': 1.465, 'epoch': 1.0}
```

Interpretation:
- The end-to-end translation pipeline works successfully.
- The model produces meaningful English-to-German translations.
- This serves as the baseline for future tuning experiments.