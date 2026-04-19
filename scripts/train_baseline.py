from pathlib import Path
import numpy as np

from datasets import Dataset, DatasetDict
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

DATA_DIR = Path("data/deu-eng-fixed")
CHECKPOINT = "Helsinki-NLP/opus-mt-en-de"
OUTPUT_DIR = "outputs/hf_ende_baseline_v1"

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128


def read_parallel_split(prefix: str) -> list[dict]:
    eng_path = DATA_DIR / f"{prefix}.eng"
    deu_path = DATA_DIR / f"{prefix}.deu"

    with eng_path.open("r", encoding="utf-8") as f_en:
        eng_lines = [line.rstrip("\n") for line in f_en]

    with deu_path.open("r", encoding="utf-8") as f_de:
        deu_lines = [line.rstrip("\n") for line in f_de]

    if len(eng_lines) != len(deu_lines):
        raise ValueError(
            f"Line count mismatch for {prefix}: "
            f"{len(eng_lines)} English lines vs {len(deu_lines)} German lines"
        )

    return [
        {"translation": {"en": en, "de": de}}
        for en, de in zip(eng_lines, deu_lines)
    ]


def build_dataset_dict() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(read_parallel_split("train")),
            "validation": Dataset.from_list(read_parallel_split("dev")),
            "test": Dataset.from_list(read_parallel_split("test")),
        }
    )


def preprocess_examples(examples, tokenizer):
    sources = [item["en"] for item in examples["translation"]]
    targets = [item["de"] for item in examples["translation"]]

    model_inputs = tokenizer(
        sources,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    print(f"Loading metric...")
    metric = evaluate.load("sacrebleu")

    print(f"Loading tokenizer/model from {CHECKPOINT} ...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

    print("Loading local dataset...")
    dataset_dict = build_dataset_dict()
    print(dataset_dict)

    # Proof-of-concept subset
    dataset_dict["train"] = dataset_dict["train"].select(range(20000))
    dataset_dict["validation"] = dataset_dict["validation"].select(range(1000))
    dataset_dict["test"] = dataset_dict["test"].select(range(min(785, len(dataset_dict["test"]))))

    print("\nSubsetted dataset:")
    print(dataset_dict)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset_dict.map(
        lambda batch: preprocess_examples(batch, tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.0,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()