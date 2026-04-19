from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DATA_DIR = Path("data/deu-eng")
CHECKPOINT = "Helsinki-NLP/opus-mt-en-de"


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
        text_target=targets,
        max_length=128,
        truncation=True,
    )
    return model_inputs


def main() -> None:
    print(f"Loading tokenizer: {CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    print(f"Loading model: {CHECKPOINT}")
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

    dataset_dict = build_dataset_dict()
    print(dataset_dict)

    small_batch = dataset_dict["train"].select(range(3))
    print("\nSample raw examples:")
    for i in range(3):
        print(small_batch[i])

    tokenized = small_batch.map(
        lambda batch: preprocess_examples(batch, tokenizer),
        batched=True,
    )

    print("\nTokenized columns:")
    print(tokenized.column_names)

    print("\nFirst tokenized example:")
    print(tokenized[0])

    print("\nModel config summary:")
    print("Model class:", model.__class__.__name__)
    print("Vocab size:", model.config.vocab_size)


if __name__ == "__main__":
    main()