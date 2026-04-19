from pathlib import Path
from datasets import Dataset, DatasetDict


DATA_DIR = Path("data/deu-eng-fixed")


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

    examples = []
    for en, de in zip(eng_lines, deu_lines):
        examples.append(
            {
                "translation": {
                    "en": en,
                    "de": de,
                }
            }
        )

    return examples


def build_dataset_dict() -> DatasetDict:
    train_examples = read_parallel_split("train")
    dev_examples = read_parallel_split("dev")
    test_examples = read_parallel_split("test")

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_examples),
            "validation": Dataset.from_list(dev_examples),
            "test": Dataset.from_list(test_examples),
        }
    )

    return dataset_dict


def main() -> None:
    dataset_dict = build_dataset_dict()

    print(dataset_dict)
    print()
    print("Split sizes:")
    for split in dataset_dict:
        print(f"{split}: {len(dataset_dict[split])}")

    print()
    print("Sample training example:")
    print(dataset_dict["train"][200])


if __name__ == "__main__":
    main()