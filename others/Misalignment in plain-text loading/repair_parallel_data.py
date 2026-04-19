from pathlib import Path


def rebuild_parallel_dataset(
    src_dir: str = "data/deu-eng",
    out_dir: str = "data/deu-eng-fixed",
    train_split: str = "train",
    dev_split: str = "dev",
    test_split: str = "test",
) -> None:
    """
    Repair JoeyNMT plain-text parallel data when hidden Unicode line separators
    (e.g., U+2028, U+2029) create source-target line-count mismatches.

    What it does
    - Reads the original train/dev/test files
    - Replaces U+2028 and U+2029 with spaces
    - Uses splitlines() to normalize line splitting the same way JoeyNMT does
    - Truncates train to the minimum aligned source/target length
    - Copies repaired dev/test files
    - Writes cleaned files to out_dir
    """

    src_root = Path(src_dir)
    dst_root = Path(out_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    def _normalize_text(p: Path) -> str:
        text = p.read_text(encoding="utf-8")
        return text.replace("\u2028", " ").replace("\u2029", " ")

    def _write_lines(lines, out_path: Path) -> None:
        out_path.write_text("\n".join(line.strip() for line in lines) + "\n", encoding="utf-8")

    # Repair train pairwise
    train_eng = _normalize_text(src_root / f"{train_split}.eng").splitlines()
    train_deu = _normalize_text(src_root / f"{train_split}.deu").splitlines()

    print("Original normalized train counts:")
    print("train.eng:", len(train_eng))
    print("train.deu:", len(train_deu))

    n = min(len(train_eng), len(train_deu))
    train_eng = train_eng[:n]
    train_deu = train_deu[:n]

    _write_lines(train_eng, dst_root / f"{train_split}.eng")
    _write_lines(train_deu, dst_root / f"{train_split}.deu")

    # Repair/copy dev and test the same way
    for split in [dev_split, test_split]:
        for lang in ["eng", "deu"]:
            lines = _normalize_text(src_root / f"{split}.{lang}").splitlines()
            _write_lines(lines, dst_root / f"{split}.{lang}")

    # Verification using the same splitlines() logic JoeyNMT uses
    print("\nVerification:")
    for split in [train_split, dev_split, test_split]:
        counts = {}
        for lang in ["eng", "deu"]:
            p = dst_root / f"{split}.{lang}"
            counts[lang] = len(p.read_text(encoding="utf-8").splitlines())
            print(split, lang, counts[lang])
        assert counts["eng"] == counts["deu"], f"Mismatch remains in {split}: {counts}"

    print("\nRepair complete.")


if __name__ == "__main__":
    rebuild_parallel_dataset()
