from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DATA_URL = "https://huggingface.co/datasets/onejetpilot/purchases/resolve/main/filtered_data.zip"
EXPECTED_FILES = [
    "apparel-messages.csv",
    "apparel-purchases.csv",
    "apparel-target_binary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw purchase prediction data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory for raw files.")
    return parser.parse_args()


def main() -> None:
    # 1) Готовим рабочие пути.
    args = parse_args()
    raw_dir = args.output_dir
    extract_dir = raw_dir / "filtered_data"
    zip_path = raw_dir / "filtered_data.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 2) Скачиваем архив только если его еще нет.
    if not zip_path.exists():
        print(f"[ingest] Downloading archive: {DATA_URL}")
        urlretrieve(DATA_URL, zip_path)
    else:
        print(f"[ingest] Archive already exists: {zip_path}")

    # 3) Распаковываем архив в data/raw.
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"[ingest] Extracted archive into: {raw_dir}")

    # 4) Копируем ключевые CSV в корень data/raw для простых путей в следующих шагах.
    for file_name in EXPECTED_FILES:
        src = extract_dir / file_name
        dst = raw_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Expected file is missing after extract: {src}")
        shutil.copyfile(src, dst)
        print(f"[ingest] Prepared file: {dst}")


if __name__ == "__main__":
    main()
