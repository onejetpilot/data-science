from __future__ import annotations

import argparse
from pathlib import Path

import requests


DATA_URL = "https://huggingface.co/datasets/onejetpilot/real_estate/resolve/main/real_estate_data.csv"
DATA_FILE = "real_estate_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw real estate dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory for raw files.")
    return parser.parse_args()


def download_file(url: str, target_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)


def main() -> None:
    # 1) Готовим директорию сырого слоя и путь к файлу.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / DATA_FILE

    # 2) Скачиваем датасет только при отсутствии локальной копии.
    if not output_file.exists():
        print(f"[ingest] Downloading dataset: {DATA_URL}")
        download_file(DATA_URL, output_file)
        print(f"[ingest] Saved file: {output_file}")
    else:
        print(f"[ingest] File already exists: {output_file}")


if __name__ == "__main__":
    main()
