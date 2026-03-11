from __future__ import annotations

import argparse
from pathlib import Path

import requests


BASE_URLS = [
    "https://huggingface.co/datasets/onejetpilot/rides/resolve/main/",
    "https://code.s3.yandex.net/datasets/",
]
FILES = ["users_go.csv", "rides_go.csv", "subscriptions_go.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw rides datasets.")
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
    # 1) Создаем директорию сырого слоя.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2) Скачиваем все исходные CSV, если их еще нет локально.
    for file_name in FILES:
        target_path = args.output_dir / file_name
        if target_path.exists():
            print(f"[ingest] File already exists: {target_path}")
            continue

        downloaded = False
        for base_url in BASE_URLS:
            url = base_url + file_name
            try:
                print(f"[ingest] Downloading: {url}")
                download_file(url, target_path)
                print(f"[ingest] Saved file: {target_path}")
                downloaded = True
                break
            except requests.RequestException:
                continue
        if not downloaded:
            raise RuntimeError(f"Failed to download {file_name} from all configured sources")


if __name__ == "__main__":
    main()
