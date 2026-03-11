from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


DEFAULT_DATA_URL = "https://huggingface.co/datasets/onejetpilot/find_image/resolve/main/find_image.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract find_image dataset archive.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="Source URL for find_image.zip.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Raw data directory.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract even if files exist.")
    return parser.parse_args()


def main() -> None:
    # 1) Читаем аргументы и создаем директории raw-слоя.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "find_image.zip"
    extract_dir = args.output_dir / "find_image"

    required_files = [
        extract_dir / "to_upload" / "train_dataset.csv",
        extract_dir / "to_upload" / "ExpertAnnotations.tsv",
        extract_dir / "to_upload" / "CrowdAnnotations.tsv",
    ]
    need_extract = args.force or not all(p.exists() for p in required_files)

    # 2) Скачиваем архив, если он отсутствует или включен force.
    downloaded = False
    if args.force or not archive_path.exists():
        urlretrieve(args.data_url, archive_path)
        downloaded = True

    # 3) Распаковываем архив в data/raw/find_image.
    extracted = False
    if need_extract:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        extracted = True

    # 4) Сохраняем manifest для воспроизводимости источника.
    manifest = {
        "data_url": args.data_url,
        "archive_path": str(archive_path.resolve()),
        "extract_dir": str(extract_dir.resolve()),
        "downloaded": downloaded,
        "extracted": extracted,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[ingest] Archive path: {archive_path}")
    print(f"[ingest] Extract dir: {extract_dir}")
    print(f"[ingest] Manifest path: {manifest_path}")
    print(f"[ingest] Downloaded now: {downloaded}")
    print(f"[ingest] Extracted now: {extracted}")


if __name__ == "__main__":
    main()
