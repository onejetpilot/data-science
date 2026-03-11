from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


BASE_URL = "https://huggingface.co/datasets/onejetpilot/offers_pers/resolve/main/"
FILES = ["market_file.csv", "market_money.csv", "market_time.csv", "money.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw CSV files for offers_pers.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Raw data directory.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they exist.")
    return parser.parse_args()


def main() -> None:
    # 1) Подготавливаем директорию для raw-слоя.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[str] = []

    # 2) Скачиваем все исходные таблицы из ноутбука.
    for filename in FILES:
        dst = args.output_dir / filename
        if dst.exists() and not args.force:
            continue
        response = requests.get(BASE_URL + filename, timeout=args.timeout_sec)
        response.raise_for_status()
        dst.write_bytes(response.content)
        downloaded_files.append(filename)

    # 3) Сохраняем manifest со списком файлов и источником.
    manifest = {
        "base_url": BASE_URL,
        "files": FILES,
        "downloaded_now": downloaded_files,
        "output_dir": str(args.output_dir.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ingest] Raw dir: {args.output_dir}")
    print(f"[ingest] Manifest: {manifest_path}")
    print(f"[ingest] Downloaded now: {downloaded_files}")


if __name__ == "__main__":
    main()
