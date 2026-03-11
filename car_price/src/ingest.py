from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


DEFAULT_DATA_URL = "https://huggingface.co/datasets/onejetpilot/autos/resolve/main/autos.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download autos.csv into raw data layer.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="Source URL for autos.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Raw data directory.")
    parser.add_argument("--timeout-sec", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    return parser.parse_args()


def main() -> None:
    # 1) Читаем аргументы и создаем папку raw-слоя.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "autos.csv"

    # 2) Скачиваем CSV или используем существующий файл.
    downloaded = False
    if not csv_path.exists() or args.force:
        response = requests.get(args.data_url, timeout=args.timeout_sec)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
        downloaded = True

    # 3) Пишем manifest для воспроизводимости источника данных.
    manifest = {
        "data_url": args.data_url,
        "csv_path": str(csv_path.resolve()),
        "csv_size_bytes": csv_path.stat().st_size,
        "downloaded": downloaded,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4) Печатаем артефакты шага ingest.
    print(f"[ingest] CSV path: {csv_path}")
    print(f"[ingest] Manifest path: {manifest_path}")
    print(f"[ingest] Downloaded now: {downloaded}")


if __name__ == "__main__":
    main()
