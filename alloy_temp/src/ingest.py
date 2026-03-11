from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


DEFAULT_DATA_URL = "https://huggingface.co/datasets/onejetpilot/alloy_temp/resolve/main/alloy_temp.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw SQLite database for alloy_temp.")
    parser.add_argument(
        "--data-url",
        type=str,
        default=DEFAULT_DATA_URL,
        help="Source URL for alloy_temp.db.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw DB and manifest will be stored.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download DB even if file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    # 1) Читаем аргументы и создаем папку для raw-слоя.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "alloy_temp.db"

    # 2) Скачиваем базу или используем уже существующую копию.
    downloaded = False
    if not db_path.exists() or args.force:
        response = requests.get(args.data_url, timeout=args.timeout_sec)
        response.raise_for_status()
        db_path.write_bytes(response.content)
        downloaded = True

    # 3) Сохраняем manifest для трассировки источника данных.
    manifest = {
        "data_url": args.data_url,
        "db_path": str(db_path.resolve()),
        "db_size_bytes": db_path.stat().st_size,
        "downloaded": downloaded,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 4) Печатаем артефакты шага ingest.
    print(f"[ingest] DB path: {db_path}")
    print(f"[ingest] Manifest path: {manifest_path}")
    print(f"[ingest] Downloaded now: {downloaded}")


if __name__ == "__main__":
    main()
