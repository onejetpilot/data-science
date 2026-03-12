from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from requests import HTTPError


DEFAULT_DATA_URL = "https://huggingface.co/datasets/onejetpilot/star_temp/resolve/main/6_class_1.csv"
DATA_FILE = "6_class_1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw star temperature dataset.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL, help="Direct URL to CSV dataset.")
    parser.add_argument(
        "--local-file",
        type=Path,
        default=None,
        help="Optional local CSV path. If provided and exists, download step is skipped.",
    )
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
    # 1) Готовим директорию сырого слоя и путь к CSV.
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / DATA_FILE

    # 2) Пытаемся использовать local-file, затем локальный fallback, затем URL.
    candidate_local_files = []
    if args.local_file is not None:
        candidate_local_files.append(args.local_file)
    candidate_local_files.extend([Path("6_class_1.csv"), Path("6_class.csv")])

    if raw_path.exists():
        print(f"[ingest] File already exists: {raw_path}")
    else:
        copied = False
        for local_candidate in candidate_local_files:
            if local_candidate.exists():
                raw_path.write_bytes(local_candidate.read_bytes())
                print(f"[ingest] Copied local file: {local_candidate} -> {raw_path}")
                copied = True
                break

        if not copied:
            print(f"[ingest] Downloading dataset: {args.data_url}")
            try:
                download_file(args.data_url, raw_path)
                print(f"[ingest] Saved file: {raw_path}")
            except HTTPError as error:
                if raw_path.exists():
                    raw_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Dataset download failed. Pass another public URL via --data-url "
                    "or provide --local-file path to CSV."
                ) from error

    # 3) Сохраняем манифест источника.
    manifest = {
        "data_url": args.data_url,
        "local_file": str(args.local_file) if args.local_file else None,
        "raw_file": str(raw_path.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
