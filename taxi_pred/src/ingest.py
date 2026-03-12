from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from requests import HTTPError


DATA_FILE = "taxi.csv"
DEFAULT_DATA_URL = "https://huggingface.co/datasets/onejetpilot/taxi/resolve/main/taxi.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw taxi time-series dataset.")
    parser.add_argument("--data-url", type=str, default=DEFAULT_DATA_URL)
    parser.add_argument("--local-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def download_file(url: str, target_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / DATA_FILE

    if raw_path.exists():
        source = "existing"
        print(f"[ingest] File already exists: {raw_path}")
    else:
        copied = False
        for candidate in [args.local_file, Path(DATA_FILE)]:
            if candidate is not None and candidate.exists() and candidate.is_file():
                raw_path.write_bytes(candidate.read_bytes())
                source = f"local:{candidate}"
                copied = True
                print(f"[ingest] Copied local file: {candidate} -> {raw_path}")
                break

        if not copied:
            print(f"[ingest] Downloading dataset: {args.data_url}")
            try:
                download_file(args.data_url, raw_path)
                source = f"url:{args.data_url}"
                print(f"[ingest] Saved file: {raw_path}")
            except HTTPError as error:
                if raw_path.exists():
                    raw_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Dataset download failed. Pass another public URL via --data-url "
                    "or provide --local-file path to CSV."
                ) from error

    manifest = {"source": source, "raw_file": str(raw_path.resolve())}
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
