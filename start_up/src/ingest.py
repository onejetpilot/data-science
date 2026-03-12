from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from requests import HTTPError


TRAIN_FILE = "kaggle_startups_train_28062024.csv"
TEST_FILE = "kaggle_startups_test_28062024.csv"
CITIES_FILE = "worldcitiespop.csv"

DEFAULT_TRAIN_URL = f"https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/{TRAIN_FILE}"
DEFAULT_TEST_URL = f"https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/{TEST_FILE}"
DEFAULT_CITIES_URL = f"https://huggingface.co/datasets/onejetpilot/start_up/resolve/main/{CITIES_FILE}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw startup project datasets.")
    parser.add_argument("--train-url", type=str, default=DEFAULT_TRAIN_URL)
    parser.add_argument("--test-url", type=str, default=DEFAULT_TEST_URL)
    parser.add_argument("--cities-url", type=str, default=DEFAULT_CITIES_URL)
    parser.add_argument("--local-train-file", type=Path, default=None)
    parser.add_argument("--local-test-file", type=Path, default=None)
    parser.add_argument("--local-cities-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def download_file(url: str, target_path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)


def ensure_file(target_path: Path, url: str, local_file: Path | None, fallback_names: list[str]) -> str:
    if target_path.exists():
        print(f"[ingest] File already exists: {target_path}")
        return "existing"

    candidates: list[Path] = []
    if local_file is not None:
        candidates.append(local_file)
    candidates.extend(Path(name) for name in fallback_names)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            target_path.write_bytes(candidate.read_bytes())
            print(f"[ingest] Copied local file: {candidate} -> {target_path}")
            return f"local:{candidate}"

    print(f"[ingest] Downloading dataset: {url}")
    try:
        download_file(url, target_path)
        print(f"[ingest] Saved file: {target_path}")
        return f"url:{url}"
    except HTTPError as error:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Dataset download failed. Pass public URLs or provide local files "
            "with --local-train-file/--local-test-file/--local-cities-file."
        ) from error


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / TRAIN_FILE
    test_path = args.output_dir / TEST_FILE
    cities_path = args.output_dir / CITIES_FILE

    train_source = ensure_file(train_path, args.train_url, args.local_train_file, [TRAIN_FILE, "train.csv"])
    test_source = ensure_file(test_path, args.test_url, args.local_test_file, [TEST_FILE, "test.csv"])
    cities_source = ensure_file(cities_path, args.cities_url, args.local_cities_file, [CITIES_FILE, "cities.csv"])

    manifest = {
        "train_source": train_source,
        "test_source": test_source,
        "cities_source": cities_source,
        "train_file": str(train_path.resolve()),
        "test_file": str(test_path.resolve()),
        "cities_file": str(cities_path.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
