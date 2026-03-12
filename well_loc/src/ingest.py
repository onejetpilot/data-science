from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from requests import HTTPError


FILES = {
    "geo_data_0.csv": "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_0.csv",
    "geo_data_1.csv": "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_1.csv",
    "geo_data_2.csv": "https://huggingface.co/datasets/onejetpilot/well_loc/resolve/main/geo_data_2.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw well location datasets.")
    parser.add_argument("--geo0-url", type=str, default=FILES["geo_data_0.csv"])
    parser.add_argument("--geo1-url", type=str, default=FILES["geo_data_1.csv"])
    parser.add_argument("--geo2-url", type=str, default=FILES["geo_data_2.csv"])
    parser.add_argument("--local-geo0-file", type=Path, default=None)
    parser.add_argument("--local-geo1-file", type=Path, default=None)
    parser.add_argument("--local-geo2-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def download_file(url: str, target_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)


def ensure_file(target_path: Path, url: str, local_file: Path | None) -> str:
    if target_path.exists():
        print(f"[ingest] File already exists: {target_path}")
        return "existing"

    for candidate in [local_file, Path(target_path.name)]:
        if candidate is not None and candidate.exists() and candidate.is_file():
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
        raise RuntimeError("Dataset download failed. Use public URLs or local files.") from error


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "geo_data_0.csv": args.geo0_url,
        "geo_data_1.csv": args.geo1_url,
        "geo_data_2.csv": args.geo2_url,
    }
    locals_map = {
        "geo_data_0.csv": args.local_geo0_file,
        "geo_data_1.csv": args.local_geo1_file,
        "geo_data_2.csv": args.local_geo2_file,
    }

    sources: dict[str, str] = {}
    for filename in ["geo_data_0.csv", "geo_data_1.csv", "geo_data_2.csv"]:
        target = args.output_dir / filename
        sources[filename] = ensure_file(target, urls[filename], locals_map[filename])

    manifest = {
        "sources": sources,
        "files": {name: str((args.output_dir / name).resolve()) for name in sources},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ingest] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
