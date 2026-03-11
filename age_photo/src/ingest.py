from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy raw labels into project storage.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/faces"),
        help="Path with labels.csv and final_files/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path for raw outputs.",
    )
    return parser.parse_args()


def main() -> None:
    # 1) Читаем аргументы запуска и формируем пути к исходным данным.
    args = parse_args()
    source_dir = args.source_dir
    labels_path = source_dir / "labels.csv"
    images_dir = source_dir / "final_files"

    # 2) Проверяем, что обязательные файлы и папки действительно существуют.
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    # 3) Копируем labels.csv в слой raw данных.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path)
    labels_out = args.output_dir / "labels.csv"
    labels.to_csv(labels_out, index=False)

    # 4) Сохраняем manifest.json для трассировки источника и объема данных.
    manifest = {
        "rows": int(len(labels)),
        "source_dir": str(source_dir.resolve()),
        "labels_path": str(labels_out.resolve()),
        "images_dir": str(images_dir.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 5) Печатаем пути артефактов шага ingest.
    print(f"[ingest] Saved raw labels: {labels_out}")
    print(f"[ingest] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
