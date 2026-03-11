from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw labels and image presence.")
    parser.add_argument(
        "--raw-labels",
        type=Path,
        default=Path("data/raw/labels.csv"),
        help="Path to raw labels.csv.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/faces/final_files"),
        help="Path to image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validated"),
        help="Path to save validated labels and report.",
    )
    return parser.parse_args()


def main() -> None:
    # 1) Читаем аргументы и проверяем наличие входных файлов.
    args = parse_args()
    if not args.raw_labels.exists():
        raise FileNotFoundError(f"Raw labels not found: {args.raw_labels}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    # 2) Загружаем таблицу и проверяем обязательные колонки схемы.
    df = pd.read_csv(args.raw_labels)
    required_cols = {"file_name", "real_age"}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 3) Запускаем quality checks, накапливая ошибки в единый список.
    errors: list[str] = []
    if df["file_name"].isna().any():
        errors.append("file_name contains null values")
    if df["real_age"].isna().any():
        errors.append("real_age contains null values")

    df["real_age"] = pd.to_numeric(df["real_age"], errors="coerce")
    if df["real_age"].isna().any():
        errors.append("real_age contains non-numeric values")

    duplicates = int(df["file_name"].duplicated().sum())
    if duplicates > 0:
        errors.append(f"file_name has duplicates: {duplicates}")

    range_mask = (df["real_age"] < 0) | (df["real_age"] > 100)
    out_of_range = int(range_mask.sum())
    if out_of_range > 0:
        errors.append(f"real_age outside [0, 100]: {out_of_range}")

    df["image_exists"] = df["file_name"].apply(lambda x: (args.images_dir / str(x)).exists())
    missing_images = int((~df["image_exists"]).sum())
    if missing_images > 0:
        errors.append(f"images missing on disk: {missing_images}")

    # 4) Сохраняем полный отчет по качеству независимо от результата проверок.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "rows": int(len(df)),
        "duplicates": duplicates,
        "out_of_range": out_of_range,
        "missing_images": missing_images,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # 5) Если есть ошибки, завершаем шаг с исключением.
    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    # 6) Если проверка прошла, сохраняем валидированные данные.
    validated_path = args.output_dir / "labels_validated.csv"
    df.to_csv(validated_path, index=False)
    print(f"[validate] Saved validated labels: {validated_path}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
