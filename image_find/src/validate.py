from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate core tables for image_find.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/find_image/to_upload"),
        help="Directory with extracted source files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    # 1) Проверяем обязательные файлы.
    args = parse_args()
    train_path = args.raw_dir / "train_dataset.csv"
    expert_path = args.raw_dir / "ExpertAnnotations.tsv"
    crowd_path = args.raw_dir / "CrowdAnnotations.tsv"

    for p in [train_path, expert_path, crowd_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # 2) Загружаем таблицы и проверяем обязательные колонки.
    train_df = pd.read_csv(train_path)
    expert_df = pd.read_csv(expert_path, sep="\t", header=None)
    crowd_df = pd.read_csv(crowd_path, sep="\t", header=None)

    expert_df.columns = ["image", "query_id", "rate_1", "rate_2", "rate_3"]
    crowd_df.columns = ["image", "query_id", "share_confirmed", "count_confirmed", "count_rejected"]

    errors: list[str] = []
    warnings_list: list[str] = []

    required_train_cols = {"image", "query_id", "query_text"}
    missing_train_cols = sorted(required_train_cols.difference(train_df.columns))
    if missing_train_cols:
        errors.append(f"train_dataset missing columns: {missing_train_cols}")

    if len(train_df) == 0:
        errors.append("train_dataset is empty")

    if train_df.duplicated(subset=["image", "query_id"]).any():
        warnings_list.append("train_dataset has duplicate (image, query_id) pairs")

    # 3) Проверяем числовые поля экспертов и крауда.
    for col in ["rate_1", "rate_2", "rate_3"]:
        n_invalid = int(pd.to_numeric(expert_df[col], errors="coerce").isna().sum())
        if n_invalid > 0:
            warnings_list.append(f"expert {col} invalid values: {n_invalid}")

    n_invalid_crowd = int(pd.to_numeric(crowd_df["share_confirmed"], errors="coerce").isna().sum())
    if n_invalid_crowd > 0:
        warnings_list.append(f"crowd share_confirmed invalid values: {n_invalid_crowd}")

    # 4) Сохраняем quality report.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "raw_dir": str(args.raw_dir.resolve()),
        "train_rows": int(len(train_df)),
        "expert_rows": int(len(expert_df)),
        "crowd_rows": int(len(crowd_df)),
        "warnings": warnings_list,
        "errors": errors,
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if errors:
        print(f"[validate] Quality check failed, see: {report_path}")
        raise ValueError("; ".join(errors))

    # 5) Сохраняем валидированные таблицы в data/validated.
    train_out = args.output_dir / "train_dataset.csv"
    expert_out = args.output_dir / "expert_annotations.csv"
    crowd_out = args.output_dir / "crowd_annotations.csv"
    train_df.to_csv(train_out, index=False)
    expert_df.to_csv(expert_out, index=False)
    crowd_df.to_csv(crowd_out, index=False)

    print(f"[validate] Saved validated train: {train_out}")
    print(f"[validate] Saved validated expert: {expert_out}")
    print(f"[validate] Saved validated crowd: {crowd_out}")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
