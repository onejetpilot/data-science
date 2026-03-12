from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {"id", "f0", "f1", "f2", "product"}
FILES = ["geo_data_0.csv", "geo_data_1.csv", "geo_data_2.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw well location datasets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict[str, int | list[str]]] = {}
    for filename in FILES:
        src = args.raw_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Raw file not found: {src}")
        df = pd.read_csv(src)
        missing = sorted(EXPECTED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"[{filename}] missing columns: {missing}")

        for col in ["f0", "f1", "f2", "product"]:
            bad = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
            if bad > 0:
                raise ValueError(f"[{filename}] invalid numeric values in {col}: {bad}")

        shutil.copyfile(src, args.output_dir / filename)
        report[filename] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "null_cells": int(df.isna().sum().sum()),
            "duplicated_rows": int(df.duplicated().sum()),
            "missing_columns": missing,
        }

    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[validate] Saved report: {report_path}")


if __name__ == "__main__":
    main()
