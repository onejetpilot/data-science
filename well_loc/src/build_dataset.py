from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILES = ["geo_data_0.csv", "geo_data_1.csv", "geo_data_2.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for well location analysis.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, object]] = {}
    for idx, filename in enumerate(FILES):
        df = pd.read_csv(args.validated_dir / filename)
        df = df.drop_duplicates().copy()
        for col in ["f0", "f1", "f2", "product"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["f0", "f1", "f2", "product"]).reset_index(drop=True)

        out_name = f"region_{idx}.parquet"
        out_path = args.output_dir / out_name
        try:
            df.to_parquet(out_path, index=False)
        except Exception:
            out_name = f"region_{idx}.csv"
            out_path = args.output_dir / out_name
            df.to_csv(out_path, index=False)

        manifest[f"region_{idx}"] = {
            "path": str(out_path.resolve()),
            "rows": int(len(df)),
            "features": ["f0", "f1", "f2"],
            "target": "product",
        }
        print(f"[build_dataset] Saved {out_name} with rows={len(df)}")

    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build_dataset] Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
