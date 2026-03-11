from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["img_width", "img_height", "pixel_mean", "pixel_std"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tabular dataset from images.")
    parser.add_argument(
        "--validated-labels",
        type=Path,
        default=Path("data/validated/labels_validated.csv"),
        help="Path to validated labels file.",
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
        default=Path("data/processed"),
        help="Path to output processed train/val tables.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def extract_features(image_path: Path) -> dict[str, float]:
    # Читаем изображение и приводим к RGB для стабильного расчета признаков.
    try:
        with Image.open(image_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot read image: {image_path}") from exc

    return {
        "img_width": float(arr.shape[1]),
        "img_height": float(arr.shape[0]),
        "pixel_mean": float(arr.mean()),
        "pixel_std": float(arr.std()),
    }


def write_table(df: pd.DataFrame, path_without_ext: Path) -> Path:
    # Пытаемся сохранить в parquet; если недоступен движок, fallback в csv.
    parquet_path = path_without_ext.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_without_ext.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> None:
    # 1) Читаем аргументы и проверяем входные пути.
    args = parse_args()
    if not args.validated_labels.exists():
        raise FileNotFoundError(f"Validated labels not found: {args.validated_labels}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    # 2) Для каждой строки labels извлекаем простые числовые признаки изображения.
    labels = pd.read_csv(args.validated_labels)
    rows: list[dict[str, float]] = []
    for _, row in labels.iterrows():
        image_path = args.images_dir / str(row["file_name"])
        features = extract_features(image_path)
        rows.append(
            {
                "file_name": row["file_name"],
                "real_age": float(row["real_age"]),
                **features,
            }
        )

    # 3) Собираем итоговый датасет и разбиваем на train/val.
    dataset = pd.DataFrame(rows)
    train_df, val_df = train_test_split(
        dataset,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 4) Сохраняем train/val в data/processed.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = write_table(train_df, args.output_dir / "train")
    val_path = write_table(val_df, args.output_dir / "val")

    # 5) Печатаем summary шага build_dataset.
    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved val: {val_path}")
    print(f"[build_dataset] Rows train={len(train_df)} val={len(val_df)}")


if __name__ == "__main__":
    main()
