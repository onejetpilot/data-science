from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.25
DATA_FILE = "6_class_1.csv"

RENAME_COLUMNS = {
    "Temperature (K)": "temperature",
    "Luminosity(L/Lo)": "luminosity",
    "Radius(R/Ro)": "radius",
    "Absolute magnitude(Mv)": "absolute_magnitude",
    "Star type": "star_type",
    "Star color": "star_color",
    "Spectral Class": "spectral_class",
}

COLOR_MAP = {
    "red": "red",
    "blue": "blue",
    "blue_white": "blue_white",
    "white": "white",
    "whitish": "white",
    "white_yellow": "yellow_white",
    "yellow_white": "yellow_white",
    "yellowish_white": "yellow_white",
    "yellowish": "yellow",
    "pale_yellow_orange": "yellow",
    "orange": "orange",
    "orange_red": "orange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for star_temp.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"), help="Validated input directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory.")
    return parser.parse_args()


def normalize_color(value: object) -> object:
    if isinstance(value, str):
        return value.strip().lower().replace("-", "_").replace(" ", "_")
    return value


def map_color(value: object) -> object:
    if not isinstance(value, str):
        return value
    return COLOR_MAP.get(value, value)


def make_ohe() -> OneHotEncoder:
    # sparse_output появился в новых версиях sklearn; fallback нужен для совместимости.
    try:
        return OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore")


def main() -> None:
    # 1) Загружаем валидированный CSV и переименовываем колонки.
    args = parse_args()
    raw_df = pd.read_csv(args.validated_dir / DATA_FILE)
    if "Unnamed: 0" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 0"])
    df = raw_df.rename(columns=RENAME_COLUMNS).copy()

    # 2) Повторяем чистку категорий по star_color как в ноутбуке.
    df["star_color"] = df["star_color"].apply(normalize_color).apply(map_color)
    df["spectral_class"] = df["spectral_class"].astype(str).str.strip().str.upper()
    df["star_type"] = pd.to_numeric(df["star_type"], errors="coerce").astype("Int64").astype(str)

    # 3) Приводим числовые признаки и убираем строки с неполными ключевыми значениями.
    num_cols = ["luminosity", "radius", "absolute_magnitude"]
    target_col = "temperature"
    for col in [target_col, *num_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[target_col, *num_cols, "star_color", "star_type", "spectral_class"]).reset_index(drop=True)

    # 4) Делим на train/val и строим preprocess трансформер.
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    cat_cols = ["star_color", "star_type", "spectral_class"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_ohe(), cat_cols),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    feature_names = list(preprocessor.get_feature_names_out())
    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    val_df = pd.DataFrame(X_val_t, columns=feature_names)
    train_df[target_col] = y_train.reset_index(drop=True).astype(float)
    val_df[target_col] = y_val.reset_index(drop=True).astype(float)

    # 5) Сохраняем processed датасеты и manifest.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"
    try:
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
    except Exception:
        train_path = args.output_dir / "train.csv"
        val_path = args.output_dir / "val.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

    manifest = {
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "target": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "output_features_count": int(len(feature_names)),
        "output_features": feature_names,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved val: {val_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Feature count: {len(feature_names)}")


if __name__ == "__main__":
    main()
