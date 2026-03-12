from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from unidecode import unidecode


TRAIN_FILE = "kaggle_startups_train_28062024.csv"
TEST_FILE = "kaggle_startups_test_28062024.csv"
CITIES_FILE = "worldcitiespop.csv"
EXP_DATE = pd.to_datetime("2018-01-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed datasets for startup model.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def process_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    splitted = out["category_list"].fillna("other").astype(str).str.split("|", expand=True).iloc[:, :5]
    col_names = [f"cat_{i + 1}" for i in range(5)]
    out[col_names] = splitted.fillna("other")
    return out.drop(columns=["category_list"], errors="ignore")


def add_date_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_train = train_df.copy()
    out_test = test_df.copy()

    date_cols_train = ["founded_at", "first_funding_at", "last_funding_at", "closed_at"]
    date_cols_test = ["founded_at", "first_funding_at", "last_funding_at"]
    for col in date_cols_train:
        if col in out_train.columns:
            out_train[col] = pd.to_datetime(out_train[col], errors="coerce")
    for col in date_cols_test:
        if col in out_test.columns:
            out_test[col] = pd.to_datetime(out_test[col], errors="coerce")

    out_train["closed_at"] = out_train.get("closed_at", pd.NaT).fillna(EXP_DATE)
    out_train["lifetime"] = (out_train["closed_at"] - out_train["founded_at"]).dt.days
    out_test["lifetime"] = (EXP_DATE - out_test["founded_at"]).dt.days

    out_train["fundingtime"] = (out_train["last_funding_at"] - out_train["first_funding_at"]).dt.days
    out_test["fundingtime"] = (out_test["last_funding_at"] - out_test["first_funding_at"]).dt.days

    out_train["lastf_expd"] = (EXP_DATE - out_train["last_funding_at"]).dt.days
    out_test["lastf_expd"] = (EXP_DATE - out_test["last_funding_at"]).dt.days

    out_train["firstf_expd"] = (EXP_DATE - out_train["first_funding_at"]).dt.days
    out_test["firstf_expd"] = (EXP_DATE - out_test["first_funding_at"]).dt.days

    out_train = out_train.drop(columns=[c for c in date_cols_train if c in out_train.columns], errors="ignore")
    out_test = out_test.drop(columns=[c for c in date_cols_test if c in out_test.columns], errors="ignore")
    return out_train, out_test


def normalize_city_column(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: unidecode(str(x)).lower() if pd.notna(x) else "")


def merge_city_features(train_df: pd.DataFrame, test_df: pd.DataFrame, city_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_filtered = city_df.sort_values("Population", ascending=False).drop_duplicates(subset=["City"])

    loc_train = train_df.copy()
    loc_test = test_df.copy()
    loc_train["city"] = normalize_city_column(loc_train["city"])
    loc_test["city"] = normalize_city_column(loc_test["city"])
    city_filtered = city_filtered.copy()
    city_filtered["City"] = city_filtered["City"].astype(str).str.lower()

    merged_train = pd.merge(loc_train, city_filtered, left_on="city", right_on="City", how="left")
    merged_test = pd.merge(loc_test, city_filtered, left_on="city", right_on="City", how="left")

    drop_cols = ["Country", "City", "AccentCity", "Region", "state_code", "region", "city", "country_code"]
    merged_train = merged_train.drop(columns=drop_cols, errors="ignore")
    merged_test = merged_test.drop(columns=drop_cols, errors="ignore")

    # Колонки исключены в ноутбуке по корреляции/мультиколлинеарности.
    prune_cols = ["cat_2", "cat_3", "cat_4", "cat_5", "firstf_expd"]
    merged_train = merged_train.drop(columns=prune_cols, errors="ignore")
    merged_test = merged_test.drop(columns=prune_cols, errors="ignore")
    return merged_train, merged_test


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.validated_dir / TRAIN_FILE)
    test_df = pd.read_csv(args.validated_dir / TEST_FILE)
    city_df = pd.read_csv(args.validated_dir / CITIES_FILE, low_memory=False)

    train_df = train_df.drop_duplicates().copy()

    cat_train = process_categories(train_df)
    cat_test = process_categories(test_df)
    feat_train, feat_test = add_date_features(cat_train, cat_test)
    model_train, model_test = merge_city_features(feat_train, feat_test, city_df)

    # Фиксируем структуру train/test для этапа train.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    test_path = args.output_dir / "test.parquet"
    try:
        model_train.to_parquet(train_path, index=False)
        model_test.to_parquet(test_path, index=False)
    except Exception:
        train_path = args.output_dir / "train.csv"
        test_path = args.output_dir / "test.csv"
        model_train.to_csv(train_path, index=False)
        model_test.to_csv(test_path, index=False)

    numeric_features = model_train.select_dtypes(include=["number"]).columns.difference(["status"]).tolist()
    categorical_features = model_train.select_dtypes(include=["object"]).columns.difference(["status", "name"]).tolist()
    manifest = {
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
        "rows_train": int(len(model_train)),
        "rows_test": int(len(model_test)),
        "target": "status",
        "id_column": "name",
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_columns": ["cat_2", "cat_3", "cat_4", "cat_5", "firstf_expd"],
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_path}")
    print(f"[build_dataset] Saved test: {test_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Train rows: {len(model_train)}")


if __name__ == "__main__":
    main()
