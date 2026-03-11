from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd


CAT_DROP_LIST = [
    "cat_124",
    "cat_415",
    "cat_244",
    "cat_431",
    "cat_432",
    "cat_344",
    "cat_57",
    "cat_505",
    "cat_445",
    "cat_326",
    "cat_243",
    "cat_440",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed dataset for purchase_prediction.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"), help="Validated input directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory for processed dataset.")
    return parser.parse_args()


def parse_category_ids(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            return []
    return []


def strip_message_prefix(value: object) -> object:
    if isinstance(value, str):
        parts = value.split("-")
        if len(parts) > 2:
            return "-".join(parts[2:])
    return value


def main() -> None:
    # 1) Загружаем валидированные таблицы.
    args = parse_args()
    messages = pd.read_csv(args.validated_dir / "apparel-messages.csv")
    purchases = pd.read_csv(args.validated_dir / "apparel-purchases.csv")
    target = pd.read_csv(args.validated_dir / "apparel-target_binary.csv")

    # 2) Объединяем таблицы так же, как в ноутбуке.
    data = purchases.merge(messages, on=["client_id", "message_id"], how="left", suffixes=("_buy", "_msg"))
    data = data.merge(target, on="client_id", how="left")
    if "date_msg" in data.columns:
        data = data.drop(columns=["date_msg"])

    # 3) Приводим даты и фильтруем выбросы.
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce")
    data["date_buy"] = pd.to_datetime(data["date_buy"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce")
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data_fltr = data[(data["price"] <= 4000) & (data["quantity"] <= 1)].copy()
    data_fltr = data_fltr.drop(columns=["quantity"])

    # 4) Преобразуем category_ids и строим бинарные cat_* признаки по top-100.
    data_fltr["category_ids"] = data_fltr["category_ids"].apply(parse_category_ids)
    all_categories = pd.Series([cat for cats in data_fltr["category_ids"] for cat in cats], dtype="object")
    top_100_list = all_categories.value_counts().head(100).index.tolist()
    top_100_set = set(top_100_list)
    data_fltr["category_ids"] = data_fltr["category_ids"].apply(
        lambda cat_list: [cat if cat in top_100_set else "other" for cat in cat_list]
    )
    binary_columns = {f"cat_{cat}": data_fltr["category_ids"].apply(lambda row, c=cat: int(c in row)) for cat in top_100_list}
    binary_data = pd.DataFrame(binary_columns)
    data_cat = pd.concat([data_fltr.reset_index(drop=True), binary_data.reset_index(drop=True)], axis=1)
    data_cat = data_cat.drop(columns=["category_ids"])

    # 5) Дорабатываем идентификаторы и временные признаки.
    data_cat["message_id"] = data_cat["message_id"].apply(strip_message_prefix)
    for col in ["date_buy", "created_at"]:
        data_cat[f"{col}_dayofweek"] = data_cat[col].dt.dayofweek.astype("Int64").astype(str)
        data_cat[f"{col}_month"] = data_cat[col].dt.month.astype("Int64").astype(str)
    data_cat = data_cat.drop(columns=["date_buy", "created_at"])
    data_cat = data_cat.drop(columns=["created_at_dayofweek", "created_at_month"], errors="ignore")

    # 6) Отбираем часть cat_* признаков по корреляции, как в ноутбуке.
    cat_cols = [col for col in data_cat.columns if col.startswith("cat_")]
    data_corr = data_cat.drop(columns=["client_id", "message_id", "bulk_campaign_id", *cat_cols], errors="ignore")
    numeric_slice = [col for col in data_cat.columns[7:107] if col.startswith("cat_")]
    cols_to_train: list[str] = []
    for col in numeric_slice:
        corr_val = pd.to_numeric(data_cat[col], errors="coerce").corr(pd.to_numeric(data_cat["target"], errors="coerce"))
        if pd.notna(corr_val) and abs(corr_val) >= 0.01:
            cols_to_train.append(col)

    # 7) Формируем финальный датасет и удаляем колонки с высоким VIF из ноутбука.
    final_features = list(dict.fromkeys(cols_to_train + list(data_corr.columns) + ["client_id"]))
    data_final = data_cat[final_features].copy()
    drop_existing = [col for col in CAT_DROP_LIST if col in data_final.columns]
    data_clean = data_final.drop(columns=drop_existing, errors="ignore")
    data_complete = data_clean.drop_duplicates().copy()

    # 8) Финальная подготовка target/категориальных колонок для train шага.
    data_complete["target"] = pd.to_numeric(data_complete["target"], errors="coerce")
    data_complete = data_complete.dropna(subset=["target"])
    data_complete["target"] = data_complete["target"].astype(int)
    for col in ["event", "channel", "date_buy_dayofweek", "date_buy_month"]:
        if col in data_complete.columns:
            data_complete[col] = data_complete[col].astype(str).fillna("other")

    # 9) Сохраняем processed датасет и manifest с описанием признаков.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "dataset.parquet"
    try:
        data_complete.to_parquet(data_path, index=False)
    except Exception:
        data_path = args.output_dir / "dataset.csv"
        data_complete.to_csv(data_path, index=False)

    bin_features = [col for col in data_complete.columns if col.startswith("cat_")]
    cat_features = [col for col in ["event", "channel", "date_buy_dayofweek", "date_buy_month"] if col in data_complete.columns]
    num_features = [col for col in ["price"] if col in data_complete.columns]
    manifest = {
        "dataset_path": str(data_path.resolve()),
        "rows": int(len(data_complete)),
        "target": "target",
        "cat_features": cat_features,
        "bin_features_count": int(len(bin_features)),
        "num_features": num_features,
        "dropped_high_vif_columns": drop_existing,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved dataset: {data_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Rows: {len(data_complete)}")
    print(f"[build_dataset] Binary features kept: {len(bin_features)}")


if __name__ == "__main__":
    main()
