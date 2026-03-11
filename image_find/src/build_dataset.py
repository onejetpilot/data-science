from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
except Exception:
    nltk = None
    WordNetLemmatizer = None


TARGET_COLUMN = "target"
TEXT_COLUMN = "query_text"
IMAGE_COLUMN = "image"
GROUP_COLUMN = "query_id"
BLOCK = {
    "child",
    "kid",
    "minor",
    "underage",
    "infant",
    "baby",
    "toddler",
    "preschool",
    "kindergarten",
    "schoolboy",
    "schoolgirl",
    "schoolchild",
    "preteen",
    "teen",
    "teenager",
    "boy",
    "girl",
    "boys",
    "girls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed train/val dataset for image_find.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def write_table(df: pd.DataFrame, base_path: Path) -> Path:
    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = base_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def majority_vote(row: pd.Series) -> float:
    vals = [row["rate_1"], row["rate_2"], row["rate_3"]]
    for v in set(vals):
        if vals.count(v) >= 2:
            return float(v)
    return float("nan")


def build_target(df: pd.DataFrame, weight_expert: float = 0.6, weight_crowd: float = 0.4, scale: int = 4) -> pd.DataFrame:
    data = df.copy()
    expert_cols = ["rate_1", "rate_2", "rate_3"]
    data[expert_cols] = data[expert_cols].apply(pd.to_numeric, errors="coerce") / scale
    data["share_confirmed"] = pd.to_numeric(data["share_confirmed"], errors="coerce")
    data["majority_vote"] = data.apply(majority_vote, axis=1)

    def get_target(row: pd.Series) -> float:
        if pd.notna(row["majority_vote"]) and pd.notna(row["share_confirmed"]):
            return weight_expert * row["majority_vote"] + weight_crowd * row["share_confirmed"]
        if pd.notna(row["majority_vote"]):
            return row["majority_vote"]
        return row["share_confirmed"]

    data[TARGET_COLUMN] = data.apply(get_target, axis=1)
    return data


def get_lemmas(text: str, lemmatizer: WordNetLemmatizer | None) -> list[str]:
    if not isinstance(text, str):
        return []
    text = re.sub(r"[^A-Za-z]", " ", text).lower()
    tokens = text.split()
    if lemmatizer is None:
        return tokens
    try:
        return [lemmatizer.lemmatize(w) for w in tokens]
    except Exception:
        return tokens


def mark_and_drop_children(df: pd.DataFrame, text_col: str = "query_text") -> pd.DataFrame:
    data = df.copy()
    lemmatizer = WordNetLemmatizer() if WordNetLemmatizer is not None else None

    def check_block(text: str) -> int:
        lemmas = get_lemmas(text, lemmatizer)
        return int(any(w in BLOCK for w in lemmas))

    data["to_block"] = data[text_col].apply(check_block)
    return data


def main() -> None:
    # 1) Читаем валидированные таблицы.
    args = parse_args()
    train_path = args.validated_dir / "train_dataset.csv"
    expert_path = args.validated_dir / "expert_annotations.csv"
    crowd_path = args.validated_dir / "crowd_annotations.csv"

    for p in [train_path, expert_path, crowd_path]:
        if not p.exists():
            raise FileNotFoundError(f"Validated file not found: {p}")

    train_df = pd.read_csv(train_path)
    expert_df = pd.read_csv(expert_path)
    crowd_df = pd.read_csv(crowd_path)

    # 2) Фильтруем crowd как в ноутбуке и объединяем таблицы.
    crowd_df["share_confirmed"] = pd.to_numeric(crowd_df["share_confirmed"], errors="coerce")
    crowd_df = crowd_df[(crowd_df["share_confirmed"] <= 0.2) | (crowd_df["share_confirmed"] >= 0.8)].copy()
    scores = expert_df.merge(crowd_df[["image", "query_id", "share_confirmed"]], on=["image", "query_id"], how="outer")
    merged = train_df.merge(scores, on=["image", "query_id"], how="outer")

    # 3) Формируем target и восстанавливаем пропуски query_text через query_id.
    data = build_target(merged)
    data["len_tokens"] = data[TEXT_COLUMN].fillna("").str.split().str.len()
    data = data[data["len_tokens"] <= 25].copy().drop(columns=["len_tokens"])

    fill_map = (
        data.dropna(subset=[TEXT_COLUMN])
        .drop_duplicates(subset=[GROUP_COLUMN])
        .set_index(GROUP_COLUMN)[TEXT_COLUMN]
    )
    data[TEXT_COLUMN] = data[TEXT_COLUMN].fillna(data[GROUP_COLUMN].map(fill_map))
    data = data.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).reset_index(drop=True)

    # 4) Удаляем строки с запрещенным контентом по логике ноутбука.
    data = mark_and_drop_children(data, text_col=TEXT_COLUMN)
    blocked = data.loc[data["to_block"] == 1, [GROUP_COLUMN, TEXT_COLUMN]].drop_duplicates()
    blocked["image"] = blocked[GROUP_COLUMN].astype(str).str[:-2]
    blocked_images = blocked[["image", TEXT_COLUMN]].drop_duplicates()
    train_clean = (
        data.loc[~data[IMAGE_COLUMN].astype(str).isin(blocked_images["image"])]
        .drop(columns=["to_block"])
        .reset_index(drop=True)
    )

    train_clean = train_clean[[TEXT_COLUMN, IMAGE_COLUMN, GROUP_COLUMN, TARGET_COLUMN]].copy()
    train_clean = train_clean.dropna(subset=[TARGET_COLUMN, IMAGE_COLUMN, GROUP_COLUMN])
    data = train_clean
    data["target_bin"] = (data[TARGET_COLUMN] >= 0.5).astype(int)

    # 5) Делим train/val по группам image (как в ноутбуке).
    gss = GroupShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.random_state)
    groups = data[IMAGE_COLUMN].astype(str).fillna("__na__")
    train_idx, val_idx = next(gss.split(data, data["target_bin"], groups=groups))
    train_out_df = data.iloc[train_idx].reset_index(drop=True)
    val_out_df = data.iloc[val_idx].reset_index(drop=True)

    # 6) Сохраняем выборки и manifest признаков.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = write_table(train_out_df, args.output_dir / "train")
    val_out = write_table(val_out_df, args.output_dir / "val")

    manifest = {
        "target": "target_bin",
        "text_feature": TEXT_COLUMN,
        "image_feature": IMAGE_COLUMN,
        "group_feature": IMAGE_COLUMN,
        "target_continuous": TARGET_COLUMN,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved train: {train_out}")
    print(f"[build_dataset] Saved val: {val_out}")
    print(f"[build_dataset] Saved feature manifest: {manifest_path}")
    print(f"[build_dataset] Rows train={len(train_out_df)} val={len(val_out_df)}")


if __name__ == "__main__":
    main()
