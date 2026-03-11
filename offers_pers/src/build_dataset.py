from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_COL = "покупательская_активность"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed dataset for offers_pers.")
    parser.add_argument("--validated-dir", type=Path, default=Path("data/validated"), help="Validated input directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Processed output directory.")
    return parser.parse_args()


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.replace(r"([A-Z])", r"_\1", regex=True)
        .str.lower()
        .str.lstrip("_")
        .str.replace(r"\s+", "_", regex=True)
    )
    return out


def main() -> None:
    # 1) Читаем валидированные таблицы.
    args = parse_args()
    m_file = pd.read_csv(args.validated_dir / "market_file.csv")
    m_money = pd.read_csv(args.validated_dir / "market_money.csv")
    m_time = pd.read_csv(args.validated_dir / "market_time.csv")
    money = pd.read_csv(args.validated_dir / "money.csv")

    # 2) Исправляем ошибки и типы как в ноутбуке.
    m_file["Тип сервиса"] = m_file["Тип сервиса"].replace("стандартт", "стандарт")
    m_money["Период"] = m_money["Период"].replace("предыдцщий_месяц", "предыдущий_месяц")
    m_money["Выручка"] = pd.to_numeric(m_money["Выручка"], errors="coerce").astype("float32") / 1000.0

    # 3) Приводим названия к snake_case.
    m_file = snake_case_columns(m_file)
    m_money = snake_case_columns(m_money)
    m_time = snake_case_columns(m_time)
    money = snake_case_columns(money)

    # 4) Разворачиваем периоды в отдельные колонки.
    m_money = pd.pivot_table(data=m_money, index="id", columns="период", values="выручка")
    rename_money = {
        "предыдущий_месяц": "выручка_предыдущий_месяц",
        "препредыдущий_месяц": "выручка_препредыдущий_месяц",
        "текущий_месяц": "выручка_текущий_месяц",
    }
    m_money = m_money.rename(columns=rename_money)
    m_time = pd.pivot_table(data=m_time, index="id", columns="период", values="минут")
    rename_time = {
        "предыдущий_месяц": "время_предыдущий_месяц",
        "текущий_месяц": "время_текущий_месяц",
    }
    m_time = m_time.rename(columns=rename_time)

    # 5) Фильтры и объединение как в ноутбуке.
    m_money = m_money[m_money["выручка_текущий_месяц"] < 10]
    m_money = m_money[
        (m_money["выручка_предыдущий_месяц"] > 0)
        & (m_money["выручка_препредыдущий_месяц"] > 0)
        & (m_money["выручка_текущий_месяц"] > 0)
    ]
    df = m_file.merge(m_money, on="id").merge(m_time, on="id")
    df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
    df = df.replace({"Прежний уровень": 0, "Снизилась": 1})

    # 6) Создаем технические признаки и удаляем, как в ноутбуке.
    df_2 = df.copy()
    df_2["выручка_тек_пре"] = df_2["выручка_текущий_месяц"] - df_2["выручка_предыдущий_месяц"]
    df_2["выручка_пре_препре"] = df_2["выручка_предыдущий_месяц"] - df_2["выручка_препредыдущий_месяц"]
    df_2 = df_2.drop(columns=["выручка_тек_пре", "выручка_пре_препре"])
    df_2 = df_2.reset_index(drop=True)

    # 7) Сохраняем итоговый processed датасет и manifest.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "dataset.parquet"
    try:
        df_2.to_parquet(data_path, index=False)
    except Exception:
        data_path = args.output_dir / "dataset.csv"
        df_2.to_csv(data_path, index=False)

    manifest = {
        "target": TARGET_COL,
        "id_column": "id",
        "categorical_expected": ["разрешить_сообщать", "популярная_категория", "тип_сервиса"],
        "dataset_path": str(data_path.resolve()),
        "rows": int(len(df_2)),
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[build_dataset] Saved dataset: {data_path}")
    print(f"[build_dataset] Saved manifest: {manifest_path}")
    print(f"[build_dataset] Rows: {len(df_2)}")


if __name__ == "__main__":
    main()
