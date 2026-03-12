from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
TARGET_COL = "temperature"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and tuned PyTorch models for star_temp.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed input directory.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Output artifacts directory.")
    return parser.parse_args()


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    parquet_path = processed_dir / f"{split_name}.parquet"
    csv_path = processed_dir / f"{split_name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Split not found: {split_name}.parquet/csv")


class MyNetA(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyNetB(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyNetC(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyNetD(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyNetE(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MyNetCD(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_full_batch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 500,
    lr: float = 1e-1,
    patience: int = 10,
    min_delta: float = 1e-3,
) -> dict[str, float | int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    last_train_rmse = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        preds = model(x_train)
        loss = loss_fn(preds, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_rmse = float(torch.sqrt(loss).item())
        last_train_rmse = train_rmse

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = loss_fn(val_preds, y_val)
            val_rmse = float(torch.sqrt(val_loss).item())

        if best_rmse - val_rmse > min_delta:
            best_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "best_val_rmse": float(best_rmse),
        "best_epoch": int(best_epoch),
        "last_train_rmse": float(last_train_rmse),
    }


def train_with_batches(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    epochs: int = 500,
    lr: float = 1e-1,
    patience: int = 10,
    min_delta: float = 1e-3,
) -> dict[str, float | int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_rmse = float("inf")
    best_epoch = 0
    patience_counter = 0
    last_train_rmse = float("nan")
    y_pred_best: torch.Tensor | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for xb, yb in dataloader:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        train_rmse = float(np.sqrt(mean_loss))
        last_train_rmse = train_rmse

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = loss_fn(val_preds, y_val)
            val_rmse = float(torch.sqrt(val_loss).item())

        if best_rmse - val_rmse > min_delta:
            best_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            y_pred_best = val_preds.detach().clone()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    y_true = y_val.detach().cpu().numpy().squeeze()
    y_pred = (y_pred_best if y_pred_best is not None else val_preds).detach().cpu().numpy().squeeze()
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "best_val_rmse": float(best_rmse),
        "best_epoch": int(best_epoch),
        "last_train_rmse": float(last_train_rmse),
        "val_mae": mae,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def main() -> None:
    # 1) Читаем train/val сплиты после build_dataset.
    args = parse_args()
    train_df = read_split(args.processed_dir, "train")
    val_df = read_split(args.processed_dir, "val")
    if TARGET_COL not in train_df.columns or TARGET_COL not in val_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in processed splits")

    x_train = train_df.drop(columns=[TARGET_COL]).to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_val = val_df.drop(columns=[TARGET_COL]).to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)
    input_dim = x_train.shape[1]

    # 2) Baseline: перебираем архитектуры из ноутбука.
    baseline_classes = [MyNetA, MyNetB, MyNetC, MyNetD, MyNetE]
    baseline_results: dict[str, dict[str, float | int]] = {}
    for model_class in baseline_classes:
        set_seed(RANDOM_STATE)
        model = model_class(input_dim=input_dim)
        result = train_full_batch(model, x_train_t, y_train_t, x_val_t, y_val_t)
        baseline_results[model_class.__name__] = result
        print(f"[train] baseline {model_class.__name__}: val_rmse={result['best_val_rmse']:.2f}")

    best_baseline_name = min(baseline_results, key=lambda name: float(baseline_results[name]["best_val_rmse"]))

    # 3) Улучшение: подбираем dropout и batch_size для MyNetCD.
    batch_sizes = [16, 32, 64, len(x_train_t)]
    dropouts = [0.0, 0.2, 0.4]
    tuned_results: dict[str, dict[str, float | int]] = {}
    best_tuned: dict[str, float | int | str | list[float]] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None

    for dropout in dropouts:
        for batch_size in batch_sizes:
            set_seed(RANDOM_STATE)
            model = MyNetCD(input_dim=input_dim, dropout=dropout)
            result = train_with_batches(
                model=model,
                x_train=x_train_t,
                y_train=y_train_t,
                x_val=x_val_t,
                y_val=y_val_t,
                batch_size=batch_size,
            )
            key = f"dropout={dropout}|batch_size={batch_size}"
            tuned_results[key] = {
                "best_val_rmse": float(result["best_val_rmse"]),
                "best_epoch": int(result["best_epoch"]),
                "last_train_rmse": float(result["last_train_rmse"]),
                "val_mae": float(result["val_mae"]),
            }
            print(
                "[train] tuned "
                f"dropout={dropout}, batch_size={batch_size}: val_rmse={float(result['best_val_rmse']):.2f}"
            )

            if best_tuned is None or float(result["best_val_rmse"]) < float(best_tuned["best_val_rmse"]):
                best_tuned = {
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "best_val_rmse": float(result["best_val_rmse"]),
                    "best_epoch": int(result["best_epoch"]),
                    "last_train_rmse": float(result["last_train_rmse"]),
                    "val_mae": float(result["val_mae"]),
                    "y_true": result["y_true"],
                    "y_pred": result["y_pred"],
                }
                best_model_state = model.state_dict()

    if best_tuned is None or best_model_state is None:
        raise RuntimeError("No tuned model results produced")

    # 4) Сохраняем модель и артефакты.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.pt"
    torch.save(
        {
            "input_dim": input_dim,
            "dropout": float(best_tuned["dropout"]),
            "state_dict": best_model_state,
        },
        model_path,
    )

    pred_df = pd.DataFrame(
        {
            "star_idx": list(range(len(best_tuned["y_true"]))),
            "y_true": best_tuned["y_true"],
            "y_pred": best_tuned["y_pred"],
        }
    )
    pred_path = args.artifacts_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    best_baseline_rmse = float(baseline_results[best_baseline_name]["best_val_rmse"])
    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "input_dim": int(input_dim),
        "target": TARGET_COL,
        "baseline_results": baseline_results,
        "best_baseline_model": best_baseline_name,
        "best_baseline_val_rmse": best_baseline_rmse,
        "tuned_results": tuned_results,
        "best_tuned_model": {
            "architecture": "MyNetCD",
            "dropout": float(best_tuned["dropout"]),
            "batch_size": int(best_tuned["batch_size"]),
            "best_val_rmse": float(best_tuned["best_val_rmse"]),
            "val_mae": float(best_tuned["val_mae"]),
            "best_epoch": int(best_tuned["best_epoch"]),
        },
        "improvement_vs_best_baseline_rmse": float(best_baseline_rmse - float(best_tuned["best_val_rmse"])),
    }
    metrics_path = args.artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Best baseline: {best_baseline_name} ({best_baseline_rmse:.2f})")
    print(f"[train] Best tuned RMSE: {float(best_tuned['best_val_rmse']):.2f}")
    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Saved predictions: {pred_path}")


if __name__ == "__main__":
    main()
