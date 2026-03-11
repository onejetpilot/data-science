from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import Normalizer


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP on SBERT+IMG features for image_find.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/find_image/to_upload"))
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means use full train set.")
    return parser.parse_args()


def read_table(base_path: Path) -> pd.DataFrame:
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither parquet nor csv found for {base_path.name}")


def image_features(image_names: pd.Series, images_dir: Path) -> np.ndarray:
    # Эмбеддинги изображений через ResNet50 (pooling before fc) в духе ноутбука.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet50_Weights.DEFAULT
    backbone = models.resnet50(weights=weights)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
    preprocess = weights.transforms()

    unique_names = pd.Series(image_names.astype(str).unique())
    emb_map: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name in unique_names:
            path = images_dir / name
            try:
                with Image.open(path) as img:
                    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
                    vec = feature_extractor(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
                    emb_map[name] = vec
            except Exception:
                emb_map[name] = np.zeros((2048,), dtype=np.float32)
    return np.vstack([emb_map[n] for n in image_names.astype(str)])


def main() -> None:
    # 1) Загружаем train/val и метаданные признаков.
    args = parse_args()
    train_df = read_table(args.processed_dir / "train")
    val_df = read_table(args.processed_dir / "val")

    manifest_path = args.processed_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_col = manifest["target"]
    text_col = manifest["text_feature"]
    image_col = manifest["image_feature"]
    images_dir = args.raw_dir / "train_images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # 2) При необходимости ограничиваем train для быстрых локальных прогонов.
    sampled = False
    if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
        train_df = train_df.sample(n=args.max_train_rows, random_state=args.random_state).reset_index(drop=True)
        sampled = True

    x_train = train_df[[text_col, image_col]]
    y_train = train_df[target_col].astype(int).to_numpy()
    x_val = val_df[[text_col, image_col]]
    y_val = val_df[target_col].astype(int).to_numpy()

    # 3) Получаем SBERT и IMG эмбеддинги, затем L2-нормализация блоков.
    sbert = SentenceTransformer(args.sbert_model)
    x_text_train = sbert.encode(
        x_train[text_col].astype(str).tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    x_text_val = sbert.encode(
        x_val[text_col].astype(str).tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    x_img_train = image_features(x_train[image_col], images_dir)
    x_img_val = image_features(x_val[image_col], images_dir)

    norm = Normalizer(norm="l2")
    x_img_train = norm.fit_transform(x_img_train)
    x_img_val = norm.transform(x_img_val)
    x_text_train = norm.fit_transform(x_text_train)
    x_text_val = norm.transform(x_text_val)

    x_train_all = np.hstack([x_img_train, x_text_train]).astype(np.float32)
    x_val_all = np.hstack([x_img_val, x_text_val]).astype(np.float32)

    # 4) Обучаем MLP (512-128-1) с class_weight balanced для класса 1.
    torch.manual_seed(args.random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP(input_dim=x_train_all.shape[1]).to(device)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_ds = TensorDataset(torch.from_numpy(x_train_all), torch.from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for _ in range(args.epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    # 5) Считаем метрики на валидации.
    model.eval()
    with torch.inference_mode():
        logits_val = model(torch.from_numpy(x_val_all).to(device)).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_val))
    preds = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, probs)
    f1 = f1_score(y_val, preds, pos_label=1)
    precision = precision_score(y_val, preds, pos_label=1, zero_division=0)
    recall = recall_score(y_val, preds, pos_label=1, zero_division=0)

    # 6) Сохраняем модель и метрики.
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts_dir / "model.joblib"
    metrics_path = args.artifacts_dir / "metrics.json"
    joblib.dump(
        {
            "model_state_dict": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()},
            "model_input_dim": int(x_train_all.shape[1]),
            "sbert_model_name": args.sbert_model,
        },
        model_path,
    )

    metrics = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "TorchMLP(512-128-1) on SBERT+ResNet50",
        "target": target_col,
        "features": [text_col, image_col],
        "sbert_model": args.sbert_model,
        "feature_dim_text": int(x_text_train.shape[1]),
        "feature_dim_img": int(x_img_train.shape[1]),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "sampled_train": sampled,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[train] Saved model: {model_path}")
    print(f"[train] Saved metrics: {metrics_path}")
    print(f"[train] Validation ROC-AUC: {roc_auc:.4f}")
    print(f"[train] Validation F1: {f1:.4f}")
    print(f"[train] Validation Precision: {precision:.4f}")
    print(f"[train] Validation Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
