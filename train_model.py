from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    FTTransformer,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = Path(
    os.environ.get("ANTENATAL_DATASET_PATH", BASE_DIR / "pakdataset.csv")
)
CHECKPOINT_PATH = Path(
    os.environ.get(
        "ANTENATAL_CHECKPOINT_PATH",
        BASE_DIR / "models" / "ft_transformer_antenatal_model.pkl",
    )
)
TARGET_COLUMN = "Labelling"
RANDOM_SEED = 42
CLASS_NAMES = ["Not Depressed", "Depressed"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Set ANTENATAL_DATASET_PATH "
            "or place pakdataset.csv beside train_model.py."
        )

    frame = pd.read_csv(path)
    missing_columns = [
        name for name in [*RAW_FEATURE_NAMES, TARGET_COLUMN] if name not in frame
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame.loc[frame[TARGET_COLUMN].notna()].copy()
    labels = frame[TARGET_COLUMN].map(normalize_label)
    label_map = {
        "not": 0,
        "not depressed": 0,
        "not-depressed": 0,
        "depressed": 1,
    }
    unknown_labels = sorted(set(labels) - set(label_map))
    if unknown_labels:
        raise ValueError(
            f"Unexpected values in {TARGET_COLUMN}: {unknown_labels}. "
            "Expected 'Not'/'Not Depressed' and 'Depressed'."
        )

    features = frame[RAW_FEATURE_NAMES].copy()
    for name in NUMERIC_FEATURE_NAMES:
        features[name] = pd.to_numeric(features[name], errors="coerce")
    for name in CATEGORICAL_FEATURE_NAMES:
        features[name] = features[name].map(normalize_category)
        features[name] = features[name].replace("", pd.NA)

    target = labels.map(label_map).to_numpy(dtype=np.int64)
    if np.unique(target).size != 2:
        raise ValueError("Training requires examples from both target classes.")
    return features, target


def normalize_category(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


def normalize_label(value: object) -> str:
    return str(value).strip().casefold()


def make_preprocessor() -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURE_NAMES),
            ("categorical", categorical, CATEGORICAL_FEATURE_NAMES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def as_dense_float32(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    dense = np.asarray(values, dtype=np.float32)
    if dense.ndim != 2 or not np.isfinite(dense).all():
        raise ValueError("Preprocessed features must be a finite 2D matrix.")
    return dense


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    _, _, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = np.unique(np.append(thresholds, 0.5))
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in candidates
    ]
    best_score = max(scores)
    best_threshold = min(
        (
            float(threshold)
            for threshold, score in zip(candidates, scores)
            if score == best_score
        ),
        key=lambda threshold: abs(threshold - 0.5),
    )
    return best_threshold


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def train_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_config: dict[str, int | float],
    device: torch.device,
) -> tuple[FTTransformer, int]:
    model = FTTransformer(model_config).to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_x = torch.from_numpy(x_val).to(device)
    validation_y = torch.from_numpy(y_val.astype(np.int64)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    max_epochs = int(os.environ.get("FT_MAX_EPOCHS", "100"))
    patience = int(os.environ.get("FT_EARLY_STOPPING_PATIENCE", "12"))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            val_loss = float(criterion(model(validation_x), validation_y).item())
        print(f"Epoch {epoch:03d}: validation_loss={val_loss:.5f}")

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    model.eval()
    return model, best_epoch


def predict_probabilities(
    model: FTTransformer,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).to(device))
        probabilities = torch.softmax(logits, dim=1)[:, 1]
    result = probabilities.cpu().numpy().astype(np.float64)
    if result.shape != (features.shape[0],) or not np.isfinite(result).all():
        raise ValueError("Model returned invalid class probabilities.")
    if ((result < 0.0) | (result > 1.0)).any():
        raise ValueError("Model probabilities must be between 0 and 1.")
    return result


def main() -> None:
    set_seed(RANDOM_SEED)
    features, target = load_dataset(DATASET_PATH)
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_val,
    )

    preprocessor = make_preprocessor()
    x_train_encoded = as_dense_float32(preprocessor.fit_transform(x_train))
    x_val_encoded = as_dense_float32(preprocessor.transform(x_val))
    x_test_encoded = as_dense_float32(preprocessor.transform(x_test))
    if x_train_encoded.shape[1] != x_val_encoded.shape[1] or (
        x_train_encoded.shape[1] != x_test_encoded.shape[1]
    ):
        raise ValueError("Train, validation, and test feature dimensions do not match.")

    model_config: dict[str, int | float] = {
        "input_dim": int(x_train_encoded.shape[1]),
        "embed_dim": 32,
        "num_heads": 4,
        "depth": 2,
        "dropout": 0.15,
        "num_classes": 2,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rows: {len(features)}; encoded input dimension: {model_config['input_dim']}")
    print(f"Label counts (0=Not Depressed, 1=Depressed): {np.bincount(target).tolist()}")
    print(f"Training device: {device}")
    print(
        "Features exclude PHQ symptom answers and Scalling to prevent target "
        "leakage; only the 16 configured antenatal features are used."
    )

    model, best_epoch = train_transformer(
        x_train_encoded,
        y_train,
        x_val_encoded,
        y_val,
        model_config,
        device,
    )
    val_probabilities = predict_probabilities(model, x_val_encoded, device)
    threshold = choose_f1_threshold(y_val, val_probabilities)
    test_probabilities = predict_probabilities(model, x_test_encoded, device)
    test_metrics = calculate_metrics(y_test, test_probabilities, threshold)

    baseline = LogisticRegression(max_iter=2000, class_weight="balanced")
    baseline.fit(x_train_encoded, y_train)
    baseline_val_probabilities = baseline.predict_proba(x_val_encoded)[:, 1]
    baseline_threshold = choose_f1_threshold(y_val, baseline_val_probabilities)
    baseline_probabilities = baseline.predict_proba(x_test_encoded)[:, 1]
    baseline_metrics = calculate_metrics(
        y_test,
        baseline_probabilities,
        baseline_threshold,
    )

    print(f"Validation-selected decision threshold (F1): {threshold:.4f}")
    print("Held-out FT-Transformer test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Logistic baseline validation-selected threshold: {baseline_threshold:.4f}")
    print("Held-out logistic baseline metrics:")
    for name, value in baseline_metrics.items():
        print(f"  {name}: {value:.4f}")

    checkpoint = {
        "checkpoint_schema_version": 1,
        "architecture": "ft_transformer_feature_tokenizer",
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "model_config": model_config,
        "preprocessor": preprocessor,
        "feature_names_raw": list(RAW_FEATURE_NAMES),
        "classes": list(CLASS_NAMES),
        "positive_class": "Depressed",
        "decision_threshold": threshold,
        "validation_best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "dataset_path": str(DATASET_PATH),
        "random_seed": RANDOM_SEED,
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=CHECKPOINT_PATH.parent,
        prefix=f".{CHECKPOINT_PATH.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(checkpoint, temporary_path)
        os.replace(temporary_path, CHECKPOINT_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Saved new FT-Transformer checkpoint: {CHECKPOINT_PATH}")
    print("The existing models/screening_pipeline.pkl was not modified.")


if __name__ == "__main__":
    main()
