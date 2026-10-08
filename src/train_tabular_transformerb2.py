from pathlib import Path
import copy
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = MODEL_DIR / "transformer_data_experiment_B.joblib"
MODEL_PATH = MODEL_DIR / "tabular_transformer_experiment_B.pt"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 2
FFN_DIM = 64
DROPOUT = 0.20

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
PATIENCE = 12

GRADIENT_CLIP = 1.0


# ============================================================
# DATASET
# ============================================================

class TabularDataset(Dataset):

    def __init__(
        self,
        numerical,
        categorical,
        labels,
    ):
        self.numerical = torch.tensor(
            numerical,
            dtype=torch.float32,
        )

        self.categorical = torch.tensor(
            categorical,
            dtype=torch.long,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.numerical[index],
            self.categorical[index],
            self.labels[index],
        )


# ============================================================
# TABULAR TRANSFORMER
# ============================================================

class TabularTransformer(nn.Module):

    def __init__(
        self,
        num_numeric_features,
        category_cardinalities,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.num_numeric_features = (
            num_numeric_features
        )

        self.num_categorical_features = len(
            category_cardinalities
        )

        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Numerical feature embeddings
        # ----------------------------------------------------

        self.numeric_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(num_numeric_features)
            ]
        )

        # ----------------------------------------------------
        # Native categorical embeddings
        # ----------------------------------------------------

        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=embed_dim,
                )
                for cardinality in category_cardinalities
            ]
        )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------

        total_tokens = (
            1
            + num_numeric_features
            + self.num_categorical_features
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(
                1,
                total_tokens,
                embed_dim,
            )
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),

            nn.Linear(
                embed_dim,
                32,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                1,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.position_embeddings,
            mean=0.0,
            std=0.02,
        )

        for embedding_module in self.categorical_embeddings:

            if isinstance(embedding_module, nn.Embedding):
                nn.init.normal_(
                    embedding_module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        numerical,
        categorical,
    ):

        batch_size = numerical.size(0)

        tokens = []

        # ----------------------------------------------------
        # Numerical tokens
        # ----------------------------------------------------

        for i, projection in enumerate(
            self.numeric_embeddings
        ):

            value = numerical[:, i:i + 1]

            token = projection(value)

            tokens.append(
                token.unsqueeze(1)
            )

        # ----------------------------------------------------
        # Categorical tokens
        # ----------------------------------------------------

        for i, embedding in enumerate(
            self.categorical_embeddings
        ):

            value = categorical[:, i]

            token = embedding(value)

            tokens.append(
                token.unsqueeze(1)
            )

        # ----------------------------------------------------
        # Combine feature tokens
        # ----------------------------------------------------

        feature_tokens = torch.cat(
            tokens,
            dim=1,
        )

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [
                cls,
                feature_tokens,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Position information
        # ----------------------------------------------------

        x = x + self.position_embeddings[
            :, :x.size(1), :
        ]

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.transformer(x)

        # ----------------------------------------------------
        # CLS representation
        # ----------------------------------------------------

        cls_output = x[:, 0]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        logits = self.classifier(
            cls_output
        ).squeeze(1)

        return logits


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5,
):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probabilities = []

    for (
        numerical,
        categorical,
        labels,
    ) in loader:

        numerical = numerical.to(
            DEVICE,
            non_blocking=True,
        )

        categorical = categorical.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            numerical,
            categorical,
        )

        loss = criterion(
            logits,
            labels,
        )

        probabilities = torch.sigmoid(
            logits
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        all_labels.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    all_labels = np.concatenate(
        all_labels
    )

    all_probabilities = np.concatenate(
        all_probabilities
    )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities,
    )

    metrics["loss"] = average_loss

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics,
    split_name,
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Transformer - {split_name} Results"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"Loss     : {metrics['loss']:.4f}"
    )

    print(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print("=" * 70)
    print("TABULAR TRANSFORMER - EXPERIMENT B")
    print("Native Categorical Embeddings")
    print("=" * 70)

    print(
        f"\nPyTorch version: {torch.__version__}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: {torch.version.cuda}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Transformer data not found:\n"
            f"{DATA_PATH}\n\n"
            f"Run prepare_transformer_data.py first."
        )

    data = joblib.load(
        DATA_PATH
    )

    X_train_numeric = data[
        "X_train_numeric"
    ]

    X_val_numeric = data[
        "X_val_numeric"
    ]

    X_test_numeric = data[
        "X_test_numeric"
    ]

    X_train_categorical = data[
        "X_train_categorical"
    ]

    X_val_categorical = data[
        "X_val_categorical"
    ]

    X_test_categorical = data[
        "X_test_categorical"
    ]

    y_train = data[
        "y_train"
    ]

    y_val = data[
        "y_val"
    ]

    y_test = data[
        "y_test"
    ]

    category_cardinalities = [
        data["category_cardinalities"][column]
        for column in data["categorical_columns"]
    ]

    print("\nData:")

    print(
        f"Train numerical   : "
        f"{X_train_numeric.shape}"
    )

    print(
        f"Train categorical : "
        f"{X_train_categorical.shape}"
    )

    print(
        f"Validation        : "
        f"{X_val_numeric.shape}"
    )

    print(
        f"Test              : "
        f"{X_test_numeric.shape}"
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = TabularDataset(
        X_train_numeric,
        X_train_categorical,
        y_train,
    )

    val_dataset = TabularDataset(
        X_val_numeric,
        X_val_categorical,
        y_val,
    )

    test_dataset = TabularDataset(
        X_test_numeric,
        X_test_categorical,
        y_test,
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = TabularTransformer(
        num_numeric_features=len(
            data["numeric_columns"]
        ),
        category_cardinalities=category_cardinalities,
    )

    model = model.to(DEVICE)

    print("\nModel:")
    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Class weighting
    # --------------------------------------------------------

    positive_count = np.sum(
        y_train == 1
    )

    negative_count = np.sum(
        y_train == 0
    )

    positive_weight = (
        negative_count
        / positive_count
    )

    pos_weight = torch.tensor(
        positive_weight,
        dtype=torch.float32,
        device=DEVICE,
    )

    print(
        f"\nClass 0 count: {negative_count}"
    )

    print(
        f"Class 1 count: {positive_count}"
    )

    print(
        f"Positive class weight: "
        f"{positive_weight:.4f}"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nStarting training...")

    best_val_auc = -np.inf
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        for (
            numerical,
            categorical,
            labels,
        ) in train_loader:

            numerical = numerical.to(
                DEVICE,
                non_blocking=True,
            )

            categorical = categorical.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                numerical,
                categorical,
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(
            val_metrics["roc_auc"]
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            val_metrics["roc_auc"]
            > best_val_auc + 1e-5
        ):

            best_val_auc = (
                val_metrics["roc_auc"]
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No valid best model was recorded."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation ROC-AUC: "
        f"{best_val_auc:.4f}"
    )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    validation_metrics = evaluate(
        model,
        val_loader,
        criterion,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
    )

    print_metrics(
        validation_metrics,
        "Validation",
    )

    print_metrics(
        test_metrics,
        "Test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    checkpoint = {

        "model_state_dict":
            model.state_dict(),

        "model_config": {
            "num_numeric_features":
                len(data["numeric_columns"]),

            "category_cardinalities":
                category_cardinalities,

            "embed_dim":
                EMBED_DIM,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "ffn_dim":
                FFN_DIM,

            "dropout":
                DROPOUT,
        },

        "numeric_columns":
            data["numeric_columns"],

        "categorical_columns":
            data["categorical_columns"],

        "category_mappings":
            data["category_mappings"],

        "validation_metrics":
            validation_metrics,

        "test_metrics":
            test_metrics,

        "best_epoch":
            best_epoch,

        "best_validation_auc":
            best_val_auc,

        "seed":
            SEED,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TRANSFORMER TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved model:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()