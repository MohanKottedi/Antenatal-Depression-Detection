# Cell 6: Clean and deduplicate the dataset

import pandas as pd
import numpy as np

from pathlib import Path

dataset_path = Path(__file__).resolve().parent / "pakdataset.csv"
if not dataset_path.is_file():
    raise FileNotFoundError(
        f"Dataset not found at {dataset_path}. Place pakdataset.csv beside nrml.py."
    )
df = pd.read_csv(dataset_path)

# Preserve original data
df_clean = df.copy()

# 1. Remove accidental whitespace from column names
df_clean.columns = df_clean.columns.str.strip()

# 2. Remove leading/trailing whitespace from text values
string_cols = df_clean.select_dtypes(
    include=["object", "string"]
).columns

for col in string_cols:
    df_clean[col] = df_clean[col].str.strip()

# 3. Standardize known categorical values
cleanup_maps = {
    "Current Appereance Acceptance": {
        "YY": "Yes"
    }
}

for col, mapping in cleanup_maps.items():
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].replace(mapping)

# 4. Convert target labels to binary values
target_col = "Labelling"

target_mapping = {"Depressed": 1, "Not": 0}
unknown_labels = set(df_clean[target_col].dropna().unique()) - set(target_mapping)
if unknown_labels:
    raise ValueError(f"Unexpected target labels: {sorted(unknown_labels)}")
df_clean[target_col] = df_clean[target_col].map(target_mapping)

# Validate target
if df_clean[target_col].isna().any():
    raise ValueError("Missing target labels detected.")

df_clean[target_col] = df_clean[target_col].astype(int)

if not set(df_clean[target_col].unique()).issubset({0, 1}):
    raise ValueError("Unexpected target values.")

# 5. Remove exact duplicate records
rows_before = len(df_clean)

df_clean = df_clean.drop_duplicates().reset_index(drop=True)

rows_after = len(df_clean)

# 6. Verify results
print("===== CLEANED DATASET =====")
print("Rows before deduplication:", rows_before)
print("Rows after deduplication:", rows_after)
print("Rows removed:", rows_before - rows_after)

print("\nRemaining exact duplicates:",
      df_clean.duplicated().sum())

print("\nDataset shape:", df_clean.shape)

print("\nTarget distribution:")
print(df_clean[target_col].value_counts())

print("\nTarget proportions:")
print(
    df_clean[target_col]
    .value_counts(normalize=True)
    .round(4)
)

# Cell 7: Verify target generation rule

symptom_cols = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying sleep or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling badabout yourself that you are failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have Noticed.",
    "Thoughts that you would be better off dead, or of hurting yourself"
]

symptom_sum = df_clean[symptom_cols].sum(axis=1)

print("Scalling equals symptom sum:",
      (df_clean["Scalling"] == symptom_sum).all())

print("\nTarget rule verification:")

print(
    "All scores below 10 labelled 0:",
    (df_clean.loc[symptom_sum < 10, "Labelling"] == 0).all()
)

print(
    "All scores 10 or above labelled 1:",
    (df_clean.loc[symptom_sum >= 10, "Labelling"] == 1).all()
)

print("\nScore range:",
      symptom_sum.min(), "to", symptom_sum.max())

      # Cell 8: Define feature groups

target_col = "Labelling"

# Clinical symptom features (9 symptom questions)
symptom_cols = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying sleep or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling badabout yourself that you are failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have Noticed.",
    "Thoughts that you would be better off dead, or of hurting yourself"
]

# Demographic, socioeconomic, and obstetric risk factors
risk_factor_cols = [
    "Age",
    "Gestational Age",
    "Number of sons",
    "Number of daughters",
    "Total Number of Children",
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Physical Health",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
    "Current Appereance Acceptance",
    "Family System",
    "Male Gender Preference",
    "Relationship with Mother in-law"
]

# Exclude Scalling and Labelling from all input features
excluded_cols = ["Scalling", target_col]

# Experiment A: symptom-informed screening
features_experiment_A = risk_factor_cols + symptom_cols

# Experiment B: background risk factors only
features_experiment_B = risk_factor_cols

# Validate column names
for name, cols in {
    "Experiment A": features_experiment_A,
    "Experiment B": features_experiment_B
}.items():
    missing_cols = [c for c in cols if c not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"{name} has missing columns: {missing_cols}")

    if target_col in cols or "Scalling" in cols:
        raise ValueError(f"{name} contains a leakage column.")

print("Experiment A features:", len(features_experiment_A))
print("Experiment B features:", len(features_experiment_B))
print("Target:", target_col)
print("Excluded columns:", excluded_cols)

# Cell 9: Create leakage-free splits

from sklearn.model_selection import train_test_split

# Inputs and target
X_all = df_clean.drop(
    columns=["Labelling", "Scalling"]
)

y_all = df_clean["Labelling"].astype(int)

# First split: 70% train, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all,
    y_all,
    test_size=0.30,
    random_state=42,
    stratify=y_all
)

# Second split: divide temporary set equally
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("===== Split Shapes =====")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\n===== Target Distribution =====")

for name, y in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name}:")
    print(y.value_counts())
    print(y.value_counts(normalize=True).round(4))

    # Cell 10: Verify no exact record overlap

def get_record_hashes(X, y):
    combined = X.copy()
    combined["__target__"] = y.to_numpy()

    return set(
        pd.util.hash_pandas_object(
            combined,
            index=False
        ).values
    )

train_hashes = get_record_hashes(X_train, y_train)
val_hashes = get_record_hashes(X_val, y_val)
test_hashes = get_record_hashes(X_test, y_test)

print("Train-Val overlap:", len(train_hashes & val_hashes))
print("Train-Test overlap:", len(train_hashes & test_hashes))
print("Val-Test overlap:", len(val_hashes & test_hashes))

assert len(train_hashes & val_hashes) == 0
assert len(train_hashes & test_hashes) == 0
assert len(val_hashes & test_hashes) == 0

print("\nSUCCESS: No exact duplicate records across splits.")

# Cell 11: Create feature matrices for each experiment

X_train_A = X_train[features_experiment_A].copy()
X_val_A = X_val[features_experiment_A].copy()
X_test_A = X_test[features_experiment_A].copy()

X_train_B = X_train[features_experiment_B].copy()
X_val_B = X_val[features_experiment_B].copy()
X_test_B = X_test[features_experiment_B].copy()

print("===== Experiment A =====")
print("Train:", X_train_A.shape)
print("Validation:", X_val_A.shape)
print("Test:", X_test_A.shape)

print("\n===== Experiment B =====")
print("Train:", X_train_B.shape)
print("Validation:", X_val_B.shape)
print("Test:", X_test_B.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Identify numerical and categorical columns for each experiment
numeric_A = X_train_A.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_A = X_train_A.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_B = X_train_B.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_B = X_train_B.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Numerical preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

preprocessor_A = build_preprocessor(
    numeric_A, categorical_A
)

preprocessor_B = build_preprocessor(
    numeric_B, categorical_B
)

print("Experiment A numerical features:", len(numeric_A))
print("Experiment A categorical features:", len(categorical_A))

print("Experiment B numerical features:", len(numeric_B))
print("Experiment B categorical features:", len(categorical_B))

# Fit preprocessing only on training data
X_train_A_processed = preprocessor_A.fit_transform(X_train_A)
X_val_A_processed = preprocessor_A.transform(X_val_A)
X_test_A_processed = preprocessor_A.transform(X_test_A)

X_train_B_processed = preprocessor_B.fit_transform(X_train_B)
X_val_B_processed = preprocessor_B.transform(X_val_B)
X_test_B_processed = preprocessor_B.transform(X_test_B)

print("Experiment A processed shapes:")
print(X_train_A_processed.shape)
print(X_val_A_processed.shape)
print(X_test_A_processed.shape)

print("\nExperiment B processed shapes:")
print(X_train_B_processed.shape)
print(X_val_B_processed.shape)
print(X_test_B_processed.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import pandas as pd

# Random Forest — Experiment A
rf_A = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_A.fit(X_train_A_processed, y_train)

# Random Forest — Experiment B
rf_B = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_B.fit(X_train_B_processed, y_train)

print("Random Forest training completed.")

def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{'=' * 50}")
    print(model_name)
    print(f"{'=' * 50}")

    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y, y_pred,
        target_names=["Not Depressed", "Depressed"],
        zero_division=0
    ))


evaluate_model(
    rf_A,
    X_val_A_processed,
    y_val,
    "Random Forest — Experiment A (Risk Factors + Symptoms)"
)

evaluate_model(
    rf_B,
    X_val_B_processed,
    y_val,
    "Random Forest — Experiment B (Risk Factors Only)"
)

# Logistic regression is a useful, interpretable baseline for this
# symptom-defined target and often generalizes better than tree models here.
logistic_A = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
logistic_B = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
logistic_A.fit(X_train_A_processed, y_train)
logistic_B.fit(X_train_B_processed, y_train)
print("Logistic-regression baselines trained.")

evaluate_model(
    logistic_A,
    X_val_A_processed,
    y_val,
    "Logistic Regression — Experiment A (Risk Factors + Symptoms)"
)

evaluate_model(
    logistic_B,
    X_val_B_processed,
    y_val,
    "Logistic Regression — Experiment B (Risk Factors Only)"
)

from sklearn.svm import SVC

# SVM — Experiment A
svm_A = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_A.fit(X_train_A_processed, y_train)

# SVM — Experiment B
svm_B = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_B.fit(X_train_B_processed, y_train)

print("SVM training completed.")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))
print("Using device:", device)


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        d_token=32,
        n_heads=4,
        n_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.num_features = num_features
        self.d_token = d_token

        # Each input feature gets a learnable embedding
        self.feature_weight = nn.Parameter(
            torch.randn(num_features, d_token) * 0.02
        )
        self.feature_bias = nn.Parameter(
            torch.zeros(num_features, d_token)
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_token)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(d_token)

        self.head = nn.Sequential(
            nn.Linear(d_token, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x shape: (batch_size, num_features)
        x = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
        x = x + self.feature_bias.unsqueeze(0)

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)

        cls_output = self.norm(x[:, 0])
        return self.head(cls_output)


print("FT-Transformer class defined successfully.")

def to_dense_float32(values):
    """Convert sklearn dense or sparse output to finite float32 for PyTorch."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("FT-Transformer inputs must be a finite 2D matrix.")
    return values


# Preprocessors are fit on training data only; reuse their fitted transforms
# for validation and test to avoid preprocessing leakage.
X_train_A_ft = to_dense_float32(X_train_A_processed)
X_val_A_ft = to_dense_float32(X_val_A_processed)
X_test_A_ft = to_dense_float32(X_test_A_processed)
X_train_B_ft = to_dense_float32(X_train_B_processed)
X_val_B_ft = to_dense_float32(X_val_B_processed)
X_test_B_ft = to_dense_float32(X_test_B_processed)
y_train_ft = y_train.to_numpy(dtype=np.int64)
y_val_ft = y_val.to_numpy(dtype=np.int64)
y_test_ft = y_test.to_numpy(dtype=np.int64)

def make_loader(X, y, batch_size=128, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda")
    )


train_loader_A = make_loader(
    X_train_A_ft, y_train_ft, shuffle=True
)
val_loader_A = make_loader(
    X_val_A_ft, y_val_ft
)

train_loader_B = make_loader(
    X_train_B_ft, y_train_ft, shuffle=True
)
val_loader_B = make_loader(
    X_val_B_ft, y_val_ft
)

print("Experiment A training batches:", len(train_loader_A))
print("Experiment B training batches:", len(train_loader_B))

import copy

def train_ft_transformer(
    model,
    train_loader,
    val_loader,
    epochs=40,
    learning_rate=1e-3,
    patience=7
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += loss.item() * len(yb)
                val_count += len(yb)
                correct += (logits.argmax(dim=1) == yb).sum().item()

        val_loss = val_loss_sum / val_count
        val_accuracy = correct / val_count

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


print("Training function defined successfully.")

model_A = FTTransformer(
    num_features=X_train_A_ft.shape[1]
)

model_A, history_A = train_ft_transformer(
    model_A,
    train_loader_A,
    val_loader_A
)

print("\nExperiment A training complete.")

model_B = FTTransformer(
    num_features=X_train_B_ft.shape[1]
)

model_B, history_B = train_ft_transformer(
    model_B,
    train_loader_B,
    val_loader_B
)

print("\nExperiment B training complete.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import torch


def evaluate_ft_transformer(model, X, y, model_name):
    model.eval()

    X_tensor = torch.tensor(
        X, dtype=torch.float32
    ).to(device)

    with torch.inference_mode():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)

    y_pred = predictions.cpu().numpy()
    y_prob = probabilities.cpu().numpy()
    y_true = np.asarray(y)

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_true, y_pred):.4f}"
    )
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Not Depressed", "Depressed"],
        zero_division=0
    ))


evaluate_ft_transformer(
    model_A,
    X_val_A_ft,
    y_val_ft,
    "FT-Transformer — Experiment A"
)

evaluate_ft_transformer(
    model_B,
    X_val_B_ft,
    y_val_ft,
    "FT-Transformer — Experiment B"
)

# Final evaluation: held-out test set

print("\n" + "#" * 65)
print("FINAL TEST SET EVALUATION")
print("#" * 65)

# Random Forest
evaluate_model(
    rf_A,
    X_test_A_processed,
    y_test,
    "Random Forest — Experiment A — TEST"
)

evaluate_model(
    rf_B,
    X_test_B_processed,
    y_test,
    "Random Forest — Experiment B — TEST"
)

# SVM
evaluate_model(
    svm_A,
    X_test_A_processed,
    y_test,
    "SVM — Experiment A — TEST"
)

evaluate_model(
    svm_B,
    X_test_B_processed,
    y_test,
    "SVM — Experiment B — TEST"
)

# Logistic regression held-out comparison
evaluate_model(
    logistic_A,
    X_test_A_processed,
    y_test,
    "Logistic Regression — Experiment A — TEST"
)

evaluate_model(
    logistic_B,
    X_test_B_processed,
    y_test,
    "Logistic Regression — Experiment B — TEST"
)

# Experiment A includes the nine answers used to construct Labelling.
# This transparent scoring-rule check is not an independent prediction result.
test_symptom_sum = X_test_A[symptom_cols].sum(axis=1)
rule_predictions = (test_symptom_sum >= 10).astype(int)
print(
    "\nKnown label-rule reference (not independent model accuracy): "
    f"{accuracy_score(y_test, rule_predictions):.4f}"
)

# FT-Transformer
evaluate_ft_transformer(
    model_A,
    X_test_A_ft,
    y_test_ft,
    "FT-Transformer — Experiment A — TEST"
)

evaluate_ft_transformer(
    model_B,
    X_test_B_ft,
    y_test_ft,
    "FT-Transformer — Experiment B — TEST"
)

print("\n" + "=" * 72)
print("HELD-OUT TEST METRICS SUMMARY")
print("=" * 72)
print(
    f"{'Model':<22} {'Experiment':<12} "
    f"{'Accuracy':>10} {'Precision':>10} {'F1-score':>10}"
)

test_model_inputs = [
    ("Random Forest", "A", rf_A, X_test_A_processed),
    ("Random Forest", "B", rf_B, X_test_B_processed),
    ("SVM", "A", svm_A, X_test_A_processed),
    ("SVM", "B", svm_B, X_test_B_processed),
    ("Logistic Regression", "A", logistic_A, X_test_A_processed),
    ("Logistic Regression", "B", logistic_B, X_test_B_processed),
]

for model_name, experiment, model, X_test in test_model_inputs:
    y_pred = model.predict(X_test)
    print(
        f"{model_name:<22} {experiment:<12} "
        f"{accuracy_score(y_test, y_pred):>10.4f} "
        f"{precision_score(y_test, y_pred, zero_division=0):>10.4f} "
        f"{f1_score(y_test, y_pred, zero_division=0):>10.4f}"
    )

for model_name, experiment, model, X_test in [
    ("FT-Transformer", "A", model_A, X_test_A_ft),
    ("FT-Transformer", "B", model_B, X_test_B_ft),
]:
    model.eval()
    with torch.inference_mode():
        y_pred = model(
            torch.as_tensor(X_test, dtype=torch.float32, device=device)
        ).argmax(dim=1).cpu().numpy()
    print(
        f"{model_name:<22} {experiment:<12} "
        f"{accuracy_score(y_test_ft, y_pred):>10.4f} "
        f"{precision_score(y_test_ft, y_pred, zero_division=0):>10.4f} "
        f"{f1_score(y_test_ft, y_pred, zero_division=0):>10.4f}"
    )

# Show actual held-out predictions from both experiments. Experiment A uses
# the symptom answers that define Labelling, so interpret it as label-rule
# reproduction rather than independent clinical prediction.
def print_ft_predictions(model, X, y, experiment_name, count=10):
    model.eval()
    prediction_tensor = torch.as_tensor(
        X[:count],
        dtype=torch.float32,
        device=device
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(prediction_tensor), dim=1)[:, 1]
    predicted_labels = (probabilities >= 0.5).to(torch.int64).cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    print(f"\n{experiment_name} sample held-out predictions:")
    print("row\tactual\tpredicted\tP(Depressed)")
    for offset, (actual, predicted, probability) in enumerate(
        zip(y[:count], predicted_labels, probabilities)
    ):
        actual_name = "Depressed" if actual == 1 else "Not Depressed"
        predicted_name = "Depressed" if predicted == 1 else "Not Depressed"
        print(
            f"{offset}\t{actual_name}\t{predicted_name}\t{probability:.4f}"
        )


print_ft_predictions(
    model_A,
    X_test_A_ft,
    y_test_ft,
    "Experiment A (risk factors + symptoms)"
)
print_ft_predictions(
    model_B,
    X_test_B_ft,
    y_test_ft,
    "Experiment B (risk factors only)"
)
