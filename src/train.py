"""
=============================================================================
MLflow Experiment Tracking - Hotel Booking Cancellation Prediction
=============================================================================
Project     : MLflow Hotel Booking Cancellation (MLOps Assignment)
Author      : Vidal Deo
Date        : 29/07/2026
Python      : 3.13
Description : End-to-end ML pipeline with MLflow experiment tracking,
              model training, evaluation, visualization, and model registry.
=============================================================================
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import os
import sys
import warnings
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Scikit-Learn – Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Scikit-Learn – Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Scikit-Learn – Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Force UTF-8 output on Windows terminals (default cp1252 cannot render
# unicode characters used by some table/chart libraries).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: CONSTANTS & PATHS
# ===========================================================================

# Resolve the project root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
ARTIFACTS_DIR   = PROJECT_ROOT / "artifacts"

# Dataset and MLflow settings
DATASET_FILENAME    = "hotel_bookings.csv"
TARGET_COLUMN       = "is_canceled"
EXPERIMENT_NAME     = "Hotel Booking Cancellation Experiment"
TEST_SIZE           = 0.20
RANDOM_STATE        = 42

# Columns known to be high-cardinality or leakage-prone – dropped before training
DROP_COLUMNS = [
    "reservation_status",          # Direct label leakage
    "reservation_status_date",     # Date of cancellation (leakage)
]


# ===========================================================================
# SECTION 2: UTILITY HELPERS
# ===========================================================================

def print_section(title: str) -> None:
    """Print a clearly visible section divider in the console."""
    border = "=" * 65
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")





def ensure_directories() -> None:
    """Create all required project directories if they don't already exist."""
    for directory in [MODELS_DIR, ARTIFACTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Project directories verified / created.")


# ===========================================================================
# SECTION 3: DATA LOADING
# ===========================================================================

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Load the hotel bookings CSV dataset with robust error handling.

    Parameters
    ----------
    filepath : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a Pandas DataFrame.
    """
    print_section("STEP 1: LOADING DATASET")

    # Validate file existence
    if not filepath.exists():
        logger.error(f"Dataset not found at: {filepath}")
        logger.error("Please place 'hotel_bookings.csv' inside the 'data/' folder.")
        sys.exit(1)

    # Load CSV with encoding fallback
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Retrying with 'latin-1' encoding.")
        df = pd.read_csv(filepath, encoding="latin-1")
    except Exception as exc:
        logger.error(f"Failed to read CSV: {exc}")
        sys.exit(1)

    # Validate dataset is non-empty
    if df.empty:
        logger.error("The dataset is empty. Please check the file.")
        sys.exit(1)

    if TARGET_COLUMN not in df.columns:
        logger.error(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
        sys.exit(1)

    # Console summary
    print(f"\n  Dataset loaded successfully from: {filepath}")
    print(f"\n  Dataset Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Target Column     : {TARGET_COLUMN}")
    print(f"\n  First 5 Rows:\n")
    print(df.head().to_string())
    print(f"\n  Dataset Info:")
    print(f"    Columns     : {df.shape[1]}")
    print(f"    Rows        : {df.shape[0]}")
    print(f"    Memory Usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"\n  Column Data Types:")
    dtypes_table = [[col, str(dtype)] for col, dtype in df.dtypes.items()]
    print(tabulate(dtypes_table, headers=["Column", "dtype"], tablefmt="simple"))

    return df


# ===========================================================================
# SECTION 4: DATA PREPROCESSING
# ===========================================================================

def analyse_missing_values(df: pd.DataFrame) -> None:
    """
    Display a detailed missing-value and duplicate analysis before cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame to analyse.
    """
    print_section("STEP 2: MISSING VALUES & DATA QUALITY ANALYSIS")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
        "Missing Count", ascending=False
    )

    print(f"\n  Total Duplicates   : {df.duplicated().sum()}")
    print(f"  Total Missing Vals : {df.isnull().sum().sum()}")

    if missing_df.empty:
        print("\n  No missing values found in the dataset.")
    else:
        print(f"\n  Columns with Missing Values ({len(missing_df)}):\n")
        print(tabulate(missing_df.reset_index(), headers=["Column", "Missing Count", "Missing %"],
                       tablefmt="simple"))

    # Data-type breakdown
    print(f"\n  Numeric  Columns : {len(df.select_dtypes(include=['number']).columns)}")
    print(f"  Object   Columns : {len(df.select_dtypes(include=['object']).columns)}")


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Perform full preprocessing pipeline:
      1. Drop leakage / irrelevant columns
      2. Remove duplicate rows
      3. Handle missing values (median for numeric, mode for categorical)
      4. Label-encode binary categorical columns
      5. One-Hot-Encode multi-class categorical columns
      6. Standard-scale numeric features

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.

    Returns
    -------
    tuple : (X_scaled, y, feature_names)
        X_scaled     : pd.DataFrame - fully preprocessed feature matrix
        y            : pd.Series    - target labels
        feature_names: list[str]    - final feature column names
    """
    print_section("STEP 3: PREPROCESSING")

    df = df.copy()

    # 4.1  Drop leakage / irrelevant columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"Dropped leakage/irrelevant columns: {cols_to_drop}")

    # 4.2  Separate target
    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])

    # 4.3  Remove duplicates
    initial_rows = len(X)
    mask = X.duplicated()
    X = X[~mask]
    y = y[~mask]
    dropped_dupes = initial_rows - len(X)
    print(f"\n  Duplicate rows removed : {dropped_dupes}")
    print(f"  Rows after dedup       : {len(X)}")

    # 4.4  Handle missing values
    numeric_cols      = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols  = X.select_dtypes(include=["object"]).columns.tolist()

    # Numeric -> fill with median
    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info(f"  Filled '{col}' NaN with median={median_val:.4f}")

    # Categorical -> fill with mode
    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
            logger.info(f"  Filled '{col}' NaN with mode='{mode_val}'")

    print(f"  Missing values after imputation : {X.isnull().sum().sum()}")

    # 4.5  Encode categorical features
    #
    # Strategy:
    #   - Binary columns (<= 2 unique non-null values) -> LabelEncoder
    #   - Low-cardinality columns (3-10 unique)        -> One-Hot Encoding
    #   - High-cardinality columns (> 10 unique)       -> LabelEncoder

    label_enc_cols   = []
    onehot_enc_cols  = []

    for col in categorical_cols:
        n_unique = X[col].nunique()
        if n_unique <= 2:
            label_enc_cols.append(col)
        elif n_unique <= 10:
            onehot_enc_cols.append(col)
        else:
            label_enc_cols.append(col)     # High-cardinality -> label encode

    print(f"\n  Label-Encoded columns  : {label_enc_cols}")
    print(f"  One-Hot-Encoded columns: {onehot_enc_cols}")

    # Apply Label Encoding
    le = LabelEncoder()
    for col in label_enc_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # Apply One-Hot Encoding
    if onehot_enc_cols:
        X = pd.get_dummies(X, columns=onehot_enc_cols, drop_first=True)

    # Convert all boolean dummies to int
    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    print(f"\n  Encoding completed")
    print(f"  Dataset shape after encoding: {X.shape}")

    # 4.6  Feature scaling (Standard Scaler on numeric features)
    numeric_cols_final = X.select_dtypes(include=["number"]).columns.tolist()

    scaler = StandardScaler()
    X[numeric_cols_final] = scaler.fit_transform(X[numeric_cols_final])

    print(f"  Standard Scaling applied to {len(numeric_cols_final)} numeric columns")

    # 4.7  Reset index
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    print(f"\n  Preprocessing complete!")
    print(f"  Final Feature Matrix Shape : {X.shape}")
    print(f"  Target Distribution:")
    print(y.value_counts().to_string())

    feature_names = X.columns.tolist()
    return X, y, feature_names


# ===========================================================================
# SECTION 5: TRAIN-TEST SPLIT
# ===========================================================================

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Split preprocessed data into 80% training and 20% testing sets.
    Stratified split ensures class balance is preserved.

    Parameters
    ----------
    X : pd.DataFrame - Feature matrix.
    y : pd.Series    - Target labels.

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
    """
    print_section("STEP 4: TRAIN / TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y             # Preserve class distribution
    )

    print(f"\n  Split Ratio  : 80% Train / 20% Test")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test  shape: {y_test.shape}")
    print(f"\n  Train class distribution:")
    print(y_train.value_counts().to_string())
    print(f"\n  Test  class distribution:")
    print(y_test.value_counts().to_string())

    return X_train, X_test, y_train, y_test


# ===========================================================================
# SECTION 6: EVALUATION HELPERS
# ===========================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model and return all required metrics.

    Parameters
    ----------
    model  : fitted sklearn estimator
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    metrics : dict containing all evaluation metrics
    """
    y_pred   = model.predict(X_test)
    y_proba  = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "accuracy"          : round(accuracy_score(y_test, y_pred),                       4),
        "precision"         : round(precision_score(y_test, y_pred, zero_division=0),      4),
        "recall"            : round(recall_score(y_test, y_pred,    zero_division=0),      4),
        "f1_score"          : round(f1_score(y_test, y_pred,        zero_division=0),      4),
        "roc_auc"           : round(roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0, 4),
        "confusion_matrix"  : confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_pred"            : y_pred,
        "y_proba"           : y_proba,
    }
    return metrics


# ===========================================================================
# SECTION 7: VISUALIZATION HELPERS
# ===========================================================================

def log_confusion_matrix(cm: np.ndarray, run_name: str, class_labels=None) -> Path:
    """
    Save and return the path of a confusion matrix plot.

    Parameters
    ----------
    cm          : np.ndarray  - Confusion matrix array from sklearn.
    run_name    : str         - Used as part of the saved filename.
    class_labels: list[str]  - Optional class label names.

    Returns
    -------
    filepath : Path - Path to the saved PNG image.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels or ["Not Canceled (0)", "Canceled (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix - {run_name}", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    safe_name = run_name.replace(" ", "_")
    filepath  = ARTIFACTS_DIR / f"confusion_matrix_{safe_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {filepath}")
    return filepath


def log_roc_curve(y_test: pd.Series, y_proba: np.ndarray, run_name: str) -> Path:
    """
    Save and return the path of a ROC curve plot.

    Parameters
    ----------
    y_test  : pd.Series   - True labels.
    y_proba : np.ndarray  - Predicted probabilities for the positive class.
    run_name: str         - Model name used in filename.

    Returns
    -------
    filepath : Path - Path to the saved PNG image.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2E86AB", lw=2,
            label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#E84855", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2E86AB")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {run_name}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = run_name.replace(" ", "_")
    filepath  = ARTIFACTS_DIR / f"roc_curve_{safe_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"ROC curve saved: {filepath}")
    return filepath


def log_feature_importance(model, feature_names: list, run_name: str, top_n: int = 20):
    """
    Save a feature-importance bar chart for tree-based models.
    Returns None if the model does not expose feature importances.

    Parameters
    ----------
    model        : fitted sklearn estimator
    feature_names: list[str]  - Names of features.
    run_name     : str        - Model name used in filename.
    top_n        : int        - How many top features to display.

    Returns
    -------
    filepath : Path | None
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    ax.barh(range(len(top_features)), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {run_name}",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    safe_name = run_name.replace(" ", "_")
    filepath  = ARTIFACTS_DIR / f"feature_importance_{safe_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Feature importance chart saved: {filepath}")
    return filepath


def log_classification_report(report_str: str, run_name: str) -> Path:
    """
    Save classification report as a plain-text artifact.

    Parameters
    ----------
    report_str : str  - Output of sklearn classification_report().
    run_name   : str  - Model name used in filename.

    Returns
    -------
    filepath : Path
    """
    safe_name = run_name.replace(" ", "_")
    filepath  = ARTIFACTS_DIR / f"classification_report_{safe_name}.txt"
    filepath.write_text(
        f"Classification Report - {run_name}\n"
        + "=" * 55 + "\n"
        + report_str,
        encoding="utf-8"
    )
    logger.info(f"Classification report saved: {filepath}")
    return filepath


# ===========================================================================
# SECTION 8: MLFLOW LOGGING CORE
# ===========================================================================

def _log_run(
    run_name: str,
    model,
    params: dict,
    metrics: dict,
    X_train: pd.DataFrame,
    feature_names: list,
    y_test: pd.Series,
) -> str:
    """
    Internal helper: open an MLflow run, log everything, and close.

    Parameters
    ----------
    run_name     : str          - Name of the MLflow run.
    model        : estimator    - Trained sklearn model.
    params       : dict         - Hyperparameters to log.
    metrics      : dict         - Evaluation metrics dict from evaluate_model().
    X_train      : pd.DataFrame - Training features (for model signature).
    feature_names: list[str]    - Feature column names.
    y_test       : pd.Series    - True test labels.

    Returns
    -------
    run_id : str  - The MLflow run ID for later reference.
    """
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_params(params)

        # Log scalar metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])
        mlflow.log_metric("roc_auc",   metrics["roc_auc"])

        # Save & log confusion matrix
        cm_path = log_confusion_matrix(metrics["confusion_matrix"], run_name)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Save & log ROC curve
        if metrics["y_proba"] is not None:
            roc_path = log_roc_curve(y_test, metrics["y_proba"], run_name)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        # Save & log feature importance (tree models only)
        fi_path = log_feature_importance(model, feature_names, run_name)
        if fi_path:
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # Save & log classification report
        report_path = log_classification_report(
            metrics["classification_report"], run_name
        )
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # Log trained model with input/output signature
        # Note: MLflow 3.14+ uses 'name' instead of deprecated 'artifact_path'
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name=None,   # Register best model separately
        )

        logger.info(f"  MLflow Run ID : {run_id}")

    return run_id


# ===========================================================================
# SECTION 9: MODEL TRAINING FUNCTIONS
# ===========================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
) -> tuple:
    """
    Train Logistic Regression, evaluate, log to MLflow, and return results.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : testing data
    feature_names    : list of feature column names

    Returns
    -------
    tuple : (model, metrics, run_id)
    """
    print_section("STEP 5a: TRAINING - LOGISTIC REGRESSION")

    # Hyperparameters
    params = {
        "model_type"   : "LogisticRegression",
        "C"            : 1.0,
        "max_iter"     : 1000,
        "solver"       : "lbfgs",
        "random_state" : RANDOM_STATE,
        "class_weight" : "balanced",
    }

    # Train
    logger.info("Training Logistic Regression ...")
    model = LogisticRegression(
        C=params["C"],
        max_iter=params["max_iter"],
        solver=params["solver"],
        random_state=params["random_state"],
        class_weight=params["class_weight"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training completed.")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Console output
    print(f"\n  Logistic Regression Training Completed")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    # MLflow logging
    run_id = _log_run(
        run_name="Logistic Regression",
        model=model,
        params=params,
        metrics=metrics,
        X_train=X_train,
        feature_names=feature_names,
        y_test=y_test,
    )
    print(f"  MLflow Run ID : {run_id}")

    return model, metrics, run_id


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
) -> tuple:
    """
    Train Decision Tree Classifier, evaluate, log to MLflow, and return results.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : testing data
    feature_names    : list of feature column names

    Returns
    -------
    tuple : (model, metrics, run_id)
    """
    print_section("STEP 5b: TRAINING - DECISION TREE CLASSIFIER")

    # Hyperparameters
    params = {
        "model_type"       : "DecisionTreeClassifier",
        "max_depth"        : 10,
        "min_samples_split": 20,
        "min_samples_leaf" : 10,
        "criterion"        : "gini",
        "random_state"     : RANDOM_STATE,
        "class_weight"     : "balanced",
    }

    # Train
    logger.info("Training Decision Tree Classifier ...")
    model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"],
        random_state=params["random_state"],
        class_weight=params["class_weight"],
    )
    model.fit(X_train, y_train)
    logger.info("Decision Tree training completed.")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Console output
    print(f"\n  Decision Tree Training Completed")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    # MLflow logging
    run_id = _log_run(
        run_name="Decision Tree",
        model=model,
        params=params,
        metrics=metrics,
        X_train=X_train,
        feature_names=feature_names,
        y_test=y_test,
    )
    print(f"  MLflow Run ID : {run_id}")

    return model, metrics, run_id


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
) -> tuple:
    """
    Train Random Forest Classifier, evaluate, log to MLflow, and return results.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : testing data
    feature_names    : list of feature column names

    Returns
    -------
    tuple : (model, metrics, run_id)
    """
    print_section("STEP 5c: TRAINING - RANDOM FOREST CLASSIFIER")

    # Hyperparameters
    params = {
        "model_type"       : "RandomForestClassifier",
        "n_estimators"     : 100,
        "max_depth"        : 15,
        "min_samples_split": 10,
        "min_samples_leaf" : 5,
        "max_features"     : "sqrt",
        "criterion"        : "gini",
        "random_state"     : RANDOM_STATE,
        "class_weight"     : "balanced",
    }

    # Train
    logger.info("Training Random Forest Classifier (this may take ~1 min) ...")
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        criterion=params["criterion"],
        random_state=params["random_state"],
        class_weight=params["class_weight"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training completed.")

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Console output
    print(f"\n  Random Forest Training Completed")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    # MLflow logging
    run_id = _log_run(
        run_name="Random Forest",
        model=model,
        params=params,
        metrics=metrics,
        X_train=X_train,
        feature_names=feature_names,
        y_test=y_test,
    )
    print(f"  MLflow Run ID : {run_id}")

    return model, metrics, run_id


# ===========================================================================
# SECTION 10: MODEL COMPARISON
# ===========================================================================

def compare_models(results: list) -> dict:
    """
    Compare all trained models and print a summary table.
    Best model is selected by highest Accuracy, F1 Score as tiebreaker.

    Parameters
    ----------
    results : list[dict]
        Each dict contains keys: 'name', 'model', 'metrics', 'run_id'.

    Returns
    -------
    best : dict - The entry for the best-performing model.
    """
    print_section("STEP 6: MODEL COMPARISON")

    # Build summary table
    table_data = []
    for r in results:
        m = r["metrics"]
        table_data.append([
            r["name"],
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            f"{m['roc_auc']:.4f}",
        ])

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    print(f"\n{tabulate(table_data, headers=headers, tablefmt='rounded_outline')}")

    # Determine best model by Accuracy then F1 Score
    best = max(results, key=lambda r: (r["metrics"]["accuracy"], r["metrics"]["f1_score"]))

    print(f"\n  Best Model      : {best['name']}")
    print(f"  Best Accuracy   : {best['metrics']['accuracy']:.4f}")
    print(f"  Best F1 Score   : {best['metrics']['f1_score']:.4f}")
    print(f"  Best ROC-AUC    : {best['metrics']['roc_auc']:.4f}")
    print(f"  MLflow Run ID   : {best['run_id']}")

    # Save a comparison chart
    _save_comparison_chart(results)

    return best


def _save_comparison_chart(results: list) -> None:
    """
    Save a grouped bar chart comparing all model metrics.
    """
    model_names = [r["name"] for r in results]
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels   = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

    x       = np.arange(len(model_names))
    width   = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(metrics_to_plot))
    colors  = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors)):
        values = [r["metrics"][metric] for r in results]
        bars   = ax.bar(x + offsets[i], values, width, label=label, color=color, alpha=0.85)
        # Add value labels on bars
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold"
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison - All Metrics", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = ARTIFACTS_DIR / "model_comparison.png"
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison chart saved: {filepath}")


# ===========================================================================
# SECTION 11: REGISTER BEST MODEL IN MLFLOW MODEL REGISTRY
# ===========================================================================

def register_best_model(best: dict) -> None:
    """
    Register the best model in the MLflow Model Registry.

    The model is registered under 'HotelBookingCancellation_BestModel'.
    If the registry is unavailable, the model is saved locally as a fallback.

    Parameters
    ----------
    best : dict  - Best model dict with keys 'name', 'model', 'run_id'.
    """
    print_section("STEP 7: REGISTERING BEST MODEL IN MLFLOW MODEL REGISTRY")

    registry_name = "HotelBookingCancellation_BestModel"
    model_uri     = f"runs:/{best['run_id']}/model"

    try:
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=registry_name,
        )
        print(f"\n  Model registered successfully!")
        print(f"  Registry Name : {registry_name}")
        print(f"  Model URI     : {model_uri}")
        print(f"  Version       : {registered.version}")
        logger.info(f"Best model '{best['name']}' registered as '{registry_name}'")

    except Exception as exc:
        # Fallback: save model to local models/ directory
        logger.warning(f"MLflow Model Registry unavailable ({exc}). Saving locally.")
        save_best_model(best)


def save_best_model(best: dict) -> None:
    """
    Fallback: Save the best model as a joblib file to the models/ directory.

    Parameters
    ----------
    best : dict  - Best model dict with keys 'name', 'model'.
    """
    import joblib

    safe_name = best["name"].replace(" ", "_").lower()
    filepath  = MODELS_DIR / f"best_model_{safe_name}.joblib"
    joblib.dump(best["model"], filepath)
    print(f"\n  Best model saved locally: {filepath}")
    logger.info(f"Best model saved to: {filepath}")


# ===========================================================================
# SECTION 12: FINAL SUMMARY
# ===========================================================================

def print_final_summary(results: list, best: dict) -> None:
    """
    Print a final, clearly formatted summary of the entire experiment.

    Parameters
    ----------
    results : list[dict]  - All model results.
    best    : dict        - Best model entry.
    """
    print_section("FINAL EXPERIMENT SUMMARY")

    print(f"""
  Experiment  : {EXPERIMENT_NAME}
  Dataset     : {DATASET_FILENAME}
  Target      : {TARGET_COLUMN}
  Split       : 80% Train / 20% Test

  Models Trained:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. Random Forest Classifier
    """)

    for r in results:
        m = r["metrics"]
        print(f"  [{r['name']}]")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1 Score  : {m['f1_score']:.4f}")
        print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"    Run ID    : {r['run_id']}\n")

    bm = best["metrics"]
    print("=" * 65)
    print(f"  BEST MODEL      : {best['name']}")
    print(f"  Best Accuracy   : {bm['accuracy']:.4f}")
    print(f"  Best Precision  : {bm['precision']:.4f}")
    print(f"  Best Recall     : {bm['recall']:.4f}")
    print(f"  Best F1 Score   : {bm['f1_score']:.4f}")
    print(f"  Best ROC-AUC    : {bm['roc_auc']:.4f}")
    print(f"  MLflow Run ID   : {best['run_id']}")
    print("=" * 65)

    print(f"\n  To view MLflow UI, run:")
    print(f"    mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"  Then open: http://127.0.0.1:5000\n")


# ===========================================================================
# SECTION 13: MAIN ENTRY POINT
# ===========================================================================

def main() -> None:
    """
    Orchestrates the end-to-end ML pipeline:
      1.  Load data
      2.  Analyse missing values
      3.  Preprocess data
      4.  Split data
      5.  Set up MLflow experiment
      6.  Train three models
      7.  Compare models
      8.  Register best model
      9.  Print final summary
    """
    print_section("MLflow Hotel Booking Cancellation - MLOps Assignment")
    print("  Training Started ...\n")

    # Directory setup
    ensure_directories()

    # Step 1: Load data
    dataset_path = DATA_DIR / DATASET_FILENAME
    df = load_data(dataset_path)



    # Step 2: Analyse missing values
    analyse_missing_values(df)



    # Step 3: Preprocess
    X, y, feature_names = preprocess_data(df)



    # Step 4: Split
    X_train, X_test, y_train, y_test = split_data(X, y)



    # Step 5: Configure MLflow experiment
    print_section("MLFLOW CONFIGURATION")

    # Set tracking URI to SQLite database backend
    # MLflow 3.14+ deprecated the file-based mlruns/ store; SQLite is the
    # recommended lightweight replacement for local development.
    db_path = (PROJECT_ROOT / 'mlflow.db').as_posix()
    mlflow_tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\n  Experiment Name : {EXPERIMENT_NAME}")
    print(f"  Experiment ID   : {experiment.experiment_id}")
    print(f"  Tracking URI    : {mlflow_tracking_uri}")
    print(f"  Database File   : {PROJECT_ROOT / 'mlflow.db'}")

    # Step 6: Train models
    results = []

    # 6a. Logistic Regression
    lr_model, lr_metrics, lr_run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test, feature_names
    )
    results.append({
        "name": "Logistic Regression",
        "model": lr_model,
        "metrics": lr_metrics,
        "run_id": lr_run_id,
    })



    # 6b. Decision Tree
    dt_model, dt_metrics, dt_run_id = train_decision_tree(
        X_train, y_train, X_test, y_test, feature_names
    )
    results.append({
        "name": "Decision Tree",
        "model": dt_model,
        "metrics": dt_metrics,
        "run_id": dt_run_id,
    })



    # 6c. Random Forest
    rf_model, rf_metrics, rf_run_id = train_random_forest(
        X_train, y_train, X_test, y_test, feature_names
    )
    results.append({
        "name": "Random Forest",
        "model": rf_model,
        "metrics": rf_metrics,
        "run_id": rf_run_id,
    })



    # Step 7: Compare models
    best = compare_models(results)



    # Print best model artifact paths
    print_section("BEST MODEL ARTIFACTS")
    print(f"\n  Best Model          : {best['name']}")
    best_cm_path  = ARTIFACTS_DIR / f"confusion_matrix_{best['name'].replace(' ', '_')}.png"
    best_roc_path = ARTIFACTS_DIR / f"roc_curve_{best['name'].replace(' ', '_')}.png"
    print(f"  Confusion Matrix    : {best_cm_path}")
    print(f"  ROC Curve           : {best_roc_path}")



    # Step 8: Register best model
    register_best_model(best)



    # Step 9: Final summary
    print_final_summary(results, best)

    print("\n  All training, evaluation, and MLflow logging completed.")
    print("  Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' inside the MLflow_Assignment/ folder to view results.\n")




# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    main()
