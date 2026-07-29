# MLflow Experiment Tracking — Hotel Booking Cancellation Prediction

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.14+-orange.svg)](https://mlflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## Project Overview

This project demonstrates a **production-quality Machine Learning pipeline** built for a college **MLOps Assignment**. It uses the **Hotel Booking Cancellation** dataset to predict whether a hotel booking will be cancelled (`is_canceled`).

The project covers:

- Automated data loading and validation
- Full preprocessing pipeline (missing values, encoding, scaling)
- Training three ML models: Logistic Regression, Decision Tree, Random Forest
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Complete MLflow experiment tracking (parameters, metrics, artifacts, models)
- Automatic best-model selection and registration in MLflow Model Registry
- Professional visualizations (Confusion Matrix, ROC Curve, Feature Importance)

---

## Dataset

| Property        | Value                        |
|-----------------|------------------------------|
| **Name**        | Hotel Bookings               |
| **File**        | `hotel_bookings.csv`         |
| **Rows**        | ~119,000                     |
| **Features**    | 32 columns                   |
| **Target**      | `is_canceled` (0 = Not Cancelled, 1 = Cancelled) |
| **Source**      | Kaggle Hotel Booking Demand  |

---

## Project Structure

```
MLflow_Assignment/
│
├── data/
│   └── hotel_bookings.csv          ← Place your dataset here
│
├── src/
│   └── train.py                    ← Main training script (entry point)
│
├── models/
│   └── best_model_*.joblib         ← Saved best model (fallback)
│
├── artifacts/
│   ├── confusion_matrix_*.png      ← Confusion matrix plots
│   ├── roc_curve_*.png             ← ROC curve plots
│   ├── feature_importance_*.png    ← Feature importance plots
│   ├── classification_report_*.txt ← Classification reports
│   └── model_comparison.png        ← All-model comparison chart
│
├── screenshots/
│   └── *.png                       ← Assignment screenshots
│
├── mlruns/
│   └── ...                         ← MLflow tracking data (auto-generated)
│
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## Installation

### Prerequisites

- Python 3.13 installed
- VS Code (recommended IDE)
- Windows OS

### Step 1 — Clone / Download the Project

Place the project folder at your desired location.

### Step 2 — Create Virtual Environment

Open a terminal inside `MLflow_Assignment/` and run:

```powershell
python -m venv .venv
```

### Step 3 — Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

> **Screenshot 2**: Take a screenshot here showing the activated virtual environment.

### Step 4 — Install Dependencies

```powershell
pip install -r requirements.txt
```

> **Screenshot 3**: Take a screenshot showing all libraries successfully installed.

### Step 5 — Add Dataset

Copy `hotel_bookings.csv` into the `data/` folder:

```
MLflow_Assignment/data/hotel_bookings.csv
```

---

## How to Run

Make sure your virtual environment is activated, then run:

```powershell
cd MLflow_Assignment
python src/train.py
```

The script will:

1. Load and validate the dataset
2. Analyse missing values and data quality
3. Preprocess (encode, scale, impute)
4. Split into 80/20 train/test
5. Train Logistic Regression, Decision Tree, Random Forest
6. Evaluate all models and log to MLflow
7. Compare models and select the best
8. Register the best model in MLflow Model Registry
9. Save all plots to `artifacts/`

> The script pauses at key milestones and prompts you to take a screenshot before continuing.

---

## MLflow Commands

### Launch MLflow UI

```powershell
# Run inside MLflow_Assignment/ folder
mlflow ui --backend-store-uri mlruns/
```

Then open your browser at: **http://127.0.0.1:5000**

### View Registered Models

In the MLflow UI, click the **"Models"** tab in the top navigation bar.

### List Experiments (CLI)

```powershell
mlflow experiments list
```

### Search Runs (CLI)

```powershell
mlflow runs list --experiment-name "Hotel Booking Cancellation Experiment"
```

---

## Models Trained

| Model                | Description                                        |
|----------------------|----------------------------------------------------|
| Logistic Regression  | Linear model with `C=1.0`, `lbfgs` solver         |
| Decision Tree        | `max_depth=10`, Gini criterion                     |
| Random Forest        | 100 estimators, `max_depth=15`, balanced weights   |

---

## Evaluation Metrics

Each model is evaluated on the **20% test set** using:

| Metric           | Description                                        |
|------------------|----------------------------------------------------|
| Accuracy         | Overall correct predictions / total predictions    |
| Precision        | True Positives / (True Positives + False Positives)|
| Recall           | True Positives / (True Positives + False Negatives)|
| F1 Score         | Harmonic mean of Precision and Recall              |
| ROC-AUC          | Area Under the ROC Curve                           |
| Confusion Matrix | Matrix of TP, TN, FP, FN                          |

---

## MLflow Experiment Tracking

Every model run logs:

- **Parameters**: All hyperparameters used during training
- **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Artifacts**:
  - `plots/confusion_matrix_*.png`
  - `plots/roc_curve_*.png`
  - `plots/feature_importance_*.png` (tree models)
  - `reports/classification_report_*.txt`
  - `model/` (serialized sklearn model with signature)

---

## Results (Expected)

> Actual results may vary slightly based on the dataset version.

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.79    | ~0.76     | ~0.72  | ~0.74    | ~0.87   |
| Decision Tree       | ~0.84    | ~0.82     | ~0.80  | ~0.81    | ~0.89   |
| **Random Forest**   | **~0.88**| **~0.86** |**~0.84**|**~0.85**|**~0.94**|

> **Best Model**: Random Forest (highest Accuracy and F1 Score)

---

## Preprocessing Pipeline

| Step                      | Action                                             |
|---------------------------|----------------------------------------------------|
| Drop Leakage Columns      | `reservation_status`, `reservation_status_date`   |
| Remove Duplicates         | Drop exact duplicate rows                          |
| Missing Value Imputation  | Median (numeric), Mode (categorical)               |
| Label Encoding            | Binary + high-cardinality categoricals             |
| One-Hot Encoding          | Low-cardinality categoricals (3–10 unique values)  |
| Standard Scaling          | Applied to all numeric features                    |

---

## Folder Structure After Running

```
MLflow_Assignment/
├── artifacts/
│   ├── confusion_matrix_Logistic_Regression.png
│   ├── confusion_matrix_Decision_Tree.png
│   ├── confusion_matrix_Random_Forest.png
│   ├── roc_curve_Logistic_Regression.png
│   ├── roc_curve_Decision_Tree.png
│   ├── roc_curve_Random_Forest.png
│   ├── feature_importance_Decision_Tree.png
│   ├── feature_importance_Random_Forest.png
│   ├── classification_report_Logistic_Regression.txt
│   ├── classification_report_Decision_Tree.txt
│   ├── classification_report_Random_Forest.txt
│   └── model_comparison.png
├── mlruns/
│   └── [experiment_id]/
│       └── [run_id]/
│           ├── params/
│           ├── metrics/
│           └── artifacts/
└── models/
    └── best_model_random_forest.joblib   (if registry unavailable)
```

---

## Future Scope

1. **Hyperparameter Tuning** — Use `GridSearchCV` or `Optuna` for automated tuning
2. **Cross-Validation** — Add k-fold cross-validation for more robust evaluation
3. **More Models** — Add XGBoost, LightGBM, SVM for comparison
4. **Feature Engineering** — Create new features from date columns
5. **SHAP Values** — Add explainability with SHAP for model interpretability
6. **CI/CD Pipeline** — Integrate with GitHub Actions for automated retraining
7. **Model Serving** — Deploy the registered model using MLflow's serving API
8. **Docker** — Containerize the project for reproducible environments
9. **Remote Tracking** — Connect to a remote MLflow tracking server (e.g., AWS, GCP)
10. **Data Drift Detection** — Monitor for data drift in production using Evidently AI

---

## Author

- **Name**: [Your Name]
- **Course**: Machine Learning & AI Operations (MLAO)
- **Semester**: 5
- **Assignment**: 2 — MLflow Experiment Tracking

---

## License

This project is created for educational purposes as part of a college assignment.
