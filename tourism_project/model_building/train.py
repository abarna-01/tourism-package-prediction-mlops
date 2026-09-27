
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

XTRAIN_PATH = "Xtrain.csv"
XTEST_PATH = "Xtest.csv"
YTRAIN_PATH = "ytrain.csv"
YTEST_PATH = "ytest.csv"

MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "tourism_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# Load train and test data
# ---------------------------------------------------------

Xtrain = pd.read_csv(XTRAIN_PATH)
Xtest = pd.read_csv(XTEST_PATH)

ytrain = pd.read_csv(YTRAIN_PATH).iloc[:, 0]
ytest = pd.read_csv(YTEST_PATH).iloc[:, 0]

print("Training data shape:", Xtrain.shape)
print("Testing data shape:", Xtest.shape)

# ---------------------------------------------------------
# Identify numerical and categorical columns
# ---------------------------------------------------------

numerical_features = Xtrain.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = make_column_transformer(
    (
        StandardScaler(),
        numerical_features
    ),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    remainder="drop"
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

# Combine preprocessing + model into one pipeline
model_pipeline = make_pipeline(
    preprocessor,
    model
)

# ---------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------

param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1]
}

print("\nHyperparameter grid:")
print(param_grid)

# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Tourism Package Prediction")

# ---------------------------------------------------------
# Hyperparameter tuning + MLflow tracking
# ---------------------------------------------------------

with mlflow.start_run(run_name="Tourism_GridSearch") as parent_run:

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(Xtrain, ytrain)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation ROC-AUC:")
    print(grid_search.best_score_)

    # -----------------------------------------------------
    # Log every hyperparameter combination as nested run
    # -----------------------------------------------------

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(
            run_name=f"GridSearch_Run_{i + 1}",
            nested=True
        ):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_test_roc_auc",
                float(mean_score)
            )

            mlflow.log_metric(
                "std_test_roc_auc",
                float(std_score)
            )

    # -----------------------------------------------------
    # Best model
    # -----------------------------------------------------

    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_roc_auc",
        float(grid_search.best_score_)
    )

    # -----------------------------------------------------
    # Evaluate on test data
    # -----------------------------------------------------

    y_pred = best_model.predict(Xtest)
    y_prob = best_model.predict_proba(Xtest)[:, 1]

    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred, zero_division=0)
    recall = recall_score(ytest, y_pred, zero_division=0)
    f1 = f1_score(ytest, y_pred, zero_division=0)
    roc_auc = roc_auc_score(ytest, y_prob)

    # -----------------------------------------------------
    # Log evaluation metrics
    # -----------------------------------------------------

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        ytest,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    print("Test Accuracy :", accuracy)
    print("Test Precision:", precision)
    print("Test Recall   :", recall)
    print("Test F1 Score :", f1)
    print("Test ROC-AUC   :", roc_auc)

    # Save classification report
    report_path = os.path.join(
        MODEL_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    mlflow.log_artifact(report_path)

    # -----------------------------------------------------
    # Save the best model
    # -----------------------------------------------------

    joblib.dump(best_model, MODEL_PATH)

    print("\nBest model saved to:")
    print(MODEL_PATH)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="tourism_model"
    )

    print("\nMLflow Run ID:")
    print(parent_run.info.run_id)
