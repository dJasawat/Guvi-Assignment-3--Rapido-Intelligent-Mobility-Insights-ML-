from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
import json
import os
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()
def evalute_and_save(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    model_name,
    modelFor
):
    """
    Evaluate classification model on training and test datasets.
    Save classification reports and return evaluation metrics.
    """
    num_classes = len(set(y_train))

    average = "binary" if num_classes == 2 else "weighted"
    # -----------------------------
    # Training Metrics
    # -----------------------------
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred,average=average, zero_division=0),
        "Recall": recall_score(y_train, y_train_pred,average=average, zero_division=0),
        "F1 Score": f1_score(y_train, y_train_pred,average=average, zero_division=0)
    }

    # -----------------------------
    # Test Metrics
    # -----------------------------
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred,average=average, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred,average=average, zero_division=0),
        "F1 Score": f1_score(y_test, y_test_pred, average=average, zero_division=0)
    }

    # -----------------------------
    # Classification Reports
    # -----------------------------
    train_report = classification_report(
        y_train,
        y_train_pred,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        y_test,
        y_test_pred,
        output_dict=True,
        zero_division=0
    )

    # -----------------------------
    # Logging
    # -----------------------------
    logger.info("=" * 60)
    logger.info(f"Model : {model_name}")
    logger.info("=" * 60)

    logger.info("Training Metrics")
    for metric, value in train_metrics.items():
        logger.info(f"Train {metric:<10}: {value:.4f}")

    logger.info("-" * 60)

    logger.info("Testing Metrics")
    for metric, value in test_metrics.items():
        logger.info(f"Test  {metric:<10}: {value:.4f}")

    logger.info("-" * 60)

    # -----------------------------
    # Overfitting Check
    # -----------------------------
    f1_gap = train_metrics["F1 Score"] - test_metrics["F1 Score"]

    if f1_gap > 0.10:
        logger.warning("Possible Overfitting Detected")
    elif (
        train_metrics["F1 Score"] < 0.60
        and test_metrics["F1 Score"] < 0.60
    ):
        logger.warning("Possible Underfitting Detected")
    else:
        logger.info("Model Generalization looks good.")

    # -----------------------------
    # Save Reports
    # -----------------------------
    os.makedirs("artifacts/reports", exist_ok=True)

    report = {
        "Model": model_name,
        "Training Metrics": train_metrics,
        "Testing Metrics": test_metrics,
        "Training Classification Report": train_report,
        "Testing Classification Report": test_report,
        "F1 Gap": round(f1_gap, 4)
    }

    report_path = (
        f"artifacts/reports/"
        f"{modelFor}_{model_name}_classification_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Classification report saved at {report_path}")

    return train_metrics, test_metrics


def evaluate_and_save_fare_prediction(
    train_y,
    train_pred,
    test_y,
    test_pred,
    model_name,
    model_for="FarePrediction"
):
    """
    Evaluate regression model on both training and test datasets
    and save the metrics to a JSON file.
    """

    train_metrics = {
        "MAE": mean_absolute_error(train_y, train_pred),
        "MSE": mean_squared_error(train_y, train_pred),
        "RMSE": np.sqrt(mean_squared_error(train_y, train_pred)),
        "R2": r2_score(train_y, train_pred)
    }

    test_metrics = {
        "MAE": mean_absolute_error(test_y, test_pred),
        "MSE": mean_squared_error(test_y, test_pred),
        "RMSE": np.sqrt(mean_squared_error(test_y, test_pred)),
        "R2": r2_score(test_y, test_pred)
    }

    logger.info("=" * 60)
    logger.info(f"Model : {model_name}")
    logger.info("=" * 60)

    logger.info("Training Metrics")
    for metric, value in train_metrics.items():
        logger.info(f"Train {metric:<5}: {value:.4f}")

    logger.info("-" * 60)

    logger.info("Testing Metrics")
    for metric, value in test_metrics.items():
        logger.info(f"Test  {metric:<5}: {value:.4f}")

    # Overfitting check
    logger.info("-" * 60)

    r2_gap = train_metrics["R2"] - test_metrics["R2"]

    if r2_gap > 0.10:
        logger.warning("Possible Overfitting Detected")
    elif train_metrics["R2"] < 0.60 and test_metrics["R2"] < 0.60:
        logger.warning("Possible Underfitting Detected")
    else:
        logger.info("Model Generalization looks good.")

    report = {
        "Model": model_name,
        "Training Metrics": train_metrics,
        "Testing Metrics": test_metrics,
        "R2 Gap": round(r2_gap, 4)
    }

    os.makedirs("artifacts/reports", exist_ok=True)

    report_path = (
        f"artifacts/reports/"
        f"{model_for}_{model_name}_regression_metrics.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Regression report saved at {report_path}")

    return test_metrics