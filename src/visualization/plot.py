import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.utils.logger import setup_logger
import os


logger = setup_logger()


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Plots and saves confusion matrix.
    """
    logger.info("Generating confusion matrix")


    cm = confusion_matrix(y_true, y_pred, labels=labels)


    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


    logger.info(f"Confusion matrix saved at {save_path}")

