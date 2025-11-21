"""
Model evaluation and visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


def evaluate_model(
    y_true, y_pred, y_pred_proba=None, genre_names=None, model_name="Model"
):
    """
    Comprehensive model evaluation with multiple metrics

    Parameters
    ----------
    y_true : array-like
        True labels (encoded as integers)
    y_pred : array-like
        Predicted labels (encoded as integers)
    y_pred_proba : array-like, optional
        Prediction probabilities for top-k accuracy (shape: n_samples x n_classes)
    genre_names : list or array, optional
        Names of genre classes for classification report
    model_name : str, optional
        Name of the model for display

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")

    # Basic accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Initialize results dictionary
    results = {"accuracy": acc, "model_name": model_name}

    # Top-K accuracy if probabilities provided
    if y_pred_proba is not None:
        for k in [3, 5, 10]:
            try:
                top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=k)
                results[f"top_{k}_accuracy"] = top_k_acc
                print(f"Top-{k} Accuracy: {top_k_acc:.4f} ({top_k_acc*100:.2f}%)")
            except ValueError:
                # Handle case where k > number of classes
                pass

    # Classification report
    print(f"\nClassification Report:")
    if genre_names is not None:
        print(classification_report(y_true, y_pred, target_names=genre_names))
    else:
        print(classification_report(y_true, y_pred))

    return results


def plot_confusion_matrix(
    y_true,
    y_pred,
    genre_names=None,
    title="Confusion Matrix",
    figsize=(12, 10),
    save_path=None,
):
    """
    Plot confusion matrix heatmap

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    genre_names : list or array, optional
        Names of genre classes for axis labels
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure (if None, just displays)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Use genre names if provided, otherwise use numeric labels
    labels = genre_names if genre_names is not None else None

    sns.heatmap(
        cm,
        annot=False,  # Don't annotate if many classes
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    if labels is not None:
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return fig


def plot_training_history(
    history, title="Training History", figsize=(15, 5), save_path=None
):
    """
    Plot training and validation metrics over epochs

    Parameters
    ----------
    history : keras.callbacks.History or dict
        Training history object from model.fit() or dictionary with metrics
    title : str, optional
        Overall plot title
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Handle both History object and dictionary
    if hasattr(history, "history"):
        history_dict = history.history
    else:
        history_dict = history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot Loss
    if "loss" in history_dict:
        ax1.plot(history_dict["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history_dict:
        ax1.plot(history_dict["val_loss"], label="Validation Loss", linewidth=2)

    ax1.set_title("Model Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy - find the accuracy metric key
    acc_keys = [
        k
        for k in history_dict.keys()
        if "accuracy" in k.lower() and not k.startswith("val_")
    ]

    if acc_keys:
        metric_name = acc_keys[0]
        val_metric = f"val_{metric_name}"

        ax2.plot(
            history_dict[metric_name], label=f"Training {metric_name}", linewidth=2
        )
        if val_metric in history_dict:
            ax2.plot(
                history_dict[val_metric], label=f"Validation {metric_name}", linewidth=2
            )

        ax2.set_title("Model Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Training history plot saved to {save_path}")

    plt.show()
    return fig


def plot_model_comparison(
    results_df,
    metrics=["Accuracy", "Top-3 Accuracy"],
    title="Model Performance Comparison",
    figsize=(10, 6),
    save_path=None,
):
    """
    Plot comparison of multiple models

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: 'Model', and metric columns
    metrics : list, optional
        List of metric column names to compare
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(results_df))
    width = 0.8 / len(metrics)  # Divide bar width by number of metrics

    bars_list = []
    for i, metric in enumerate(metrics):
        if metric in results_df.columns:
            offset = width * i - (width * (len(metrics) - 1) / 2)
            bars = ax.bar(
                x + offset, results_df[metric], width, label=metric, alpha=0.8
            )
            bars_list.append(bars)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)  # Set y-axis from 0 to slightly above 1

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Model comparison plot saved to {save_path}")

    plt.show()
    return fig


def save_metrics_json(metrics_dict, filepath):
    """
    Save evaluation metrics to JSON file

    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing metrics
    filepath : str
        Path to save JSON file
    """
    import json
    import os

    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_dict = {
        key: convert_to_serializable(value) for key, value in metrics_dict.items()
    }

    with open(filepath, "w") as f:
        json.dump(serializable_dict, f, indent=2)

    print(f"Metrics saved to {filepath}")
