"""
Utility functions for the ML pipeline.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_results_dir(base_dir="results"):
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def plot_feature_importance(importance_df, top_n=20, save_path=None):
    """Plot feature importance."""
    plt.figure(figsize=(12, 8))

    # Plot top N features
    top_features = importance_df.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=top_features)

    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix for classification results."""
    from sklearn.metrics import confusion_matrix

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"Confusion matrix plot saved to {save_path}")
    else:
        plt.show()


def plot_roc_curve(y_true, y_score, save_path=None):
    """Plot ROC curve for binary classification."""
    from sklearn.metrics import roc_curve, auc

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logging.info(f"ROC curve plot saved to {save_path}")
    else:
        plt.show()


def save_metrics(metrics, path):
    """Save evaluation metrics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {path}")


def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Detect outliers in specified columns."""
    outliers = {}

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if method == 'iqr':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            elif method == 'zscore':
                # Z-score method
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                outliers[col] = df[abs(z_scores) > threshold].index.tolist()

    return outliers


def correlation_analysis(df, target_col=None, threshold=0.7):
    """Analyze correlation between features and with target."""
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()

    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    # Target correlation
    target_corr = None
    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].sort_values(ascending=False)

    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'target_correlation': target_corr
    }


if __name__ == "__main__":
    # Test the utility functions
    print("Utility functions loaded successfully")
