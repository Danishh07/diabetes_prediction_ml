"""
Utility functions for the diabetes prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(filepath):
    """
    Load the diabetes dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def data_info(data):
    """
    Display basic information about the dataset.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
    """
    print("Dataset Information:")
    print("=" * 50)
    print(f"Shape: {data.shape}")
    print(f"\nColumn names: {list(data.columns)}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nMissing values:\n{data.isnull().sum()}")
    print(f"\nBasic statistics:\n{data.describe()}")

def plot_correlation_matrix(data, figsize=(10, 8)):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data (pd.DataFrame): Dataset
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(data, target_col='Outcome'):
    """
    Plot distributions of all features split by target variable.
    
    Args:
        data (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    features = [col for col in data.columns if col != target_col]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(features):
        if i < len(axes):
            for outcome in data[target_col].unique():
                subset = data[data[target_col] == outcome]
                axes[i].hist(subset[feature], alpha=0.7, 
                           label=f'{target_col}={outcome}', bins=20)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
    
    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name (str): Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
    """
    print(f"\nClassification Report - {model_name}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, 
                              target_names=['No Diabetes', 'Diabetes']))

def save_results(results_dict, filepath):
    """
    Save model results to a file.
    
    Args:
        results_dict (dict): Dictionary containing model results
        filepath (str): Path to save the results
    """
    import json
    try:
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=4)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")

def interactive_feature_plots(data, target_col='Outcome'):
    """
    Create interactive plots using Plotly for feature exploration.
    
    Args:
        data (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    features = [col for col in data.columns if col != target_col]
    
    # Create subplots
    fig = make_subplots(
        rows=len(features), cols=1,
        subplot_titles=[f'Distribution of {feature}' for feature in features],
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red']
    target_labels = ['No Diabetes', 'Diabetes']
    
    for i, feature in enumerate(features):
        for outcome in data[target_col].unique():
            subset = data[data[target_col] == outcome]
            fig.add_trace(
                go.Histogram(
                    x=subset[feature],
                    name=f'{target_labels[outcome]}',
                    opacity=0.7,
                    marker_color=colors[outcome],
                    showlegend=(i == 0)  # Only show legend for first subplot
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=300*len(features),
        title_text="Feature Distributions by Diabetes Outcome",
        showlegend=True
    )
    
    fig.show()
