"""
Model evaluation module for diabetes prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ModelEvaluator:
    """
    Class for comprehensive model evaluation.
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Initialize the evaluator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name (str): Name of the model
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(self):
        """
        Calculate all evaluation metrics.
        """
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred)
        self.metrics['f1_score'] = f1_score(self.y_true, self.y_pred)
        self.metrics['specificity'] = self._calculate_specificity()
        
        # AUC metrics if probabilities are available
        if self.y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
            self.metrics['pr_auc'] = auc(recall, precision)
        
        return self.metrics
    
    def _calculate_specificity(self):
        """
        Calculate specificity (true negative rate).
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def print_metrics(self):
        """
        Print all calculated metrics.
        """
        if not self.metrics:
            self.calculate_metrics()
        
        print(f"\n{self.model_name} - Evaluation Metrics")
        print("=" * 50)
        print(f"Accuracy:     {self.metrics['accuracy']:.4f}")
        print(f"Precision:    {self.metrics['precision']:.4f}")
        print(f"Recall:       {self.metrics['recall']:.4f}")
        print(f"Specificity:  {self.metrics['specificity']:.4f}")
        print(f"F1-Score:     {self.metrics['f1_score']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"ROC AUC:      {self.metrics['roc_auc']:.4f}")
        if 'pr_auc' in self.metrics:
            print(f"PR AUC:       {self.metrics['pr_auc']:.4f}")
    
    def plot_confusion_matrix(self, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print confusion matrix interpretation
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (TN):  {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP):  {tp}")
    
    def plot_roc_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot ROC curve.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for ROC curve.")
            return
        
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'{self.model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, figsize=(8, 6), save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for PR curve.")
            return
        
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'{self.model_name} (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {self.model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, figsize=(10, 6), save_path=None):
        """
        Plot distribution of predictions.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for distribution plot.")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot histograms for each class
        for class_val in [0, 1]:
            class_probs = self.y_pred_proba[self.y_true == class_val]
            label = 'No Diabetes' if class_val == 0 else 'Diabetes'
            plt.hist(class_probs, bins=30, alpha=0.7, label=label, density=True)
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Probability Distribution - {self.model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def interactive_roc_curve(self):
        """
        Create interactive ROC curve using Plotly.
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for ROC curve.")
            return
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{self.model_name} (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>Threshold: %{text:.3f}<extra></extra>',
            text=thresholds
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Interactive ROC Curve - {self.model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=600,
            showlegend=True
        )
        
        fig.show()
    
    def interactive_precision_recall_curve(self):
        """
        Create interactive Precision-Recall curve using Plotly.
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for PR curve.")
            return
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{self.model_name} (AUC = {pr_auc:.3f})',
            line=dict(color='blue', width=3),
            hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{text:.3f}<extra></extra>',
            text=thresholds
        ))
        
        fig.update_layout(
            title=f'Interactive Precision-Recall Curve - {self.model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=600,
            showlegend=True
        )
        
        fig.show()
    
    def threshold_analysis(self, thresholds=None):
        """
        Analyze model performance at different thresholds.
        
        Args:
            thresholds (list): List of thresholds to analyze
        """
        if self.y_pred_proba is None:
            print("No probability predictions available for threshold analysis.")
            return
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            accuracy = accuracy_score(self.y_true, y_pred_thresh)
            precision = precision_score(self.y_true, y_pred_thresh)
            recall = recall_score(self.y_true, y_pred_thresh)
            f1 = f1_score(self.y_true, y_pred_thresh)
            
            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
        df = pd.DataFrame(results)
        print(f"\nThreshold Analysis - {self.model_name}")
        print("=" * 50)
        print(df.round(4))
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(df['Threshold'], df['Accuracy'], marker='o')
        plt.title('Accuracy vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(df['Threshold'], df['Precision'], marker='o')
        plt.title('Precision vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(df['Threshold'], df['Recall'], marker='o')
        plt.title('Recall vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(df['Threshold'], df['F1-Score'], marker='o')
        plt.title('F1-Score vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F1-Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        """
        report = classification_report(self.y_true, self.y_pred, 
                                     target_names=['No Diabetes', 'Diabetes'],
                                     output_dict=True)
        
        print(f"\nDetailed Classification Report - {self.model_name}")
        print("=" * 60)
        print(classification_report(self.y_true, self.y_pred, 
                                  target_names=['No Diabetes', 'Diabetes']))
        
        return report
    
    def save_evaluation_report(self, filepath):
        """
        Save comprehensive evaluation report to file.
        
        Args:
            filepath (str): Path to save the report
        """
        if not self.metrics:
            self.calculate_metrics()
        
        report = {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'classification_report': self.get_classification_report(),
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist()
        }
        
        import json
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Evaluation report saved to {filepath}")
        except Exception as e:
            print(f"Error saving report: {e}")

def compare_models(evaluators):
    """
    Compare multiple model evaluators.
    
    Args:
        evaluators (list): List of ModelEvaluator objects
    """
    comparison_data = []
    
    for evaluator in evaluators:
        if not evaluator.metrics:
            evaluator.calculate_metrics()
        
        comparison_data.append({
            'Model': evaluator.model_name,
            'Accuracy': evaluator.metrics['accuracy'],
            'Precision': evaluator.metrics['precision'],
            'Recall': evaluator.metrics['recall'],
            'F1-Score': evaluator.metrics['f1_score'],
            'Specificity': evaluator.metrics['specificity'],
            'ROC AUC': evaluator.metrics.get('roc_auc', 'N/A'),
            'PR AUC': evaluator.metrics.get('pr_auc', 'N/A')
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    print("\nModel Comparison")
    print("=" * 80)
    print(df.round(4))
    
    return df

def main():
    """
    Example usage of the ModelEvaluator class.
    """
    print("Example usage of ModelEvaluator")
    print("This requires model predictions from model_training.py")

if __name__ == "__main__":
    main()
