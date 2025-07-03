"""
Model training module for diabetes prediction project.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictionModel:
    """
    Class for training and evaluating diabetes prediction models.
    """
    
    def __init__(self):
        """
        Initialize the model trainer.
        """
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, X_train, X_test, y_train, y_test):
        """
        Load preprocessed training and testing data.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print(f"Data loaded successfully.")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
    
    def initialize_models(self):
        """
        Initialize all models with default parameters.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        print(f"Initialized {len(self.models)} models: {', '.join(self.models.keys())}")
    
    def train_single_model(self, model_name, model):
        """
        Train a single model and evaluate it.
        
        Args:
            model_name (str): Name of the model
            model: Scikit-learn model instance
            
        Returns:
            dict: Model performance metrics
        """
        if self.X_train is None:
            print("No training data loaded.")
            return None
        
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Calculate AUC if probability predictions are available
        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        if auc:
            print(f"  AUC: {auc:.4f}")
        print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        return results
    
    def train_all_models(self):
        """
        Train all initialized models.
        """
        if not self.models:
            self.initialize_models()
        
        print("Training all models...")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            self.results[model_name] = self.train_single_model(model_name, model)
            print("-" * 30)
        
        # Find best model based on F1-score
        best_f1 = 0
        for model_name, result in self.results.items():
            if result and result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                self.best_model = result['model']
                self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} (F1-Score: {best_f1:.4f})")
    
    def hyperparameter_tuning(self, model_name, param_grid):
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name (str): Name of the model
            param_grid (dict): Parameter grid for tuning
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found in initialized models.")
            return None
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, 
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Re-evaluate the tuned model
        self.results[model_name] = self.train_single_model(model_name, grid_search.best_estimator_)
        
        return grid_search.best_estimator_
    
    def tune_all_models(self):
        """
        Perform hyperparameter tuning for all models.
        """
        param_grids = {
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        for model_name in self.models.keys():
            if model_name in param_grids:
                self.hyperparameter_tuning(model_name, param_grids[model_name])
        
        # Re-find best model after tuning
        best_f1 = 0
        for model_name, result in self.results.items():
            if result and result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                self.best_model = result['model']
                self.best_model_name = model_name
        
        print(f"\nBest model after tuning: {self.best_model_name} (F1-Score: {best_f1:.4f})")
    
    def get_results_summary(self):
        """
        Get a summary of all model results.
        
        Returns:
            pd.DataFrame: Summary of model performance
        """
        if not self.results:
            print("No results available. Train models first.")
            return None
        
        summary_data = []
        for model_name, result in self.results.items():
            if result:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1_score'],
                    'AUC': result['auc'] if result['auc'] else 'N/A',
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('F1-Score', ascending=False)
        return df
    
    def save_model(self, model_name=None, filepath=None):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save (if None, saves best model)
            filepath (str): Path to save the model
        """
        if model_name is None:
            if self.best_model is None:
                print("No best model found. Train models first.")
                return
            model_to_save = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.results or self.results[model_name] is None:
                print(f"Model {model_name} not found or not trained.")
                return
            model_to_save = self.results[model_name]['model']
        
        if filepath is None:
            filepath = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
        
        try:
            joblib.dump(model_to_save, filepath)
            print(f"Model {model_name} saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            print("No trained model available. Train models first.")
            return None
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions using the best model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if self.best_model is None:
            print("No trained model available. Train models first.")
            return None
        
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X)
        else:
            print("Best model doesn't support probability predictions.")
            return None

def main():
    """
    Example usage of the DiabetesPredictionModel class.
    """
    # This would typically be called after data preprocessing
    print("Example usage of DiabetesPredictionModel")
    print("This requires preprocessed data from data_preprocessing.py")

if __name__ == "__main__":
    main()
