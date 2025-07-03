"""
Main execution script for Diabetes Prediction Project.
This script demonstrates the complete workflow from data loading to model deployment.
"""

import sys
import os
sys.path.append('src')

from src.data_preprocessing import DataPreprocessor
from src.model_training import DiabetesPredictionModel
from src.model_evaluation import ModelEvaluator, compare_models
from src.utils import load_data, data_info

def main():
    """
    Main execution function that runs the complete diabetes prediction pipeline.
    """
    print("🩺 Diabetes Prediction System")
    print("=" * 50)
    
    # Step 1: Data Loading and Exploration
    print("\n📊 Step 1: Loading and Exploring Data")
    print("-" * 30)
    
    preprocessor = DataPreprocessor('diabetes.csv')
    data = preprocessor.load_data()
    
    if data is None:
        print("❌ Failed to load data. Please check the file path.")
        return
    
    preprocessor.explore_data()
    
    # Step 2: Data Preprocessing
    print("\n🔧 Step 2: Data Preprocessing")
    print("-" * 30)
    
    # Handle missing values
    preprocessor.handle_missing_values(method='median')
    
    # Detect outliers
    outlier_info = preprocessor.detect_outliers(method='iqr')
    print(f"Outliers detected: {sum(outlier_info.values())} total")
    
    # Create features
    preprocessor.create_features()
    
    # Prepare features for ML
    preprocessor.prepare_features(include_categorical=True)
    
    # Scale features
    preprocessor.scale_features(scaler_type='standard')
    
    # Split data
    preprocessor.split_data(test_size=0.2, random_state=42)
    
    # Get processed data
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()
    print(f"✅ Data preprocessing completed!")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training")
    print("-" * 30)
    
    # Initialize model trainer
    model_trainer = DiabetesPredictionModel()
    model_trainer.load_data(X_train, X_test, y_train, y_test)
    
    # Train all models
    model_trainer.train_all_models()
    
    # Get results summary
    results_summary = model_trainer.get_results_summary()
    print("\n📈 Model Performance Summary:")
    print(results_summary)
    
    # Step 4: Model Evaluation
    print("\n📊 Step 4: Detailed Model Evaluation")
    print("-" * 30)
    
    # Create evaluators for top 3 models
    evaluators = []
    top_models = results_summary.head(3)
    
    for _, model_row in top_models.iterrows():
        model_name = model_row['Model']
        model_results = model_trainer.results[model_name]
        
        evaluator = ModelEvaluator(
            y_test, 
            model_results['predictions'],
            model_results['probabilities'],
            model_name
        )
        evaluator.calculate_metrics()
        evaluators.append(evaluator)
    
    # Compare top models
    comparison_df = compare_models(evaluators)
    print("\n🏆 Top Model Comparison:")
    print(comparison_df)
    
    # Step 5: Model Optimization (for best model)
    print("\n⚡ Step 5: Hyperparameter Tuning")
    print("-" * 30)
    
    best_model_name = results_summary.iloc[0]['Model']
    print(f"Tuning {best_model_name}...")
    
    # Define parameter grids
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        model_trainer.hyperparameter_tuning(best_model_name, param_grid)
    
    # Step 6: Final Model Selection and Saving
    print("\n💾 Step 6: Model Persistence")
    print("-" * 30)
    
    # Save the best model
    model_trainer.save_model()
    print(f"✅ Best model ({model_trainer.best_model_name}) saved successfully!")
    
    # Step 7: Demonstration
    print("\n🎯 Step 7: Prediction Demonstration")
    print("-" * 30)
    
    # Example predictions
    test_cases = [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # High risk
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],   # Low risk
        [3, 120, 78, 30, 135, 32.0, 0.400, 35] # Medium risk
    ]
    
    case_descriptions = ["High Risk", "Low Risk", "Medium Risk"]
    
    for i, test_case in enumerate(test_cases):
        # Prepare test case
        test_df = preprocessor.X_test.iloc[:1].copy()
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Update with test case values
        prediction = model_trainer.predict(test_df)
        prob = model_trainer.predict_proba(test_df)
        
        result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
        print(f"{case_descriptions[i]} Case: {result}")
        if prob is not None:
            print(f"  Probability: {prob[0][1]:.3f}")
    
    print("\n🎉 Diabetes Prediction System Complete!")
    print("=" * 50)
    print("📋 Summary:")
    print(f"• Dataset: {data.shape[0]} samples, {data.shape[1]-1} features")
    print(f"• Best Model: {model_trainer.best_model_name}")
    print(f"• Accuracy: {model_trainer.results[model_trainer.best_model_name]['accuracy']:.3f}")
    print(f"• F1-Score: {model_trainer.results[model_trainer.best_model_name]['f1_score']:.3f}")
    print("• Model saved and ready for deployment!")

if __name__ == "__main__":
    main()
