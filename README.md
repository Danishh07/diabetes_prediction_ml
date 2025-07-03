# 🩺 Diabetes Prediction using Machine Learning

This project implements a machine learning model to predict diabetes based on the Pima Indians Diabetes Database from Kaggle.

## 📈 Dataset Description

The dataset contains 768 samples with 8 features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1) - Target variable

## 📁 Project Structure

```
Diabetics Prediction/
├── diabetes.csv                           # Dataset file
├── notebooks/
│   └── diabetes_prediction_complete.ipynb # Complete analysis notebook
├── src/
│   ├── data_preprocessing.py              # Data preprocessing functions
│   ├── model_training.py                  # Model training utilities
│   ├── model_evaluation.py               # Evaluation metrics
│   └── utils.py                          # Helper functions
├── models/
│   ├── gradient_boosting_diabetes_model.pkl # Trained model
│   ├── feature_scaler.pkl                # Feature scaler
│   └── preprocessing_info.pkl            # Preprocessing metadata
├── web_app.py                            # Streamlit web application
├── main.py                               # Command-line interface
├── requirements.txt                      # Python dependencies
└── README.md                             # Complete project documentation
```

## ⚙️ Quick Start

### Option 1: Web Application (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web application
streamlit run web_app.py
```
Open your browser to `http://localhost:8501` for the interactive interface.

### Option 2: Jupyter Notebook
```bash
# Install dependencies
pip install -r requirements.txt

# Open the complete analysis notebook
jupyter notebook notebooks/diabetes_prediction_complete.ipynb
```

### Option 3: Command Line
```bash
# Run the main script
python main.py
```

## 🤖 Models Implemented

- Logistic Regression
- Random Forest
- Gradient Boosting

## 📜 Results

The best performing model (Gradient Boosting) achieved the following metrics:
- **Accuracy**: 76.6%
- **F1-Score**: 64.7%
- **Precision**: 68.8%
- **Recall**: 61.1%
- **ROC-AUC**: 83.4%

All three models provide different strengths:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Robust ensemble method with feature importance
- **Gradient Boosting**: Best performance with optimized hyperparameters

## 📄 Features

### Technical Features
- **Complete ML Pipeline**: Data preprocessing → Feature engineering → Model training → Evaluation
- **Production Ready**: Saved models with preprocessing pipeline
- **Web Interface**: Interactive Streamlit application

### Business Features
- 🩺 **Clinical Risk Assessment**: Low/Medium/High risk categorization
- 📊 **Feature Importance**: Understand key diabetes predictors
- 🎯 **Actionable Insights**: Personalized recommendations
- 📈 **Performance Tracking**: Comprehensive metrics and benchmarks

## 🛠️ Troubleshooting

### Common Issues

**sklearn Warning about feature names:**
```
X does not have valid feature names, but GradientBoostingClassifier was fitted with feature names
```
**Solution**: This has been fixed in the latest version. The model now properly maintains feature names throughout the prediction pipeline.

**Model files not found:**
```
❌ Model files not found. Please run the training notebook first.
```
**Solution**: Run the complete notebook (`notebooks/diabetes_prediction_complete.ipynb`) to generate the required model files.

**No module named '_loss' error:**
```
❌ Error making prediction: No module named '_loss'
```
**Solution**: This is a scikit-learn version compatibility issue between your local environment and deployment environment. Make sure your requirements.txt specifies the exact scikit-learn version (scikit-learn==1.3.0) and add setuptools and wheel to ensure proper installation:
```
setuptools>=65.5.1
wheel>=0.38.0
scikit-learn==1.3.0
```

## 🎯 Key Project Highlights

### Technical Excellence
- **End-to-End ML Pipeline**: Complete workflow from raw data to deployment
- **Production Ready**: Saved models with preprocessing pipeline
- **Error Handling**: Robust input validation and exception management
- **Best Practices**: Stratified sampling, cross-validation, proper metrics

### Data Science Achievements
- **Feature Engineering**: Created 6 meaningful features from domain knowledge
- **Data Quality**: Handled medical impossibilities and missing values (48.7% insulin missing)
- **Model Selection**: Systematic comparison of 3 algorithms with different approaches
- **Validation**: 5-fold stratified cross-validation with stable performance (±5.8%)

### Business Impact
- **Healthcare Value**: Early diabetes detection tool with 76.6% accuracy
- **Clinical Relevance**: Risk stratification (Low/Medium/High) with actionable insights
- **Cost Effective**: Uses standard medical measurements available in routine checkups
- **Scalable**: Ready for integration with healthcare systems

## 📊 Detailed Results

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** | **76.6%** | **68.8%** | **61.1%** | **64.7%** | **83.4%** |
| Random Forest | 74.7% | 70.4% | 53.4% | 60.8% | 73.8% |
| Logistic Regression | 75.3% | 68.9% | 55.2% | 61.2% | 74.3% |

### Feature Importance (Top 5)
1. **Glucose** (34.2%) - Primary diabetes indicator
2. **BMI** (15.6%) - Obesity risk factor
3. **Age** (13.4%) - Risk increases with age
4. **DiabetesPedigreeFunction** (9.8%) - Genetic predisposition
5. **Insulin_Glucose_Ratio** (8.7%) - Engineered metabolic indicator

### Clinical Metrics
- **Sensitivity**: 61.1% (identifies 61 of 100 diabetic patients)
- **Specificity**: 84.0% (correctly classifies 84 of 100 non-diabetic patients)
- **Positive Predictive Value**: 68.8% (69 of 100 positive predictions are correct)
- **False Positive Rate**: 16.0% (acceptable for screening tool)

## 🧬 Data Analysis Insights

### Data Quality Issues Found
- **Missing Values**: 48.7% of insulin values were zero (medical impossibility)
- **Other Zeros**: Glucose (0.7%), BloodPressure (4.6%), SkinThickness (29.6%), BMI (1.4%)
- **Solution**: Replaced with median values based on medical domain knowledge

### Feature Engineering Success
| Original Feature | Engineered Feature | Importance Gain |
|------------------|-------------------|-----------------|
| Insulin + Glucose | Insulin_Glucose_Ratio | +8.7% |
| BMI | BMI_Category | Clinical thresholds |
| Age | Age_Group | Life stage risk |
| Glucose | Glucose_Level | Diabetes thresholds |

## 🤝🏽 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

#

**🩺 Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare providers for medical decisions.