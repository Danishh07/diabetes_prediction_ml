"""
Simple web interface for Diabetes Prediction System using Streamlit.
Run with: streamlit run web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import sys

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-bottom: 1rem;
}
.result-box {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    color: #333;
    font-weight: 500;
}
.high-risk {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    color: #d32f2f;
}
.medium-risk {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    color: #f57c00;
}
.low-risk {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Diabetes Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered diabetes risk assessment using machine learning</p>', unsafe_allow_html=True)

# Sidebar for model information
st.sidebar.markdown("## 📊 Model Information")

# Check if model files exist
model_files = {
    'model': 'models/gradient_boosting_diabetes_model.pkl',
    'scaler': 'models/feature_scaler.pkl',
    'preprocessing': 'models/preprocessing_info.pkl'
}

files_exist = all(os.path.exists(file) for file in model_files.values())

if files_exist:
    st.sidebar.success("✅ Model loaded successfully")
    try:
        # Load preprocessing info
        preprocessing_info = joblib.load(model_files['preprocessing'])
        st.sidebar.write(f"**Model**: {preprocessing_info['model_name']}")
        st.sidebar.write(f"**Accuracy**: {preprocessing_info['model_performance']['accuracy']:.3f}")
        st.sidebar.write(f"**F1-Score**: {preprocessing_info['model_performance']['f1_score']:.3f}")
    except:
        st.sidebar.warning("⚠️ Could not load model info")
else:
    st.sidebar.error("❌ Model files not found. Please run the training notebook first.")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="sub-header">📝 Patient Information</h2>', unsafe_allow_html=True)
    
    # Create form for input
    with st.form("prediction_form"):
        # Input fields in columns
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0, max_value=20, value=1,
                help="Number of times pregnant"
            )
            
            glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0, max_value=300, value=120,
                help="Plasma glucose concentration in oral glucose tolerance test"
            )
            
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=200, value=80,
                help="Diastolic blood pressure"
            )
            
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20,
                help="Triceps skinfold thickness"
            )
        
        with input_col2:
            insulin = st.number_input(
                "Insulin Level (μU/mL)",
                min_value=0, max_value=1000, value=85,
                help="2-Hour serum insulin"
            )
            
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                help="Body Mass Index"
            )
            
            diabetes_pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                help="Genetic predisposition to diabetes"
            )
            
            age = st.number_input(
                "Age (years)",
                min_value=18, max_value=100, value=30,
                help="Age in years"
            )
        
        # Submit button
        submitted = st.form_submit_button(
            "🔍 Predict Diabetes Risk",
            use_container_width=True
        )

with col2:
    st.markdown('<h2 class="sub-header">ℹ️ Information</h2>', unsafe_allow_html=True)
    
    # Information boxes
    st.info("""
    **Normal Ranges:**
    - Glucose: 70-100 mg/dL (fasting)
    - Blood Pressure: <80 mm Hg (diastolic)
    - BMI: 18.5-24.9 kg/m²
    """)
    
    st.warning("""
    **High Risk Indicators:**
    - Glucose >126 mg/dL
    - BMI >30 kg/m²
    - Family history of diabetes
    - Age >45 years
    """)
    
    st.error("""
    **⚠️ Medical Disclaimer:**
    This tool is for educational purposes only. 
    Consult healthcare professionals for medical advice.
    """)

# Prediction logic
if submitted and files_exist:
    try:
        # Load model components
        model = joblib.load(model_files['model'])
        scaler = joblib.load(model_files['scaler'])
        preprocessing_info = joblib.load(model_files['preprocessing'])
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Apply same preprocessing as training
        # Handle missing values (zeros)
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']:
            if input_data[col].iloc[0] == 0:
                input_data[col] = preprocessing_info['feature_medians'][col]
        
        # Feature engineering
        input_data['BMI_Category'] = pd.cut(input_data['BMI'], 
                                           bins=[0, 18.5, 25, 30, float('inf')], 
                                           labels=[0, 1, 2, 3]).astype(int)
        
        input_data['Age_Group'] = pd.cut(input_data['Age'], 
                                        bins=[0, 25, 35, 45, float('inf')], 
                                        labels=[0, 1, 2, 3]).astype(int)
        
        input_data['Glucose_Level'] = pd.cut(input_data['Glucose'], 
                                            bins=[0, 100, 126, float('inf')], 
                                            labels=[0, 1, 2]).astype(int)
        
        input_data['BP_Category'] = pd.cut(input_data['BloodPressure'], 
                                          bins=[0, 80, 90, float('inf')], 
                                          labels=[0, 1, 2]).astype(int)
        
        input_data['Insulin_Glucose_Ratio'] = input_data['Insulin'] / input_data['Glucose']
        input_data['Pregnancy_Age_Risk'] = input_data['Pregnancies'] * (input_data['Age'] / 50)
        
        # Select features and scale
        features = input_data[preprocessing_info['feature_names']]
        features_scaled = scaler.transform(features)
        
        # Convert back to DataFrame with feature names to avoid sklearn warning
        features_scaled_df = pd.DataFrame(features_scaled, columns=preprocessing_info['feature_names'])
        
        # Make prediction
        prediction = model.predict(features_scaled_df)[0]
        probability = model.predict_proba(features_scaled_df)[0, 1]
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "High"
            risk_class = "high-risk"
            risk_icon = "🔴"
        elif probability >= 0.3:
            risk_level = "Medium"
            risk_class = "medium-risk"
            risk_icon = "🟡"
        else:
            risk_level = "Low"
            risk_class = "low-risk"
            risk_icon = "🟢"
        
        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
        
        result_text = "Diabetes" if prediction == 1 else "No Diabetes"
        
        # Main result box
        text_color = "#d32f2f" if risk_level == "High" else "#f57c00" if risk_level == "Medium" else "#2e7d32"
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: {text_color};">{risk_icon} Prediction: {result_text}</h3>
            <p style="margin: 0.3rem 0; font-size: 1.1rem; color: {text_color};"><strong>Probability:</strong> {probability:.1%}</p>
            <p style="margin: 0.3rem 0; font-size: 1.1rem; color: {text_color};"><strong>Risk Level:</strong> {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Diabetes Probability", f"{probability:.1%}")
        
        with col2:
            st.metric("Risk Level", risk_level)
        
        with col3:
            confidence = max(probability, 1-probability)
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Risk factors analysis
        st.markdown("### 📋 Risk Factor Analysis")
        
        risk_factors = []
        protective_factors = []
        
        # Analyze risk factors
        if glucose >= 126:
            risk_factors.append(f"High glucose level ({glucose} mg/dL)")
        elif glucose <= 100:
            protective_factors.append(f"Normal glucose level ({glucose} mg/dL)")
        
        if bmi >= 30:
            risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
        elif bmi < 25:
            protective_factors.append(f"Healthy weight (BMI: {bmi:.1f})")
        
        if age >= 45:
            risk_factors.append(f"Advanced age ({age} years)")
        elif age < 35:
            protective_factors.append(f"Young age ({age} years)")
        
        if blood_pressure >= 90:
            risk_factors.append(f"High blood pressure ({blood_pressure} mm Hg)")
        
        if pregnancies >= 3:
            risk_factors.append(f"Multiple pregnancies ({pregnancies})")
        
        # Display risk factors
        if risk_factors:
            st.markdown("**⚠️ Risk Factors Identified:**")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        
        if protective_factors:
            st.markdown("**✅ Protective Factors:**")
            for factor in protective_factors:
                st.markdown(f"- {factor}")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if risk_level == "High":
            st.error("""
            **High Risk - Immediate Action Recommended:**
            - Consult with a healthcare provider immediately
            - Consider diabetes screening tests (HbA1c, fasting glucose)
            - Implement lifestyle changes (diet, exercise)
            - Monitor blood glucose regularly
            """)
        elif risk_level == "Medium":
            st.warning("""
            **Medium Risk - Prevention Recommended:**
            - Schedule regular check-ups with healthcare provider
            - Adopt healthy lifestyle habits
            - Maintain healthy weight through diet and exercise
            - Monitor risk factors periodically
            """)
        else:
            st.success("""
            **Low Risk - Maintain Healthy Lifestyle:**
            - Continue current healthy habits
            - Regular physical activity and balanced diet
            - Periodic health screenings as recommended
            - Stay informed about diabetes prevention
            """)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error("Please ensure all model files are properly trained and saved.")

elif submitted and not files_exist:
    st.error("❌ Model files not found. Please run the training notebook first to generate the model files.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🩺 Diabetes Prediction System | Built with Machine Learning | For Educational Use Only</p>
    <p>Always consult healthcare professionals for medical advice and diagnosis.</p>
</div>
""", unsafe_allow_html=True)

# Add streamlit to requirements if not already there
if not os.path.exists('requirements.txt'):
    pass
else:
    with open('requirements.txt', 'r') as f:
        content = f.read()
    if 'streamlit' not in content:
        with open('requirements.txt', 'a') as f:
            f.write('\nstreamlit==1.28.0')