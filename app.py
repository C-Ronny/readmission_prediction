import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.model_loader import load_models
from utils.preprocessor import preprocess_input
from utils.predictor import predict_readmission, get_risk_message

# Page configuration
st.set_page_config(
    page_title="Diabetes Readmission Predictor",    
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for hospital theme
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0077B6;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0077B6;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28A745;
    }
    .risk-medium {
        background-color: #FFF3CD;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
    }
    .risk-high {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #DC3545;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_all_models():
    return load_models()

models_data = load_all_models()

if models_data is None:
    st.error("Failed to load models. Please ensure all model files are in the 'models/' directory.")
    st.stop()

# Header
st.markdown('<div class="main-header">Hospital Readmission Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict 30-day readmission risk using machine learning</div>', unsafe_allow_html=True)

# Sidebar - Model Information
with st.sidebar:
    st.header("Model Performance")
    
    comparison = models_data['comparison']['model_comparison']
    
    perf_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'Neural Network'],
        'ROC-AUC': [
            comparison['Logistic_Regression']['roc_auc'],
            comparison['XGBoost_Optimized']['roc_auc'],
            comparison['Neural_Network']['roc_auc']
        ],
        'F1-Score': [
            comparison['Logistic_Regression']['f1_score'],
            comparison['XGBoost_Optimized']['f1_score'],
            comparison['Neural_Network']['f1_score']
        ]
    })
    
    st.dataframe(perf_df.style.format({'ROC-AUC': '{:.4f}', 'F1-Score': '{:.4f}'}), 
                 use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.header("About")
    st.write("""
    This application predicts the risk of hospital readmission within 30 days for diabetes patients.
    
    **Dataset:** 101,766 patient encounters from 130 US hospitals (1999-2008)
    
    **Models:** 
    - Logistic Regression
    - XGBoost (Optimized)
    - Neural Network
    """)
    
    st.markdown("---")
    st.caption("")

# Main content
tab1, tab2 = st.tabs(["Make Prediction", "Model Comparison"])

with tab1:
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        age_group = st.selectbox("Age Group", 
                                 options=['Age_0_30', 'Age_30_60', 'Age_60_plus'],
                                 index=2,
                                 format_func=lambda x: x.replace('Age_', '').replace('_', '-') + ' years')
        
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        
        race = st.selectbox("Race", 
                           options=['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other'])
    
    with col2:
        st.markdown("**Medical History**")
        primary_diagnosis = st.selectbox("Primary Diagnosis",
                                        options=['Circulatory', 'Diabetes', 'Respiratory', 
                                                'Digestive', 'Injury', 'Musculoskeletal',
                                                'Genitourinary', 'Neoplasms', 'Other'])
        
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 4)
        
        num_medications = st.number_input("Number of Medications", 0, 50, 15)
        
        num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 100, 45)
    
    with col3:
        st.markdown("**Clinical Tests & History**")
        hba1c_category = st.selectbox("HbA1c Test Result",
                                      options=['No_HbA1c_Test', 'Normal_HbA1c', 
                                              'High_HbA1c_MedChanged', 'High_HbA1c_NoMedChange'],
                                      format_func=lambda x: x.replace('_', ' '))
        
        diabetes_med = st.selectbox("Diabetes Medication Prescribed?", options=['Yes', 'No'])
        
        number_inpatient = st.number_input("Previous Inpatient Visits", 0, 20, 0)
        
        number_emergency = st.number_input("Previous Emergency Visits", 0, 20, 0)
    
    st.markdown("---")
    
    # Model selection
    col_model, col_button = st.columns([2, 1])
    
    with col_model:
        selected_model = st.selectbox(
            "Select Prediction Model",
            options=['XGBoost (Recommended)', 'Logistic Regression', 'Neural Network'],
            help="XGBoost has the best performance with optimal threshold tuning"
        )
    
    with col_button:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        # Prepare user input
        user_input = {
            'age_group': age_group,
            'gender': gender,
            'race': race,
            'primary_diagnosis': primary_diagnosis,
            'time_in_hospital': time_in_hospital,
            'num_medications': num_medications,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': 2,  # Default
            'number_inpatient': number_inpatient,
            'number_emergency': number_emergency,
            'number_outpatient': 0,  # Default
            'hba1c_category': hba1c_category,
            'diabetes_med': diabetes_med,
            'num_medications_prescribed': num_medications,
            'admission_type_id': 1,  # Default: Emergency
            'discharge_disposition_id': 1,  # Default: Home
            'admission_source_id': 7  # Default: Emergency Room
        }
        
        # Map model selection to model name
        model_mapping = {
            'XGBoost (Recommended)': 'xgboost',
            'Logistic Regression': 'logistic_regression',
            'Neural Network': 'neural_network'
        }
        model_name = model_mapping[selected_model]
        
        try:
            # Preprocess input
            with st.spinner("Preprocessing patient data..."):
                feature_vector = preprocess_input(user_input, models_data['preprocessing'])
            
            # Make prediction
            with st.spinner("Analyzing readmission risk..."):
                result = predict_readmission(models_data[model_name], feature_vector, model_name)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Risk level banner
            risk_class = f"risk-{result['risk_color']}"
            st.markdown(f'<div class="{risk_class}">{get_risk_message(result["risk_level"], result["probability"])}</div>', 
                       unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Readmission Probability", f"{result['probability']*100:.1f}%")
            
            with col2:
                st.metric("Risk Level", result['risk_level'])
            
            with col3:
                st.metric("Model Used", selected_model.split(' ')[0])
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Readmission Risk"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': result['risk_color']},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.markdown("### Model Performance Metrics")
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                st.metric("ROC-AUC", f"{result['model_performance']['roc_auc']:.4f}")
            with perf_col2:
                st.metric("F1-Score", f"{result['model_performance']['f1_score']:.4f}")
            with perf_col3:
                st.metric("Precision", f"{result['model_performance']['precision']:.4f}")
            with perf_col4:
                st.metric("Recall", f"{result['model_performance']['recall']:.4f}")
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check that all input fields are filled correctly.")

with tab2:
    st.subheader("Model Performance Comparison")
    
    # Detailed comparison table
    detailed_comparison = pd.DataFrame({
        'Metric': ['ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy'],
        'Logistic Regression': [
            comparison['Logistic_Regression']['roc_auc'],
            comparison['Logistic_Regression']['f1_score'],
            comparison['Logistic_Regression']['precision'],
            comparison['Logistic_Regression']['recall'],
            comparison['Logistic_Regression']['accuracy']
        ],
        'XGBoost (Optimized)': [
            comparison['XGBoost_Optimized']['roc_auc'],
            comparison['XGBoost_Optimized']['f1_score'],
            comparison['XGBoost_Optimized']['precision'],
            comparison['XGBoost_Optimized']['recall'],
            comparison['XGBoost_Optimized']['accuracy']
        ],
        'Neural Network': [
            comparison['Neural_Network']['roc_auc'],
            comparison['Neural_Network']['f1_score'],
            comparison['Neural_Network']['precision'],
            comparison['Neural_Network']['recall'],
            comparison['Neural_Network']['accuracy']
        ]
    })
    
    # Format the dataframe (compatibility fix - no highlighting)
    formatted_comparison = detailed_comparison.copy()
    for col in ['Logistic Regression', 'XGBoost (Optimized)', 'Neural Network']:
        formatted_comparison[col] = formatted_comparison[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(formatted_comparison, use_container_width=True, hide_index=True)
    
    st.info("**Best values per metric:** Higher is better for all metrics shown. XGBoost performs best overall.")
    
    st.markdown("---")
    
    # Best model info
    best_model = models_data['comparison']['best_model']
    
    st.success(f"""
    **Best Performing Model:** {best_model['name']}
    - **ROC-AUC:** {best_model['roc_auc']:.4f}
    - **F1-Score:** {best_model['f1_score']:.4f}
    - **Optimal Threshold:** {best_model['optimal_threshold']:.3f}
    """)
    
    # Dataset info
    st.markdown("---")
    st.subheader("Dataset Information")
    
    dataset_info = models_data['comparison']['dataset_info']
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.metric("Total Patient Encounters", f"{dataset_info['total_encounters']:,}")
        st.metric("Training Set Size", f"{dataset_info['train_size']:,}")
    
    with info_col2:
        st.metric("Test Set Size", f"{dataset_info['test_size']:,}")
        st.metric("Engineered Features", f"{dataset_info['n_features_engineered']}")