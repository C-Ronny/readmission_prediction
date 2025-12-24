import joblib
import json
import os
from tensorflow import keras
import streamlit as st

@st.cache_resource
def load_models():
    """Load all models and preprocessing pipeline"""
    
    models_dir = 'models'
    
    try:
        # Load preprocessing pipeline
        preprocessing = joblib.load(os.path.join(models_dir, 'preprocessing_pipeline.pkl'))
        
        # Load Logistic Regression
        lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
        with open(os.path.join(models_dir, 'logistic_regression_metadata.json'), 'r') as f:
            lr_metadata = json.load(f)
        
        # Load XGBoost
        xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
        with open(os.path.join(models_dir, 'xgboost_metadata.json'), 'r') as f:
            xgb_metadata = json.load(f)
        
        # Load Neural Network
        nn_model = keras.models.load_model(os.path.join(models_dir, 'neural_network_model.h5'))
        with open(os.path.join(models_dir, 'neural_network_metadata.json'), 'r') as f:
            nn_metadata = json.load(f)
        
        # Load model comparison
        with open(os.path.join(models_dir, 'model_comparison.json'), 'r') as f:
            comparison = json.load(f)
        
        return {
            'preprocessing': preprocessing,
            'logistic_regression': {'model': lr_model, 'metadata': lr_metadata},
            'xgboost': {'model': xgb_model, 'metadata': xgb_metadata},
            'neural_network': {'model': nn_model, 'metadata': nn_metadata},
            'comparison': comparison
        }
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

def get_feature_names():
    """Get feature names from preprocessing pipeline"""
    models_dir = 'models'
    preprocessing = joblib.load(os.path.join(models_dir, 'preprocessing_pipeline.pkl'))
    return preprocessing['feature_names']