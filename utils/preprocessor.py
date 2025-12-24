import pandas as pd
import numpy as np

def create_feature_vector(user_input, feature_names):
    """
    Create a feature vector from user input that matches the model's expected features
    
    Args:
        user_input: Dictionary of user inputs
        feature_names: List of all feature names expected by the model
    
    Returns:
        pandas DataFrame with all features
    """
    
    # Initialize all features with 0
    feature_vector = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map user inputs to features
    
    # Numerical features (direct mapping)
    numerical_mappings = {
        'time_in_hospital': 'time_in_hospital',
        'num_lab_procedures': 'num_lab_procedures',
        'num_procedures': 'num_procedures',
        'num_medications': 'num_medications',
        'number_emergency': 'number_emergency',
        'number_inpatient': 'number_inpatient',
        'number_outpatient': 'number_outpatient',
        'num_medications_prescribed': 'num_medications_prescribed',
        'admission_type_id': 'admission_type_id',
        'discharge_disposition_id': 'discharge_disposition_id',
        'admission_source_id': 'admission_source_id'
    }
    
    for user_key, feature_key in numerical_mappings.items():
        if user_key in user_input and feature_key in feature_vector.columns:
            feature_vector[feature_key] = user_input[user_key]
    
    # One-hot encoded features
    # Gender
    gender_col = f"gender_{user_input.get('gender', 'Male')}"
    if gender_col in feature_vector.columns:
        feature_vector[gender_col] = 1
    
    # Race
    race_col = f"race_{user_input.get('race', 'Caucasian')}"
    if race_col in feature_vector.columns:
        feature_vector[race_col] = 1
    
    # Age group
    age_col = f"age_group_{user_input.get('age_group', 'Age_60_plus')}"
    if age_col in feature_vector.columns:
        feature_vector[age_col] = 1
    
    # Primary diagnosis
    diag_col = f"diag_1_grouped_{user_input.get('primary_diagnosis', 'Diabetes')}"
    if diag_col in feature_vector.columns:
        feature_vector[diag_col] = 1
    
    # HbA1c category
    hba1c_col = f"HbA1c_category_{user_input.get('hba1c_category', 'No_HbA1c_Test')}"
    if hba1c_col in feature_vector.columns:
        feature_vector[hba1c_col] = 1
    
    # Diabetes medication
    if 'diabetesMed_Yes' in feature_vector.columns:
        feature_vector['diabetesMed_Yes'] = 1 if user_input.get('diabetes_med', 'Yes') == 'Yes' else 0
    
    # Interaction features - try to set the most likely one
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    interaction_col = f"HbA1c_Diag_interaction_{hba1c_cat}_{primary_diag}"
    if interaction_col in feature_vector.columns:
        feature_vector[interaction_col] = 1
    
    return feature_vector

def preprocess_input(user_input, preprocessing_pipeline):
    """
    Preprocess user input using the saved preprocessing pipeline
    
    Args:
        user_input: Dictionary of user inputs
        preprocessing_pipeline: Loaded preprocessing objects
    
    Returns:
        Preprocessed feature vector ready for prediction
    """
    
    # Get feature names
    feature_names = preprocessing_pipeline['feature_names']
    
    # Create feature vector
    feature_vector = create_feature_vector(user_input, feature_names)
    
    # Apply scaling (scaler was already fit during training)
    scaler = preprocessing_pipeline['scaler']
    
    # Only scale numerical features (not one-hot encoded ones)
    numerical_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                         'num_medications', 'number_emergency', 'number_inpatient',
                         'number_outpatient', 'num_medications_prescribed',
                         'admission_type_id', 'discharge_disposition_id', 'admission_source_id']
    
    # Check which numerical features exist in the dataframe
    existing_numerical = [f for f in numerical_features if f in feature_vector.columns]
    
    if existing_numerical:
        feature_vector[existing_numerical] = scaler.transform(feature_vector[existing_numerical])
    
    return feature_vector