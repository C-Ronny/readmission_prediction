import pandas as pd
import numpy as np

def preprocess_input(user_input, preprocessing_pipeline):
    """
    Create feature vector matching EXACTLY what the model expects
    """
    
    # Get exact feature names from the model
    expected_features = preprocessing_pipeline['feature_names']
    
    # Initialize with zeros for ALL expected features
    feature_vector = pd.DataFrame(0.0, index=[0], columns=expected_features)
    
    # Map user inputs to the exact feature names
    # IMPORTANT: Only set features that actually exist in expected_features
    
    # Numerical features
    num_mappings = {
        'time_in_hospital': user_input.get('time_in_hospital', 4),
        'num_lab_procedures': user_input.get('num_lab_procedures', 45),
        'num_procedures': user_input.get('num_procedures', 2),
        'num_medications': user_input.get('num_medications', 15),
        'number_emergency': user_input.get('number_emergency', 0),
        'number_inpatient': user_input.get('number_inpatient', 0),
        'number_outpatient': user_input.get('number_outpatient', 0),
        'admission_type_id': user_input.get('admission_type_id', 1),
        'discharge_disposition_id': user_input.get('discharge_disposition_id', 1),
        'admission_source_id': user_input.get('admission_source_id', 7),
        'number_diagnoses': 9,
        'num_medications_prescribed': user_input.get('num_medications', 15),
        'num_medications_changed': 0
    }
    
    for feat, val in num_mappings.items():
        if feat in expected_features:
            feature_vector[feat] = val
    
    # One-hot encoded features - only set to 1 if the exact column exists
    
    # Gender
    gender_col = f"gender_{user_input.get('gender', 'Male')}"
    if gender_col in expected_features:
        feature_vector[gender_col] = 1
    
    # Race  
    race_col = f"race_{user_input.get('race', 'Caucasian')}"
    if race_col in expected_features:
        feature_vector[race_col] = 1
    
    # Age
    age_col = f"age_group_{user_input.get('age_group', 'Age_60_plus')}"
    if age_col in expected_features:
        feature_vector[age_col] = 1
    
    # Diagnosis
    diag_col = f"diag_1_grouped_{user_input.get('primary_diagnosis', 'Diabetes')}"
    if diag_col in expected_features:
        feature_vector[diag_col] = 1
    
    # Diag 2 (default same as primary)
    diag2_col = f"diag_2_grouped_{user_input.get('primary_diagnosis', 'Diabetes')}"
    if diag2_col in expected_features:
        feature_vector[diag2_col] = 1
    
    # Diag 3 (default same as primary)
    diag3_col = f"diag_3_grouped_{user_input.get('primary_diagnosis', 'Diabetes')}"
    if diag3_col in expected_features:
        feature_vector[diag3_col] = 1
    
    # HbA1c
    hba1c_col = f"HbA1c_category_{user_input.get('hba1c_category', 'No_HbA1c_Test')}"
    if hba1c_col in expected_features:
        feature_vector[hba1c_col] = 1
    
    # Diabetes Med
    if 'diabetesMed_Yes' in expected_features:
        feature_vector['diabetesMed_Yes'] = 1 if user_input.get('diabetes_med') == 'Yes' else 0
    
    # Interaction feature
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    interaction_col = f"HbA1c_Diag_interaction_{hba1c_cat}_{primary_diag}"
    if interaction_col in expected_features:
        feature_vector[interaction_col] = 1
    
    # Scale numerical features
    scaler = preprocessing_pipeline['scaler']
    numerical_to_scale = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                          'num_medications', 'number_emergency', 'number_inpatient',
                          'number_outpatient', 'admission_type_id', 'discharge_disposition_id',
                          'admission_source_id', 'number_diagnoses']
    
    existing_num = [f for f in numerical_to_scale if f in expected_features]
    if existing_num:
        feature_vector[existing_num] = scaler.transform(feature_vector[existing_num])
    
    return feature_vector