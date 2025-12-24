import pandas as pd
import numpy as np

def create_feature_vector(user_input, feature_names):
    """
    Create a complete feature vector with ALL features the model expects
    
    Args:
        user_input: Dictionary of user inputs
        feature_names: List of ALL feature names expected by the model (116 features)
    
    Returns:
        pandas DataFrame with all 116 features
    """
    
    # Initialize ALL features with 0 (this is critical!)
    feature_vector = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # ========================================================================
    # NUMERICAL FEATURES (Direct from user input)
    # ========================================================================
    
    numerical_mappings = {
        'time_in_hospital': user_input.get('time_in_hospital', 4),
        'num_lab_procedures': user_input.get('num_lab_procedures', 45),
        'num_procedures': user_input.get('num_procedures', 2),
        'num_medications': user_input.get('num_medications', 15),
        'number_emergency': user_input.get('number_emergency', 0),
        'number_inpatient': user_input.get('number_inpatient', 0),
        'number_outpatient': user_input.get('number_outpatient', 0),
        'num_medications_prescribed': user_input.get('num_medications_prescribed', 15),
        'admission_type_id': user_input.get('admission_type_id', 1),
        'discharge_disposition_id': user_input.get('discharge_disposition_id', 1),
        'admission_source_id': user_input.get('admission_source_id', 7),
        'medical_specialty': user_input.get('medical_specialty', 0),
        'number_diagnoses': user_input.get('number_diagnoses', 9),
        'num_medications_changed': user_input.get('num_medications_changed', 0)
    }
    
    for feature_name, value in numerical_mappings.items():
        if feature_name in feature_vector.columns:
            feature_vector[feature_name] = value
    
    # ========================================================================
    # ONE-HOT ENCODED FEATURES
    # ========================================================================
    
    # Gender (one-hot)
    gender = user_input.get('gender', 'Male')
    gender_cols = [col for col in feature_vector.columns if col.startswith('gender_')]
    for col in gender_cols:
        if gender in col:
            feature_vector[col] = 1
    
    # Race (one-hot)
    race = user_input.get('race', 'Caucasian')
    race_cols = [col for col in feature_vector.columns if col.startswith('race_')]
    for col in race_cols:
        if race in col:
            feature_vector[col] = 1
    
    # Age group (one-hot)
    age_group = user_input.get('age_group', 'Age_60_plus')
    age_cols = [col for col in feature_vector.columns if col.startswith('age_group_')]
    for col in age_cols:
        if age_group in col:
            feature_vector[col] = 1
    
    # Primary diagnosis (one-hot)
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    diag1_cols = [col for col in feature_vector.columns if col.startswith('diag_1_grouped_')]
    for col in diag1_cols:
        if primary_diag in col:
            feature_vector[col] = 1
    
    # Secondary diagnosis (default: same as primary)
    diag2_cols = [col for col in feature_vector.columns if col.startswith('diag_2_grouped_')]
    for col in diag2_cols:
        if primary_diag in col:
            feature_vector[col] = 1
    
    # Tertiary diagnosis (default: same as primary)
    diag3_cols = [col for col in feature_vector.columns if col.startswith('diag_3_grouped_')]
    for col in diag3_cols:
        if primary_diag in col:
            feature_vector[col] = 1
    
    # HbA1c category (one-hot)
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    hba1c_cols = [col for col in feature_vector.columns if col.startswith('HbA1c_category_')]
    for col in hba1c_cols:
        if hba1c_cat in col:
            feature_vector[col] = 1
    
    # Diabetes medication (one-hot)
    diabetes_med = user_input.get('diabetes_med', 'Yes')
    if 'diabetesMed_Yes' in feature_vector.columns:
        feature_vector['diabetesMed_Yes'] = 1 if diabetes_med == 'Yes' else 0
    
    # Payer code (default to one common value)
    payer_cols = [col for col in feature_vector.columns if col.startswith('payer_code_')]
    if payer_cols and 'payer_code_MC' in feature_vector.columns:
        feature_vector['payer_code_MC'] = 1  # Medicare (most common)
    
    # ========================================================================
    # MEDICATION BINARY FEATURES (all set to 0 by default)
    # ========================================================================
    # These are already initialized to 0, so we just leave them
    # unless you want to let users specify specific medications
    
    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    
    # HbA1c × Primary Diagnosis interaction
    interaction_cols = [col for col in feature_vector.columns if col.startswith('HbA1c_Diag_interaction_')]
    target_interaction = f"HbA1c_Diag_interaction_{hba1c_cat}_{primary_diag}"
    
    for col in interaction_cols:
        if target_interaction == col:
            feature_vector[col] = 1
            break
    
    # ========================================================================
    # BOOLEAN INTERACTION FEATURES
    # ========================================================================
    
    # long_stay_high_procedures
    if 'long_stay_high_procedures' in feature_vector.columns:
        long_stay = user_input.get('time_in_hospital', 4) > 7
        high_proc = user_input.get('num_procedures', 2) > 3
        feature_vector['long_stay_high_procedures'] = 1 if (long_stay and high_proc) else 0
    
    # elderly_polypharmacy
    if 'elderly_polypharmacy' in feature_vector.columns:
        is_elderly = user_input.get('age_group', 'Age_60_plus') == 'Age_60_plus'
        many_meds = user_input.get('num_medications_prescribed', 15) >= 5
        feature_vector['elderly_polypharmacy'] = 1 if (is_elderly and many_meds) else 0
    
    return feature_vector


def preprocess_input(user_input, preprocessing_pipeline):
    """
    Preprocess user input using the saved preprocessing pipeline
    
    Args:
        user_input: Dictionary of user inputs
        preprocessing_pipeline: Loaded preprocessing objects
    
    Returns:
        Preprocessed feature vector ready for prediction (116 features)
    """
    
    # Get ALL feature names from the saved pipeline
    feature_names = preprocessing_pipeline['feature_names']
    
    print(f"DEBUG: Model expects {len(feature_names)} features")
    
    # Create complete feature vector with ALL features
    feature_vector = create_feature_vector(user_input, feature_names)
    
    print(f"DEBUG: Created feature vector with {len(feature_vector.columns)} features")
    
    # Apply scaling ONLY to numerical features (not one-hot encoded)
    scaler = preprocessing_pipeline['scaler']
    
    # List of numerical features that need scaling
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_emergency', 'number_inpatient',
        'number_outpatient', 'num_medications_prescribed',
        'admission_type_id', 'discharge_disposition_id', 
        'admission_source_id', 'medical_specialty', 'number_diagnoses',
        'num_medications_changed'
    ]
    
    # Only scale features that exist in the dataframe
    existing_numerical = [f for f in numerical_features if f in feature_vector.columns]
    
    if existing_numerical:
        # Scale numerical features
        feature_vector[existing_numerical] = scaler.transform(feature_vector[existing_numerical])
    
    print(f"DEBUG: Final feature vector shape: {feature_vector.shape}")
    print(f"DEBUG: Feature vector columns: {feature_vector.columns.tolist()[:10]}...")  # Show first 10
    
    return feature_vector