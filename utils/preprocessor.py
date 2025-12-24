import pandas as pd
import numpy as np

def preprocess_input(user_input, preprocessing_pipeline):
    """
    Create feature vector with EXACT 116 features matching training data
    Returns numpy array in exact feature order
    """
    
    # Get exact feature names from saved pipeline
    feature_names = preprocessing_pipeline['feature_names']
    
    # Initialize ALL 116 features with 0
    feature_vector = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # ========================================================================
    # FILL IN ALL FEATURE VALUES
    # ========================================================================
    
    # Numerical features
    feature_vector.loc[0, 'admission_type_id'] = float(user_input.get('admission_type_id', 1))
    feature_vector.loc[0, 'discharge_disposition_id'] = float(user_input.get('discharge_disposition_id', 1))
    feature_vector.loc[0, 'admission_source_id'] = float(user_input.get('admission_source_id', 7))
    feature_vector.loc[0, 'time_in_hospital'] = float(user_input.get('time_in_hospital', 4))
    feature_vector.loc[0, 'medical_specialty'] = float(user_input.get('medical_specialty', 0))
    feature_vector.loc[0, 'num_lab_procedures'] = float(user_input.get('num_lab_procedures', 45))
    feature_vector.loc[0, 'num_procedures'] = float(user_input.get('num_procedures', 2))
    feature_vector.loc[0, 'num_medications'] = float(user_input.get('num_medications', 15))
    feature_vector.loc[0, 'number_outpatient'] = float(user_input.get('number_outpatient', 0))
    feature_vector.loc[0, 'number_emergency'] = float(user_input.get('number_emergency', 0))
    feature_vector.loc[0, 'number_inpatient'] = float(user_input.get('number_inpatient', 0))
    feature_vector.loc[0, 'number_diagnoses'] = 9.0
    
    # Medication prescribed features (all stay 0 - already initialized)
    
    feature_vector.loc[0, 'num_medications_prescribed'] = float(user_input.get('num_medications', 15))
    feature_vector.loc[0, 'num_medications_changed'] = 0.0
    
    # Derived features
    long_stay = user_input.get('time_in_hospital', 4) > 7
    high_proc = user_input.get('num_procedures', 2) > 3
    feature_vector.loc[0, 'long_stay_high_procedures'] = 1.0 if (long_stay and high_proc) else 0.0
    
    is_elderly = user_input.get('age_group', 'Age_60_plus') == 'Age_60_plus'
    many_meds = user_input.get('num_medications', 15) >= 5
    feature_vector.loc[0, 'elderly_polypharmacy'] = 1.0 if (is_elderly and many_meds) else 0.0
    
    # Race
    race = user_input.get('race', 'Caucasian')
    if race == 'AfricanAmerican':
        race = 'Other'
    race_col = f'race_{race}'
    if race_col in feature_vector.columns:
        feature_vector.loc[0, race_col] = 1.0
    
    # Gender (Female is baseline - no column for it)
    gender = user_input.get('gender', 'Male')
    if gender != 'Female':
        gender_col = f'gender_{gender}'
        if gender_col in feature_vector.columns:
            feature_vector.loc[0, gender_col] = 1.0
    
    # Payer code (default Medicare)
    if 'payer_code_MC' in feature_vector.columns:
        feature_vector.loc[0, 'payer_code_MC'] = 1.0
    
    # Diabetes medication
    if user_input.get('diabetes_med', 'Yes') == 'Yes':
        feature_vector.loc[0, 'diabetesMed_Yes'] = 1.0
    
    # HbA1c category
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    if hba1c_cat == 'High_HbA1c_NoMedChange':
        if 'HbA1c_category_High_HbA1c_NoMedChange' in feature_vector.columns:
            feature_vector.loc[0, 'HbA1c_category_High_HbA1c_NoMedChange'] = 1.0
    elif hba1c_cat == 'Normal_HbA1c':
        if 'HbA1c_category_Normal_HbA1c' in feature_vector.columns:
            feature_vector.loc[0, 'HbA1c_category_Normal_HbA1c'] = 1.0
    
    # Primary diagnosis
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    
    # diag_1 (Circulatory doesn't exist, use Other)
    if primary_diag == 'Circulatory':
        if 'diag_1_grouped_Other' in feature_vector.columns:
            feature_vector.loc[0, 'diag_1_grouped_Other'] = 1.0
    else:
        col = f'diag_1_grouped_{primary_diag}'
        if col in feature_vector.columns:
            feature_vector.loc[0, col] = 1.0
    
    # diag_2
    col = f'diag_2_grouped_{primary_diag}'
    if col in feature_vector.columns:
        feature_vector.loc[0, col] = 1.0
    
    # diag_3
    col = f'diag_3_grouped_{primary_diag}'
    if col in feature_vector.columns:
        feature_vector.loc[0, col] = 1.0
    
    # Age group
    age_group = user_input.get('age_group', 'Age_60_plus')
    if age_group == 'Age_30_60':
        if 'age_group_Age_30_60' in feature_vector.columns:
            feature_vector.loc[0, 'age_group_Age_30_60'] = 1.0
    elif age_group == 'Age_60_plus':
        if 'age_group_Age_60_plus' in feature_vector.columns:
            feature_vector.loc[0, 'age_group_Age_60_plus'] = 1.0
    
    # Interaction features
    interaction_col = None
    if hba1c_cat == 'High_HbA1c_MedChanged':
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_MedChanged_{primary_diag}'
    elif hba1c_cat == 'High_HbA1c_NoMedChange':
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_NoMedChange_{primary_diag}'
    elif hba1c_cat == 'Normal_HbA1c':
        interaction_col = f'HbA1c_Diag_interaction_Normal_HbA1c_{primary_diag}'
    
    if interaction_col and interaction_col in feature_vector.columns:
        feature_vector.loc[0, interaction_col] = 1.0
    
    # ========================================================================
    # SCALING - Only numerical features
    # ========================================================================
    
    scaler = preprocessing_pipeline['scaler']
    
    numerical_features = [
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'medical_specialty', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]
    
    # Extract numerical values, scale, put back
    numerical_values = feature_vector[numerical_features].values
    scaled_values = scaler.transform(numerical_values)
    feature_vector[numerical_features] = scaled_values
    
    # ========================================================================
    # RETURN AS NUMPY ARRAY in exact column order
    # ========================================================================
    
    # This bypasses sklearn's feature name checking
    return feature_vector[feature_names].values