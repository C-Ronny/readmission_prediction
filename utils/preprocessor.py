import pandas as pd
import numpy as np

def preprocess_input(user_input, preprocessing_pipeline):
    """
    Create feature vector with EXACT 116 features matching training data
    """
    
    # Get exact feature names from saved pipeline
    feature_names = preprocessing_pipeline['feature_names']
    
    # Initialize ALL 116 features with 0
    feature_vector = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # ========================================================================
    # NUMERICAL FEATURES (first 12 features)
    # ========================================================================
    
    feature_vector['admission_type_id'] = user_input.get('admission_type_id', 1)
    feature_vector['discharge_disposition_id'] = user_input.get('discharge_disposition_id', 1)
    feature_vector['admission_source_id'] = user_input.get('admission_source_id', 7)
    feature_vector['time_in_hospital'] = user_input.get('time_in_hospital', 4)
    feature_vector['medical_specialty'] = user_input.get('medical_specialty', 0)
    feature_vector['num_lab_procedures'] = user_input.get('num_lab_procedures', 45)
    feature_vector['num_procedures'] = user_input.get('num_procedures', 2)
    feature_vector['num_medications'] = user_input.get('num_medications', 15)
    feature_vector['number_outpatient'] = user_input.get('number_outpatient', 0)
    feature_vector['number_emergency'] = user_input.get('number_emergency', 0)
    feature_vector['number_inpatient'] = user_input.get('number_inpatient', 0)
    feature_vector['number_diagnoses'] = 9
    
    # Medication prescribed features (already 0)
    # num_medications_prescribed and num_medications_changed
    feature_vector['num_medications_prescribed'] = user_input.get('num_medications', 15)
    feature_vector['num_medications_changed'] = 0
    
    # Interaction features
    long_stay = user_input.get('time_in_hospital', 4) > 7
    high_proc = user_input.get('num_procedures', 2) > 3
    feature_vector['long_stay_high_procedures'] = 1.0 if (long_stay and high_proc) else 0.0
    
    is_elderly = user_input.get('age_group', 'Age_60_plus') == 'Age_60_plus'
    many_meds = user_input.get('num_medications', 15) >= 5
    feature_vector['elderly_polypharmacy'] = 1.0 if (is_elderly and many_meds) else 0.0
    
    # ========================================================================
    # ONE-HOT ENCODED FEATURES
    # ========================================================================
    
    # RACE
    race = user_input.get('race', 'Caucasian')
    if race == 'AfricanAmerican':
        race = 'Other'
    race_col = f'race_{race}'
    if race_col in feature_vector.columns:
        feature_vector[race_col] = 1.0
    
    # GENDER
    gender = user_input.get('gender', 'Male')
    if gender != 'Female':  # Female is baseline
        gender_col = f'gender_{gender}'
        if gender_col in feature_vector.columns:
            feature_vector[gender_col] = 1.0
    
    # PAYER CODE
    if 'payer_code_MC' in feature_vector.columns:
        feature_vector['payer_code_MC'] = 1.0
    
    # DIABETES MED
    if user_input.get('diabetes_med', 'Yes') == 'Yes':
        feature_vector['diabetesMed_Yes'] = 1.0
    
    # HbA1c CATEGORY
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    if hba1c_cat == 'High_HbA1c_NoMedChange' and 'HbA1c_category_High_HbA1c_NoMedChange' in feature_vector.columns:
        feature_vector['HbA1c_category_High_HbA1c_NoMedChange'] = 1.0
    elif hba1c_cat == 'Normal_HbA1c' and 'HbA1c_category_Normal_HbA1c' in feature_vector.columns:
        feature_vector['HbA1c_category_Normal_HbA1c'] = 1.0
    
    # DIAGNOSIS GROUPS
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    
    # diag_1
    if primary_diag == 'Circulatory':
        if 'diag_1_grouped_Other' in feature_vector.columns:
            feature_vector['diag_1_grouped_Other'] = 1.0
    else:
        diag1_col = f'diag_1_grouped_{primary_diag}'
        if diag1_col in feature_vector.columns:
            feature_vector[diag1_col] = 1.0
    
    # diag_2
    diag2_col = f'diag_2_grouped_{primary_diag}'
    if diag2_col in feature_vector.columns:
        feature_vector[diag2_col] = 1.0
    
    # diag_3
    diag3_col = f'diag_3_grouped_{primary_diag}'
    if diag3_col in feature_vector.columns:
        feature_vector[diag3_col] = 1.0
    
    # AGE GROUP
    age_group = user_input.get('age_group', 'Age_60_plus')
    if age_group == 'Age_30_60' and 'age_group_Age_30_60' in feature_vector.columns:
        feature_vector['age_group_Age_30_60'] = 1.0
    elif age_group == 'Age_60_plus' and 'age_group_Age_60_plus' in feature_vector.columns:
        feature_vector['age_group_Age_60_plus'] = 1.0
    
    # INTERACTION FEATURES
    if hba1c_cat == 'High_HbA1c_MedChanged':
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_MedChanged_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1.0
    elif hba1c_cat == 'High_HbA1c_NoMedChange':
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_NoMedChange_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1.0
    elif hba1c_cat == 'Normal_HbA1c':
        interaction_col = f'HbA1c_Diag_interaction_Normal_HbA1c_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1.0
    
    # ========================================================================
    # CRITICAL FIX: Pass ALL 116 features to scaler in EXACT order
    # ========================================================================
    
    scaler = preprocessing_pipeline['scaler']
    
    # The scaler was fit on ALL features, so we must transform ALL features
    # Pass the entire dataframe in the exact column order
    feature_vector_scaled = pd.DataFrame(
        scaler.transform(feature_vector),
        columns=feature_names
    )
    
    return feature_vector_scaled