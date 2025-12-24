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
    feature_vector['number_diagnoses'] = 9  # Default
    
    # ========================================================================
    # MEDICATION PRESCRIBED FEATURES (18 medications - all 0 by default)
    # ========================================================================
    # Already initialized to 0, leave as is
    
    # ========================================================================
    # DERIVED FEATURES
    # ========================================================================
    
    feature_vector['num_medications_prescribed'] = user_input.get('num_medications', 15)
    feature_vector['num_medications_changed'] = 0  # Default
    
    # long_stay_high_procedures
    long_stay = user_input.get('time_in_hospital', 4) > 7
    high_proc = user_input.get('num_procedures', 2) > 3
    feature_vector['long_stay_high_procedures'] = 1 if (long_stay and high_proc) else 0
    
    # elderly_polypharmacy
    is_elderly = user_input.get('age_group', 'Age_60_plus') == 'Age_60_plus'
    many_meds = user_input.get('num_medications', 15) >= 5
    feature_vector['elderly_polypharmacy'] = 1 if (is_elderly and many_meds) else 0
    
    # ========================================================================
    # ONE-HOT ENCODED FEATURES
    # ========================================================================
    
    # RACE (only 5 options, no AfricanAmerican)
    race = user_input.get('race', 'Caucasian')
    if race == 'AfricanAmerican':
        race = 'Other'  # Map to Other since AfricanAmerican doesn't exist
    race_col = f'race_{race}'
    if race_col in feature_vector.columns:
        feature_vector[race_col] = 1
    
    # GENDER (only Male and Unknown/Invalid, no Female!)
    gender = user_input.get('gender', 'Male')
    if gender == 'Female':
        # Female was dropped during one-hot encoding, all females are baseline
        pass  # Leave all gender columns as 0
    else:
        gender_col = f'gender_{gender}'
        if gender_col in feature_vector.columns:
            feature_vector[gender_col] = 1
    
    # PAYER CODE (set to MC - Medicare as default)
    if 'payer_code_MC' in feature_vector.columns:
        feature_vector['payer_code_MC'] = 1
    
    # MEDICATION COMBINATION FEATURES (all 0 by default)
    # glimepiride-pioglitazone_Steady, metformin-rosiglitazone_Steady, metformin-pioglitazone_Steady
    # Already 0, leave as is
    
    # DIABETES MED
    diabetes_med = user_input.get('diabetes_med', 'Yes')
    if diabetes_med == 'Yes':
        feature_vector['diabetesMed_Yes'] = 1
    
    # ========================================================================
    # HbA1c CATEGORY (CRITICAL - ONLY 2 OPTIONS!)
    # ========================================================================
    # Available: HbA1c_category_High_HbA1c_NoMedChange, HbA1c_category_Normal_HbA1c
    # Missing: No_HbA1c_Test (baseline), High_HbA1c_MedChanged (doesn't exist!)
    
    hba1c_cat = user_input.get('hba1c_category', 'No_HbA1c_Test')
    
    if hba1c_cat == 'High_HbA1c_NoMedChange':
        feature_vector['HbA1c_category_High_HbA1c_NoMedChange'] = 1
    elif hba1c_cat == 'Normal_HbA1c':
        feature_vector['HbA1c_category_Normal_HbA1c'] = 1
    elif hba1c_cat == 'High_HbA1c_MedChanged':
        # This category doesn't exist in training data, leave as baseline
        pass
    # else: No_HbA1c_Test is baseline (all 0)
    
    # ========================================================================
    # DIAGNOSIS GROUPS (8 options each, Circulatory missing from diag_1!)
    # ========================================================================
    
    primary_diag = user_input.get('primary_diagnosis', 'Diabetes')
    
    # diag_1_grouped (NO Circulatory!)
    if primary_diag != 'Circulatory':
        diag1_col = f'diag_1_grouped_{primary_diag}'
        if diag1_col in feature_vector.columns:
            feature_vector[diag1_col] = 1
    # else: Circulatory doesn't exist in diag_1, map to Other
    else:
        if 'diag_1_grouped_Other' in feature_vector.columns:
            feature_vector['diag_1_grouped_Other'] = 1
    
    # diag_2_grouped (same as primary by default)
    diag2_col = f'diag_2_grouped_{primary_diag}'
    if diag2_col in feature_vector.columns:
        feature_vector[diag2_col] = 1
    
    # diag_3_grouped (same as primary by default)
    diag3_col = f'diag_3_grouped_{primary_diag}'
    if diag3_col in feature_vector.columns:
        feature_vector[diag3_col] = 1
    
    # ========================================================================
    # AGE GROUP (ONLY 2 OPTIONS!)
    # ========================================================================
    # Available: age_group_Age_30_60, age_group_Age_60_plus
    # Missing: Age_0_30 (baseline)
    
    age_group = user_input.get('age_group', 'Age_60_plus')
    
    if age_group == 'Age_30_60':
        feature_vector['age_group_Age_30_60'] = 1
    elif age_group == 'Age_60_plus':
        feature_vector['age_group_Age_60_plus'] = 1
    # else: Age_0_30 is baseline (all 0)
    
    # ========================================================================
    # INTERACTION FEATURES (24 total)
    # ========================================================================
    
    # Determine interaction based on HbA1c category and diagnosis
    
    if hba1c_cat == 'High_HbA1c_MedChanged':
        # Pattern: HbA1c_Diag_interaction_High_HbA1c_MedChanged_{Diagnosis}
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_MedChanged_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1
    
    elif hba1c_cat == 'High_HbA1c_NoMedChange':
        # Pattern: HbA1c_Diag_interaction_High_HbA1c_NoMedChange_{Diagnosis}
        interaction_col = f'HbA1c_Diag_interaction_High_HbA1c_NoMedChange_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1
    
    elif hba1c_cat == 'Normal_HbA1c':
        # Pattern: HbA1c_Diag_interaction_Normal_HbA1c_{Diagnosis}
        interaction_col = f'HbA1c_Diag_interaction_Normal_HbA1c_{primary_diag}'
        if interaction_col in feature_vector.columns:
            feature_vector[interaction_col] = 1
    
    # else: No_HbA1c_Test - no interaction feature set
    
    # ========================================================================
    # SCALING (ONLY numerical features, not one-hot encoded)
    # ========================================================================
    
    scaler = preprocessing_pipeline['scaler']
    
    numerical_features = [
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'medical_specialty', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'number_diagnoses'
    ]
    
    # Scale only these numerical features
    feature_vector[numerical_features] = scaler.transform(feature_vector[numerical_features])
    
    return feature_vector