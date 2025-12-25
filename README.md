# Hospital Readmission Prediction for Diabetic Patients

Machine learning system for predicting 30-day hospital readmissions in patients with diabetes mellitus. Deployed as an interactive web application using Streamlit.

## Overview

This project develops and compares three machine learning models (Logistic Regression, XGBoost, Neural Network) to predict hospital readmission risk within 30 days of discharge. The final optimized XGBoost model achieved ROC-AUC of 0.688, competitive with published research in this domain.

**Live Demo:** https://readmissionprediction-t7djpqmkfittggvoxgtbsa.streamlit.app/

## Dataset

**Source:** UCI Machine Learning Repository - Diabetes 130-US Hospitals (1999-2008)

**Original Dataset:**
- 101,766 patient encounters
- 130 US hospitals
- 50 features (demographics, diagnoses, medications, lab results)

**Final Analysis Dataset:**
- 69,984 unique patient encounters
- 11.16% readmission rate (severe class imbalance: 7.96:1)
- Train/test split: 81,412 / 20,354 (stratified)

## Key Features

### Data Processing
- ICD-9 diagnosis code grouping (717 codes → 9 categories)
- HbA1c 4-category feature engineering
- 24 medication binary indicators
- Interaction features (HbA1c × Diagnosis)
- Feature expansion: 50 → 116 engineered features

### Models Trained
1. **Logistic Regression** (interpretable baseline)
2. **XGBoost** (best performance)
3. **Neural Network** (3-layer architecture)

### Class Imbalance Handling
- Class weighting (0.56 for majority, 4.47 for minority)
- SMOTE tested but rejected (degraded performance)
- Threshold optimization for XGBoost (0.50 → 0.53)

## Project Structure
```
diabetes-readmission-prediction/
├── notebooks/
│   └── diabetes_readmission_analysis.ipynb    # Complete ML pipeline
├── app/
│   ├── app.py                                  # Streamlit application
│   ├── requirements.txt                        # Python dependencies
│   ├── .streamlit/
│   │   └── config.toml                         # Theme configuration
│   ├── models/                                 # Trained models
│   │   ├── logistic_regression_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── neural_network_model.h5
│   │   ├── preprocessing_pipeline.pkl
│   │   └── *.json                              # Metadata files
│   └── utils/
│       ├── model_loader.py                     # Model loading utilities
│       ├── preprocessor.py                     # Feature engineering
│       └── predictor.py                        # Prediction logic
├── data/
│   └── diabetic_data.csv                       # Original dataset
├── figures/                                    # Visualizations (15+ plots)
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- pip

### Setup
```bash
# Clone repository
git clone https://github.com/C-Ronny/readmission-prediction.git
cd readmission-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app locally
cd app
streamlit run app.py
```

### Dependencies
```
streamlit
pandas
numpy
scikit-learn
xgboost
tensorflow
joblib
plotly
matplotlib
```
