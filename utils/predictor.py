import numpy as np

def predict_readmission(model_data, feature_vector, model_name):
    """
    Make prediction using the selected model
    
    Args:
        model_data: Dictionary containing model and metadata
        feature_vector: Preprocessed input features
        model_name: Name of the model ('logistic_regression', 'xgboost', 'neural_network')
    
    Returns:
        Dictionary with prediction results
    """
    
    model = model_data['model']
    metadata = model_data['metadata']
    
    try:
        if model_name == 'neural_network':
            # Neural network returns probability directly
            probability = float(model.predict(feature_vector, verbose=0)[0][0])
            prediction = 1 if probability > 0.5 else 0
        else:
            # Scikit-learn models
            prediction = int(model.predict(feature_vector)[0])
            probability = float(model.predict_proba(feature_vector)[0][1])
            
            # Apply optimal threshold for XGBoost
            if model_name == 'xgboost' and 'optimal_threshold' in metadata:
                optimal_threshold = metadata['optimal_threshold']
                prediction = 1 if probability >= optimal_threshold else 0
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "High"
            risk_color = "red"
        
        return {
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_performance': metadata['performance']
        }
    
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

def get_risk_message(risk_level, probability):
    """Generate a user-friendly risk message"""
    
    messages = {
        "Low": f"This patient has a **LOW risk** ({probability*100:.1f}%) of being readmitted within 30 days. Continue standard post-discharge care protocols.",
        "Medium": f"This patient has a **MODERATE risk** ({probability*100:.1f}%) of being readmitted within 30 days. Consider enhanced follow-up care and medication adherence monitoring.",
        "High": f"This patient has a **HIGH risk** ({probability*100:.1f}%) of being readmitted within 30 days. Recommend intensive case management, early follow-up appointments, and patient education."
    }
    
    return messages.get(risk_level, "Unable to determine risk level")