# app.py - Flask API for Churn Prediction

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ---- Load saved model files ----
model = joblib.load('../models/churn_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
feature_names = joblib.load('../models/feature_names.pkl')

# ---- Create Flask app ----
app = Flask(__name__)

# ---- Health check endpoint ----
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'churn-predictor'})

# ---- Main prediction endpoint ----
@app.route('/predict', methods=['POST'])
def predict():

    # Step 1: Get data from request
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Step 3: Smart defaults for missing columns
    # Based on data analysis — better than defaulting everything to 0
    smart_defaults = {
        'SeniorCitizen': 0,
        'Partner': 0,
        'Dependents': 0,
        'PhoneService': 1,        # most customers have phone
        'PaperlessBilling': 0,
        'gender': 1,
        'tenure': 12,             # assume average-ish tenure
        'MonthlyCharges': 65.0,   # average monthly charge
        'TotalCharges': 0,
        'is_new_customer': 0,
        'total_services': 2,
        'charges_per_tenure': 5.0
    }

    for col in feature_names:
        if col not in input_df.columns:
            default_val = smart_defaults.get(col, 0)
            input_df[col] = default_val

    # Step 4: Keep only columns model expects, in right order
    input_df = input_df[feature_names]

    # Step 5: Scale the input
    input_scaled = scaler.transform(input_df)

    # Step 6: Get probability
    churn_probability = model.predict_proba(input_scaled)[0][1]

    # Step 7: Apply our tuned threshold
    threshold = 0.20
    prediction = int(churn_probability >= threshold)

    # Step 8: Track what was provided vs defaulted
    provided_features = list(data.keys())
    defaulted_features = [col for col in feature_names
                          if col not in data.keys()]

    # Step 9: Return result
    result = {
        'churn_prediction': prediction,
        'churn_probability': round(float(churn_probability), 3),
        'risk_level': 'HIGH' if churn_probability >= 0.5 else
                      'MEDIUM' if churn_probability >= 0.20 else 'LOW',
        'message': 'Customer likely to churn' if prediction == 1
                   else 'Customer likely to stay',
        'features_provided': len(provided_features),
        'features_defaulted': len(defaulted_features),
        'warning': 'Partial data - result may be less accurate'
                   if len(defaulted_features) > 20 else None
    }

    return jsonify(result)

# ---- Run the app ----
if __name__ == '__main__':
    app.run(debug=True, port=5000)