# app.py - Flask API for Churn Prediction

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# ---- Load saved model files ----
# These were saved in notebook 03_model_building.ipynb
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
    
    # Step 3: Add missing columns with 0
    # (user might not send all 33 features)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Step 4: Keep only the columns model expects, in right order
    input_df = input_df[feature_names]

    # Step 5: Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Step 6: Get probability
    churn_probability = model.predict_proba(input_scaled)[0][1]
    
    # Step 7: Apply our tuned threshold
    threshold = 0.20
    prediction = int(churn_probability >= threshold)
    
    # Step 8: Return result
    result = {
        'churn_prediction': prediction,
        'churn_probability': round(float(churn_probability), 3),
        'risk_level': 'HIGH' if churn_probability >= 0.5 else 
                      'MEDIUM' if churn_probability >= 0.20 else 'LOW',
        'message': 'Customer likely to churn' if prediction == 1 
                   else 'Customer likely to stay'
    }
    
    return jsonify(result)

# ---- Run the app ----
if __name__ == '__main__':
    app.run(debug=True, port=5000)

    