# Customer Churn Predictor

A production-grade ML system that predicts telecom customer churn,
enabling proactive retention before customers leave.

## Business Problem

Telecom companies lose 15-25% of customers annually to churn.
This system identifies at-risk customers BEFORE they leave so
retention offers can be made proactively instead of randomly.

**Baseline:** Random discounts to all customers (wasteful)  
**Solution:** Target only predicted churners (efficient + data-driven)

## Results

| Model | Recall | Precision | Accuracy |
|-------|--------|-----------|----------|
| Logistic Regression (baseline) | 52% | 67% | 80% |
| Random Forest (default) | 49% | 65% | 79% |
| Random Forest (tuned threshold=0.20) | **84%** | 48% | 72% |

Final model catches 84% of churners — exceeding our 80% target.

## Key Findings

- `charges_per_tenure` (engineered feature) is the #1 predictor
- Month-to-month contract customers churn at 42% vs 3% for 2-year
- New customers (0-6 months) are at highest risk
- High monthly charges ($70+) strongly correlate with churn

## Project Structure
churn-predictor/
├── data/
│   ├── telco_churn.csv               raw IBM dataset
│   └── telco_churn_processed.csv     cleaned + engineered features
├── notebooks/
│   ├── 01_data_exploration.ipynb     EDA and cleaning
│   ├── 02_feature_engineering.ipynb  feature creation
│   └── 03_model_building.ipynb       model training + evaluation
├── src/
│   └── app.py                        Flask prediction API
├── models/
│   ├── churn_model.pkl               trained Random Forest
│   ├── scaler.pkl                    fitted StandardScaler
│   └── feature_names.pkl             feature order for API
└── requirements.txt

## How to Run

**1. Clone and setup:**
```bash
git clone https://github.com/manvibuilds/churn-predictor.git
cd churn-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Start the API:**
```bash
cd src
python app.py
```

**3. Make a prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0}'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.33,
  "risk_level": "MEDIUM",
  "message": "Customer likely to churn",
  "features_provided": 2,
  "features_defaulted": 31,
  "warning": "Partial data - result may be less accurate"
}
```

## Feature Engineering

Three features created from domain knowledge:

| Feature | Logic | Why It Matters |
|---------|-------|----------------|
| `charges_per_tenure` | MonthlyCharges / (tenure+1) | New customers paying high feel overcharged |
| `is_new_customer` | tenure <= 6 | First 6 months = highest churn risk period |
| `total_services` | sum of active services | More services = more locked in = less likely to leave |

`charges_per_tenure` became the #1 most important feature —
it did not exist in the original dataset.

## Tech Stack

- Python 3.13
- Pandas + NumPy — data processing
- Scikit-learn — ML models
- Matplotlib + Seaborn — visualizations
- Flask — REST API
- Joblib — model persistence

## What I Learned

- Threshold tuning: default 0.5 gave 49% recall,
  tuned 0.20 threshold gave 84% recall
- Feature engineering beat raw features: engineered
  charges_per_tenure became #1 most important feature
- Class imbalance handling with class_weight='balanced'
- Production API design: smart defaults, warnings for
  partial data, honest uncertainty reporting
- Full ML lifecycle: data → features → model → API