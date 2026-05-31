# 🔄 Customer Churn Predictor

> Predicts which telecom customers will leave — before they do.  
> Built following production ML standards with a live REST API.

---

## 🎯 Business Problem

Telecom companies lose 15–25% of customers annually.  
Random discount campaigns waste money on customers who weren't leaving.  
This system identifies **exactly who is at risk** so retention efforts are targeted.

| | Before | After |
|---|---|---|
| Strategy | Offer discounts to everyone | Target predicted churners only |
| Churn caught | ~26% (random) | **84%** (ML-powered) |

---

## 📊 Model Results

| Model | Recall | Precision | Accuracy |
|-------|--------|-----------|----------|
| Logistic Regression (baseline) | 52% | 67% | 80% |
| Random Forest (default) | 49% | 65% | 79% |
| **Random Forest (threshold=0.20)** | **84%** | 48% | 72% |

**Key insight:** Tuning the decision threshold from 0.5 → 0.20  
improved recall from 49% to 84% — hitting our business target.

---

## 🔍 Key Findings From Data

- **`charges_per_tenure`** (feature we engineered) = #1 most important predictor
- Month-to-month contracts churn at **42%** vs only 3% for 2-year contracts
- Customers in first **0–6 months** are at highest risk
- Customers paying **$70+/month** show significantly higher churn

---

## 🚀 API Demo

Start the server and make predictions via REST API:

**High risk customer** (new + expensive plan):
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0}'
```

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

---

## 🏗️ Project Structure
churn-predictor/
├── data/
│   ├── telco_churn.csv                 # raw IBM Telco dataset
│   └── telco_churn_processed.csv       # cleaned + engineered
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA, cleaning, visualizations
│   ├── 02_feature_engineering.ipynb    # feature creation + encoding
│   └── 03_model_building.ipynb         # training, tuning, evaluation
├── src/
│   └── app.py                          # Flask REST API
├── models/
│   ├── churn_model.pkl                 # trained Random Forest
│   ├── scaler.pkl                      # fitted StandardScaler
│   └── feature_names.pkl               # feature order for API
└── requirements.txt

---

## ⚙️ How to Run

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
# Running on http://127.0.0.1:5000
```

**3. Health check:**
```bash
curl http://localhost:5000/health
```

---

## 🧠 Feature Engineering

3 features created from domain knowledge — not in original dataset:

| Feature | Formula | Insight |
|---------|---------|---------|
| `charges_per_tenure` | MonthlyCharges ÷ (tenure+1) | High ratio = feels overcharged = churn risk |
| `is_new_customer` | tenure ≤ 6 → 1 else 0 | First 6 months = highest risk window |
| `total_services` | count of active services | More services = more locked in |

`charges_per_tenure` became **#1 most important feature** in the model.

---

## 🛠️ Tech Stack

`Python 3.13` `Pandas` `NumPy` `Scikit-learn` `Flask` `Matplotlib` `Seaborn` `Joblib`

---

## 📁 Data Source

IBM Telco Customer Churn Dataset — 7,043 customers, 21 features.  
Available publicly via IBM Developer resources.
