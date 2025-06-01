# Predicting-Employee-Attrition
# 🧠 Employee Attrition Prediction

This is a Flask-based web app that predicts whether an employee will leave the company, using a machine learning model trained on HR data.

---

## 📁 Project Files

- `app.py` – Flask app to run the prediction server.
- `model.py` – Trains the ML models (XGBoost, LightGBM, etc.).
- `test_model.py` – Tests the saved model with sample input.
- `model.html` – Web form for entering employee data.
- `HR_data.csv` – Dataset used for training.
- `model/lgbm_model.pkl` – Trained LightGBM model (used by the Flask app).

---

## 🚀 How to Run

### 1. Install required libraries
```bash
pip install flask pandas numpy scikit-learn xgboost lightgbm imbalanced-learn joblib
Train the model
//python model.py
3. Start the Flask app
//python app.py

 Built With
-Python

-Flask

-XGBoost

-LightGBM

-Pandas / NumPy / Scikit-learn
