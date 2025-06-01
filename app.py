from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = os.path.join('model', 'lgbm_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')  # Or just return a message for API

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        if request.is_json:
            input_data = request.get_json()
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame([{
                'Age': float(request.form['age']),
                'MonthlyIncome': float(request.form['income']),
                'OverTime': request.form['overtime'],
                'JobLevel': float(request.form['job_level']),
                'TotalWorkingYears': float(request.form['total_working_years']),
                'YearsAtCompany': float(request.form['years_at_company']),
                'YearsInCurrentRole': float(request.form['years_in_role']),
                'JobSatisfaction': float(request.form['job_satisfaction']),
                'WorkLifeBalance': float(request.form['work_life_balance']),
                'EnvironmentSatisfaction': float(request.form['environment_satisfaction'])
            }])

        # Calculate engineered features
        df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
        df['PreviousExperience'] = df['TotalWorkingYears'] - df['YearsAtCompany']
        df['PreviousExperience'] = df['PreviousExperience'].apply(lambda x: max(x, 0))
        df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

        # Convert categorical variables
        df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'attrition_probability': round(float(probability), 3),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
