import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model/lgbm_model.pkl')

# Create a sample input with all required features
sample_input = pd.DataFrame({
    'Age': [35],
    'DailyRate': [500],
    'DistanceFromHome': [10],
    'Education': [2],
    'EmployeeCount': [1],
    'EmployeeNumber': [123],
    'EnvironmentSatisfaction': [3],
    'HourlyRate': [50],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobSatisfaction': [3],
    'MonthlyIncome': [5000],
    'MonthlyRate': [15000],
    'NumCompaniesWorked': [2],
    'PercentSalaryHike': [15],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StandardHours': [80],
    'StockOptionLevel': [1],
    'TotalWorkingYears': [10],
    'TrainingTimesLastYear': [2],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [5],
    'YearsInCurrentRole': [3],
    'YearsSinceLastPromotion': [2],
    'YearsWithCurrManager': [4],
    'BusinessTravel_Travel_Frequently': [0],
    'BusinessTravel_Travel_Rarely': [1],
    'Department_Research & Development': [1],
    'Department_Sales': [0],
    'EducationField_Life Sciences': [1],
    'EducationField_Marketing': [0],
    'EducationField_Medical': [0],
    'EducationField_Other': [0],
    'EducationField_Technical Degree': [0],
    'Gender_Male': [1],
    'JobRole_Human Resources': [0],
    'JobRole_Laboratory Technician': [0],
    'JobRole_Manager': [0],
    'JobRole_Manufacturing Director': [0],
    'JobRole_Research Director': [0],
    'JobRole_Research Scientist': [1],
    'JobRole_Sales Executive': [0],
    'JobRole_Sales Representative': [0],
    'MaritalStatus_Married': [1],
    'MaritalStatus_Single': [0],
    'OverTime_Yes': [1]
})

# Calculate engineered features
sample_input['IncomePerLevel'] = sample_input['MonthlyIncome'] / (sample_input['JobLevel'] + 1)
sample_input['PreviousExperience'] = sample_input['TotalWorkingYears'] - sample_input['YearsAtCompany']
sample_input['PreviousExperience'] = sample_input['PreviousExperience'].apply(lambda x: max(x, 0))
sample_input['TenureRatio'] = sample_input['YearsAtCompany'] / (sample_input['TotalWorkingYears'] + 1)

# Make prediction
prediction = model.predict(sample_input)
probability = model.predict_proba(sample_input)

print("Prediction (0=Stay, 1=Leave):", prediction[0])
print("Probability of leaving:", probability[0][1])