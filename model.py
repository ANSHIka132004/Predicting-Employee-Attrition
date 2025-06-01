import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier  # Changed this import
import os


df = pd.read_csv("HR_data.csv")  


print(df.head())
# Step 2: Understand the data

# 1. Shape of the data
print("Shape of the dataset (rows, columns):", df.shape)

# 2. General info (column types, non-null counts)
print("\n--- Data Info ---")
print(df.info())

# 3. Summary statistics (mean, std, min, max)
print("\n--- Summary Statistics ---")
print(df.describe())

# 4. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 5. Check class distribution for Attrition
print("\n--- Attrition Value Counts ---")
print(df['Attrition'].value_counts())

# 6. Visualize class balance

sns.countplot(x="Attrition", data=df)
plt.title("Number of Employees Who Left vs Stayed")
plt.show()

# STEP 3: Clean and Prepare the Data

# STEP 3: Clean and Prepare the Data

# 1. Convert 'Attrition' to binary (Yes = 1, No = 0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# ======= FEATURE ENGINEERING ADDED HERE =======

# Example feature: Income per Job Level (to normalize income)
df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)

# Example feature: Previous experience (TotalWorkingYears - YearsAtCompany)
df['PreviousExperience'] = df['TotalWorkingYears'] - df['YearsAtCompany']
df['PreviousExperience'] = df['PreviousExperience'].apply(lambda x: max(x, 0))  # no negative values

# Example feature: Tenure Ratio (YearsAtCompany / TotalWorkingYears)
df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)  # +1 to avoid division by zero

# ======= END OF FEATURE ENGINEERING =======

# 2. Convert categorical variables into dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]


# 4. Check shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# STEP 4: Split the Data

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Show the shape of the resulting sets
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_smote, label=y_train_smote)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters without use_label_encoder
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',  # or 'auc' or any metric you prefer
    'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1),  # helps with imbalance
    # other parameters here...
}

# Train the model
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)

# Predict on test data
y_pred_proba = bst.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Confusion matrix and classification report
print("Confusion Matrix (SMOTE):")
print(confusion_matrix(y_test, y_pred))

print("Classification Report (SMOTE):")
print(classification_report(y_test, y_pred))

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model_weighted = xgb.XGBClassifier(
    eval_metric='logloss', 
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model_weighted.fit(X_train, y_train)

y_pred_weighted = model_weighted.predict(X_test)

print("Confusion Matrix (Weighted XGBoost):")
print(confusion_matrix(y_test, y_pred_weighted))

print("\nClassification Report (Weighted XGBoost):")
print(classification_report(y_test, y_pred_weighted))

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', 
    eval_metric='logloss',
    scale_pos_weight=4.94,  # from your class imbalance ratio
    random_state=42
)

# Remove or comment out the previous RandomizedSearchCV section and replace with:

# Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'min_child_weight': [1, 3, 5],  # Added this important parameter
    'scale_pos_weight': [scale_pos_weight]  # Use the calculated class weight
}

# Create XGBoost classifier with basic parameters
base_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',  # Changed to AUC for imbalanced classification
    random_state=42
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring=['accuracy', 'f1', 'roc_auc'],  # Multiple metrics
    refit='f1',  # Optimize for F1 score
    verbose=2
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print results
print("\nBest Parameters:", grid_search.best_params_)
print("Best F1 Score: {:.4f}".format(grid_search.best_score_))

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate final model
print("\nFinal Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the best model (not 'model' which is undefined)
joblib.dump(best_model, 'employee_xgb_model.pkl')

# Load and verify the saved model
loaded_model = joblib.load('employee_xgb_model.pkl')
y_pred_loaded = loaded_model.predict(X_test)

print("\nVerification of Saved Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_loaded))
print("\nClassification Report:\n", classification_report(y_test, y_pred_loaded))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_loaded))

# Print comprehensive model performance metrics
print("\n=== Model Performance Summary ===")
print("\n1. Best Model Parameters from Grid Search:")
print(grid_search.best_params_)

print("\n2. Cross-validation Scores:")
print("Best F1 Score: {:.4f}".format(grid_search.best_score_))
print("Mean ROC-AUC: {:.4f}".format(grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_]))
print("Mean Accuracy: {:.4f}".format(grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]))

print("\n3. Test Set Performance:")
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n4. Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Initialize base models
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    class_weight='balanced',
    random_state=42
)

lgb_model = LGBMClassifier(  # Changed from lgbm.LGBMClassifier to LGBMClassifier
    objective='binary',
    boosting_type='goss',  # Gradient-based One-Side Sampling
    n_estimators=200,
    num_leaves=31,
    max_depth=7,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Use the same weight as XGBoost
      # Handle imbalanced datasets
    random_state=42,
    # Additional parameters for imbalanced data
    verbose=-1,
    metric='auc',  # AUC is better for imbalanced classification
    boost_from_average=False  # Better for imbalanced datasets
)

# Create voting classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lgb', lgb_model)
    ],
    voting='soft'  # Use probability predictions
)

# Train and evaluate each model separately
models = {
    'XGBoost': xgb_model,
    'Random Forest': rf_model,
    'LightGBM': lgb_model,
    'Voting Ensemble': voting_classifier
}

# Compare model performances
print("\n=== Model Comparison ===")
for name, model in models.items():
    # Train model
    model.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1')
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Save the best performing model
best_model = voting_classifier
joblib.dump(best_model, 'employee_ensemble_model.pkl')

# Visualize feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Most Important Features (Random Forest)')
plt.show()

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier

# Define base models
xgb_best = best_model  # From your earlier grid search
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm = LGBMClassifier(random_state=42)

# Ensemble: Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_best), ('rf', rf), ('lgbm', lgbm)],
    voting='soft'  # 'soft' uses predicted probabilities
)

# Fit on training data
voting_clf.fit(X_train, y_train)

# Predict on test data
y_pred_ensemble = voting_clf.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n=== Ensemble Voting Classifier Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ensemble))
print("Classification Report:\n", classification_report(y_test, y_pred_ensemble))

# Update the LightGBM model configuration
lgb_model = LGBMClassifier(  # Changed from lgbm.LGBMClassifier to LGBMClassifier
    objective='binary',
    boosting_type='goss',  # Gradient-based One-Side Sampling
    n_estimators=200,
    num_leaves=31,
    max_depth=7,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Keep only this for imbalance handling
    random_state=42,
    metric='auc',
    verbose=-1
)

# Create a parameter grid for LightGBM
lgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [15, 31, 63],
    'scale_pos_weight': [scale_pos_weight]
}

# Grid search for LightGBM
lgb_grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=lgb_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
print("\nTuning LightGBM hyperparameters...")
lgb_grid_search.fit(X_train_smote, y_train_smote)

# Update the models dictionary with the best LightGBM model
models = {
    'XGBoost': xgb_model,
    'Random Forest': rf_model,
    'LightGBM': lgb_grid_search.best_estimator_,  # Use the best model
    'Voting Ensemble': voting_classifier
}

# Add evaluation metrics specific to imbalanced classification
print("\n=== LightGBM Best Model Performance ===")
lgb_best = lgb_grid_search.best_estimator_
y_pred_lgb = lgb_best.predict(X_test)

print("Best Parameters:", lgb_grid_search.best_params_)
print("Best F1 Score: {:.4f}".format(lgb_grid_search.best_score_))
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred_lgb))

# Visualize LightGBM feature importance
lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': lgb_best.feature_importances_
})
lgb_importance = lgb_importance.sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=lgb_importance)
plt.title('Top 10 Most Important Features (LightGBM)')
plt.tight_layout()
plt.show()

# Save the best LightGBM model
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Save the model
model_path = os.path.join('model', 'lgbm_model.pkl')
joblib.dump(lgb_grid_search.best_estimator_, model_path)
print(f"Model saved to {model_path}")


# Create base models with corrected parameters
xgb_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    class_weight='balanced',
    random_state=42
)

# Fixed LightGBM configuration - removed conflicting parameters
lgb_model = LGBMClassifier(
    objective='binary',
    boosting_type='goss',
    n_estimators=200,
    num_leaves=31,
    max_depth=7,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,  # Remove is_unbalance parameter
    random_state=42,
    metric='auc',
    verbose=-1
)

# Create voting classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lgb', lgb_model)
    ],
    voting='soft'
)

# Train and evaluate models
models = {
    'XGBoost': xgb_model,
    'Random Forest': rf_model,
    'LightGBM': lgb_model,
    'Voting Ensemble': voting_classifier
}

# Model evaluation and saving
print("\n=== Model Comparison ===")
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Save the best model (using voting classifier)
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(voting_classifier, os.path.join('model', 'lgbm_model.pkl'))
print("\nModel saved successfully!")

