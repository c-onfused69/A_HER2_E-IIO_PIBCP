import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib  # Import joblib for saving the model

# Load the dataset
data = pd.read_csv('D:/Projects/A_HER2_E&IIO_PIBCP/dataset/HER2_Cancer_Data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing HER2 status or survival data
data = data.dropna(subset=['HER2 Status', 'Overall Survival (Months)', 'Overall Survival Status'])

# Handle missing values for other columns
data['Age at Diagnosis'] = data['Age at Diagnosis'].fillna(data['Age at Diagnosis'].median())
data['Tumor Size'] = data['Tumor Size'].fillna(data['Tumor Size'].median())
data['Neoplasm Histologic Grade'] = data['Neoplasm Histologic Grade'].fillna(data['Neoplasm Histologic Grade'].mode()[0])

# Summary statistics
print(data.describe())

# Count of HER2 status
her2_counts = data['HER2 Status'].value_counts()
print(her2_counts)

# Visualize HER2 status distribution
sns.countplot(x='HER2 Status', data=data)
plt.title('Distribution of HER2 Status')
plt.show()

# Convert survival status to binary (1 for deceased, 0 for living)
data['Survival Status'] = data['Overall Survival Status'].apply(lambda x: 1 if 'DECEASED' in x else 0)

# Fit the Kaplan-Meier estimator
kmf = KaplanMeierFitter()

# Plot survival curves for HER2 positive and negative patients
plt.figure(figsize=(10, 6))
for her2_status in data['HER2 Status'].unique():
    mask = data['HER2 Status'] == her2_status
    kmf.fit(data['Overall Survival (Months)'][mask], event_observed=data['Survival Status'][mask], label=her2_status)
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by HER2 Status')
plt.xlabel('Survival Time (Months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()

# Create masks for HER2 positive and negative
her2_positive = data[data['HER2 Status'] == 'Positive']
her2_negative = data[data['HER2 Status'] == 'Negative']

# Perform log-rank test
results = logrank_test(her2_positive['Overall Survival (Months)'], her2_negative['Overall Survival (Months)'],
                        event_observed_A=her2_positive['Survival Status'], event_observed_B=her2_negative['Survival Status'])

print(f'Log-Rank Test p-value: {results.p_value}')

# Prepare features and target variable
features = data[['Age at Diagnosis', 'Tumor Size', 'Neoplasm Histologic Grade', 'HER2 Status']]
features = pd.get_dummies(features, drop_first=True)  # Convert categorical variables to dummy variables
target = data['Survival Status']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Classifier with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print(f'Best parameters: {grid_search.best_params_}')

# Predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'],
                        yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Visualize the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                              display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
importances = best_model.feature_importances_
feature_names = features.columns
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Save the trained model to a file
model_filename = 'random_forest_model.pkl'
joblib.dump(best_model, model_filename)
print(f'Model saved to {model_filename}')