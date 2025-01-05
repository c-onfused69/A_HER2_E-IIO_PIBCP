import pandas as pd

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing HER2 status or survival data
data = data.dropna(subset=['HER2 Status', 'Overall Survival (Months)', 'Overall Survival Status'])