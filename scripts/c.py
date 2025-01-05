import pandas as pd
# Visualize HER2 status distribution
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

from lifelines import KaplanMeierFitter

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