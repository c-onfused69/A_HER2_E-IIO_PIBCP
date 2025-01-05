import pandas as pd

# Load the dataset
data = pd.read_csv('path_to_your_dataset.csv')

# Summary statistics
print(data.describe())

# Count of HER2 status
her2_counts = data['HER2 Status'].value_counts()
print(her2_counts)

# Visualize HER2 status distribution
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='HER2 Status', data=data)
plt.title('Distribution of HER2 Status')
plt.show()
