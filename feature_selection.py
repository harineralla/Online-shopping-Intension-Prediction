import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the data into a pandas DataFrame
data = pd.read_csv('./train_data.csv')

# Split the data into positive and negative samples based on the values in the 'column_name' column
positive_samples = data[data['Revenue'] == 1]
negative_samples = data[data['Revenue'] == 0]

# # Save the positive and negative samples to separate Excel sheets
# with pd.ExcelWriter('./sampled_data.xlsx') as writer:
#     positive_samples.to_excel(writer, sheet_name='Positive Samples', index=False)
#     negative_samples.to_excel(writer, sheet_name='Negative Samples', index=False)


## correlation for feature selection
# Convert the 'Revenue' column to binary 1/0 labels
data['Revenue'] = data['Revenue'].map({True: 1, False: 0})

# Calculate the Pearson correlation coefficients between the features
corr = data.corr(method='pearson')

# Create a heatmap using Seaborn
sns.set(style='white')
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Pearson Correlation Coefficients')
plt.show()