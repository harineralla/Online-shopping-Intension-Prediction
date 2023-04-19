import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

# Step 1: Import the Data
data = pd.read_csv('online_shoppers_intention.csv')

# Step 2: Explore the Data
# print(data.info())
# print(data.describe())
# print(data['Revenue'].value_counts())

# Step 3: Check for Missing Values
# print("Total missing values:", data.isnull().sum())

# Step 4: Deal with Categorical Features
cat_features = ['Month', 'VisitorType', 'Weekend']
data = pd.get_dummies(data, columns=cat_features)

# Step 5: Scale and Normalize Numerical Features
# from sklearn.preprocessing import StandardScaler
# num_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
#                 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
# scaler = StandardScaler()
# data[num_features] = scaler.fit_transform(data[num_features])

# Step 6: Encode the Target Variable
# data['Revenue'] = data['Revenue'].astype(int)

# Step 7: Visualize the Data
# sns.countplot(x='Revenue', data=data)
# plt.show()

# sns.pairplot(data, hue='Revenue')
# plt.show()
