import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %matplotlib inline
#---------------------------------------------
# def dataPreprocessing(data):
# Step 1: Import the Data
data = pd.read_csv('online_shoppers_intention.csv')


# Step 2: Explore the Data
print(data.info())
print(data.describe())
print(data['Revenue'].value_counts())

# Step 3: Check for Missing Values
# print("Total missing values:", data.isnull().sum())
# Remove missing values
# data = data.dropna()

# Step 4: Deal with Categorical Features
# cat_features = ['Month', 'VisitorType', 'Weekend']
# data = pd.get_dummies(data, columns=cat_features)
# print(data.columns)

#or # Encode categorical variables
cat_cols = ['Month', 'VisitorType', 'Weekend']
encoder = LabelEncoder()
for col in cat_cols:
    data[col] = encoder.fit_transform(data[col])

# Step 5: Scale and Normalize Numerical Features
num_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
scaler = StandardScaler()
data[num_features] = scaler.fit_transform(data[num_features])

# Step 6: Encode the Target Variable
# data['Revenue'] = data['Revenue'].map({'TRUE': 1, 'FALSE': 0})

# Step 7: Visualize the Data
sns.countplot(x='Revenue', data=data)
plt.show()

sns.pairplot(data, hue='Revenue')
plt.show()


"""

# Feature selection: We can use SelectKBest from scikit-learn to select the best features
# based on their scores in a statistical test. Here is an example code:

# Split the data into features and target
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Select the 15 best features
kbest = SelectKBest(score_func=f_classif, k=15)
X_new = kbest.fit_transform(X, y)
selected_features = X.columns[kbest.get_support()]

"""


# perform feature engineering (code from previous step)

# create X (features) and y (target variable) variables
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a list of models to evaluate
models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('GB', GradientBoostingClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append([acc, prec, rec, f1])
    names.append(name)

# print out the results
for i in range(len(names)):
    print(names[i], "Accuracy:", results[i][0])
    print(names[i], "Precision:", results[i][1])
    print(names[i], "Recall:", results[i][2])
    print(names[i], "F1-score:", results[i][3])
    print()
