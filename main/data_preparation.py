from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data into a pandas DataFrame
data = pd.read_csv('./online_shoppers_intention.csv')

str_columns = ["Month", "VisitorType", "Weekend"]
data = pd.get_dummies(data, columns=str_columns)

# Convert the 'Revenue' column to binary 1/0 labels
data['Revenue'] = data['Revenue'].map({True: 1, False: 0})

# Split the data into features and target
X = data.drop('Revenue', axis=1)
y = data['Revenue']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the train and test data to a CSV file
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)
