from sklearn.ensemble import GradientBoostingClassifier
from data_utils import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,confusion_matrix

Xtrn, ytrn = get_batch_1_4() # change the batch values fucntion from 1-1 to 1-2... etc

Xtst, ytst = get_test_data()

clf = GradientBoostingClassifier(n_estimators=800, 
                                 learning_rate=0.01,
                                 max_depth=6, 
                                 random_state=5).fit(Xtrn, ytrn)
# take predictions on the test data
predictions = clf.predict(Xtst)
# visualize confusion matrix with seaborn heatmap
CM_grb= confusion_matrix(ytst, predictions)

cm_matr = pd.DataFrame(data=CM_grb, columns=['Predict Positive:0', 'Predict Negative:1'], index=['Actual Positive:0', 'Actual Negative:1'])
sns.heatmap(cm_matr, annot=True, fmt='d', cmap='RdYlGn')
plt.yticks(rotation=30)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",round(accuracy_score(ytst, predictions),4))
# Precision Score on the test data
print("Precision Score:",round(precision_score(ytst, predictions),4))
# Precision Score on the test data
print("F-1 Score:",round(f1_score(ytst, predictions),4))

print("Confusion Matrix", CM_grb)