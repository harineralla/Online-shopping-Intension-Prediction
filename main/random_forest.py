from sklearn.ensemble import RandomForestClassifier
from data_utils import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from decision_tree import confusion_matrix


def confusion_matrix(y, y_pred, fig):
    confusion_matrix = np.zeros((2, 2))
    rows = ["Actual Positive", "Actual Negative"]
    cols = ("Classifier Positive", "Classifier Negative")
    for i in range(len(y_pred)):
        if int(y_pred[i]) == 1 and int(y[i]) == 1:
            confusion_matrix[0, 0] += 1 
        elif int(y_pred[i]) == 1 and int(y[i]) == 0:
            confusion_matrix[1, 0] += 1  
        elif int(y_pred[i]) == 0 and int(y[i]) == 1:
            confusion_matrix[0, 1] += 1  
        elif int(y_pred[i]) == 0 and int(y[i]) == 0:
            confusion_matrix[1, 1] += 1  

    fig.subplots_adjust(left=0.3, top=0.8, wspace=1)
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=2, rowspan=2)
    ax.table(cellText=confusion_matrix.tolist(),
             rowLabels=rows,
             colLabels=cols, loc="upper center")
    ax.axis("off")

def process_random_forest(Xtrn, ytrn, Xtst, ytst, batch_count):
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(Xtrn, ytrn)
    # take predictions on the test data
    predictions = clf.predict(Xtst)

    print("Accuracy:", round(accuracy_score(ytst, predictions), 4))
    print("Precision Score:", round(precision_score(ytst, predictions), 4))
    print("Recall score :", recall_score(ytst, predictions))
    print("F-1 Score:", round(f1_score(ytst, predictions), 4))

    fig1 = plt.figure(1)
    confusion_matrix(ytst, predictions, fig1)
    fig1.suptitle("Random Forest Confusion Matrix - Batch {0}".format(batch_count))
    plt.show()

if __name__ == '__main__':
    Xtrn1, ytrn1 = get_batch_1_1()
    Xtrn2, ytrn2 = get_batch_1_2()
    Xtrn3, ytrn3 = get_batch_1_3()
    Xtrn4, ytrn4 = get_batch_1_4()
    Xtst, ytst = get_test_data()
    process_random_forest(Xtrn1,ytrn1, Xtst,ytst,1)
    process_random_forest(Xtrn2,ytrn2, Xtst,ytst,2)
    process_random_forest(Xtrn3,ytrn3, Xtst,ytst,3)
    process_random_forest(Xtrn4,ytrn4, Xtst,ytst,4)