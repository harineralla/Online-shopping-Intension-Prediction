from sklearn.linear_model import LogisticRegression
from data_utils import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc


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


def process_logistic_regression(Xtrn, ytrn, Xtst, ytst, batch_count):
    clf = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000).fit(Xtrn, ytrn)
    # take predictions on the test data
    predictions = clf.predict(Xtst)

    print("Accuracy:", round(accuracy_score(ytst, predictions), 4))
    print("Precision Score:", round(precision_score(ytst, predictions), 4))
    print("Recall score :", recall_score(ytst, predictions))
    print("F-1 Score:", round(f1_score(ytst, predictions), 4))

    # calculate learning curve for training set sizes from 10% to 100% in steps of 10%
    train_sizes, train_scores, test_scores = learning_curve(clf, Xtrn, ytrn, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

    # calculate mean and standard deviation of training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot the learning curve
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regression Learning Curve for Batch 3')
    plt.legend(loc='lower right')
    plt.show()

    # plot the ROC curve
    probs = clf.predict_proba(Xtst)
    fpr, tpr, thresholds = roc_curve(ytst, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression Batch {0}'.format(batch_count))
    plt.legend(loc="lower right")
    # plt.show()

    fig1 = plt.figure(2)
    confusion_matrix(ytst, predictions, fig1)
    fig1.suptitle('Logistic Regression Confusion Matrix - Batch {0}'.format(batch_count))
    plt.show()


if __name__ == '__main__':

    Xtrn1, ytrn1 = get_batch_1_1() # please uncomment the function that the batch you want to run
    # Xtrn2, ytrn2 = get_batch_1_2()
    # Xtrn3, ytrn3 = get_batch_1_3()
    # Xtrn4, ytrn4 = get_batch_1_4()
    Xtst, ytst = get_test_data()

    # Initialize SelectKBest with the desired number of features to select
    k = 15  # Number of features to select
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(Xtrn1, ytrn1)
    selected_feature_indices = selector.get_support(indices=True)
    Xtrn1 = Xtrn1[:, selected_feature_indices]
    # w=Xtrn1.shape[1]
    # print("shape after", w)
    Xtst = Xtst[:, selected_feature_indices]

    # Feature scaling Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn1)
    X_test_scaled = scaler.transform(Xtst)
    process_logistic_regression(X_train_scaled,ytrn1, X_test_scaled,ytst,1)
    # process_logistic_regression(Xtrn2,ytrn2, Xtst,ytst,2) # please uncomment the function that the batch you want to run
    # process_logistic_regression(Xtrn3,ytrn3, Xtst,ytst,3)
    # process_logistic_regression(Xtrn4,ytrn4, Xtst,ytst,4)
