import numpy as np
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

class SVM:
    def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.tol = tol
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = [0] * n_features
        self.b = 0
        for epoch in range(self.epochs):
            for i in range(n_samples):
                if y[i] * (self.predict(X[i]) - self.b) >= 1:
                    dw = [2 * self.C * w for w in self.w]
                    db = 0
                else:
                    dw = [2 * self.C * w - X[i][j] * y[i] for j, w in enumerate(self.w)]
                    db = y[i]
                self.w = [w - self.lr * d for w, d in zip(self.w, dw)]
                self.b -= self.lr * db
                

                
    def predict(self, X):
        y_pred = np.sign(np.dot(X, self.w) - self.b)
        return y_pred.astype(int)


# # test the SVM
# y_pred = svm.predict(X_test)
# acc = np.sum(y_pred == y_test) / len(y_test)
# print("Test Accuracy: ", acc)

    
def compute_error(y_true, y_pred):
    try:
        n = len(y_true)
        return np.sum(np.abs(y_true-y_pred))/n
    except:
        raise Exception('Function not yet implemented!')
    
if __name__ == '__main__':

    Xtrn, ytrn = get_batch_1_1() # change the batch values fucntion from 1-1 to 1-2... etc

    Xtst, ytst = get_test_data()

    # print("x train ",Xtrn[0:9])
    # print("y train ",ytrn)
    # print("x test ",Xtst[0:9])

    # print("y test ",ytst[0:9])

    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)
    X_test_scaled = scaler.transform(Xtst)

    # train the SVM
    svm = SVM(lr=0.01, C=10, epochs=1000, tol=0.001)

    svm.fit(X_train_scaled, ytrn)
    y_pred = svm.predict(X_test_scaled)
    print("x train scaled",X_train_scaled)
    print("x test scaled",X_test_scaled)

    print("y predicted", y_pred)
    
    test_error = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(test_error*100))
    
    print("Accuracy score from scikit: ",accuracy_score(ytst, y_pred))



    # # Train the SVM model
    # svm = LinearSVC(C=1, max_iter=10000)
    # svm.fit(X_train_scaled, ytrn)

    # # Make predictions on the test set
    # y_pred = svm.predict(X_test_scaled)

    # acc = accuracy_score(ytst, y_pred)
    # print("Test Accuracy: ", acc)
 
