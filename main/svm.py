import numpy as np
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

# class SVM:
#     def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001):
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs
#         self.tol = tol
    
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.w = [0] * n_features
#         self.b = 0
#         for epoch in range(self.epochs):
#             for i in range(n_samples):
#                 if y[i] * (self.predict(X[i]) - self.b) >= 1:
#                     dw = [2 * self.C * w for w in self.w]
#                     db = 0
#                 else:
#                     dw = [2 * self.C * w - X[i][j] * y[i] for j, w in enumerate(self.w)]
#                     db = y[i]
#                 self.w = [w - self.lr * d for w, d in zip(self.w, dw)]
#                 self.b -= self.lr * db
                

                
#     def predict(self, X):
#         y_pred = np.sign(np.dot(X, self.w) - self.b)
#         return y_pred.astype(int)


# # test the SVM
# y_pred = svm.predict(X_test)
# acc = np.sum(y_pred == y_test) / len(y_test)
# print("Test Accuracy: ", acc)
# class SVM:

#     def __init__(self, lr=1, C=1, epochs=10000):
#         self.lr = lr
#         self.lambda_param = C
#         self.n_iters = epochs
#         self.w = None
#         self.b = None


#     def fit(self, X, y):
#         n_samples, n_features = X.shape

#         y_ = np.where(y <= 0, -1, 1)

#         self.w = np.zeros(n_features)
#         self.b = 0

#         for _ in range(self.n_iters):
#             for idx, x_i in enumerate(X):
#                 condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
#                 if condition:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w)
#                 else:
#                     self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
#                     self.b -= self.lr * y_[idx]


#     def predict(self, X):
#         approx = np.dot(X, self.w) - self.b
#         return np.sign(approx)

class SVM:
    def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.tol = tol
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize the weight and bias parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Perform gradient descent to optimize the weight and bias parameters
        for epoch in range(self.epochs):
            # Compute the predicted values using the current weight and bias parameters
            y_pred = np.dot(X, self.w) + self.b
            # Compute the hinge loss function for the predicted values
            loss = np.maximum(0, 1 - y * y_pred)
            # Compute the gradient of the hinge loss function
            dW = np.zeros(n_features)
            dB = 0
            for i in range(n_samples):
                if loss[i] == 0:
                    dW += self.w
                else:
                    dW += self.w - self.C * y[i] * X[i]
                    dB += -self.C * y[i]
            # Update the weight and bias parameters
            self.w -= self.lr * dW / n_samples
            self.b -= self.lr * dB / n_samples
    
    def predict(self, X):
        # Compute the predicted values for the input data using the learned weight and bias parameters
        y_pred = np.sign(np.dot(X, self.w) + self.b)
        return y_pred
import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.num_iterations):
            # Compute the hinge loss
            activation = np.dot(X, self.weights) + self.bias
            loss = np.maximum(0, 1 - y * activation)
            cost = self.lambda_param * np.dot(self.weights, self.weights) + np.mean(loss)
            
            # Compute the gradients
            d_weights = np.zeros(n_features)
            d_bias = 0
            for j in range(n_samples):
                if loss[j] == 0:
                    d_weights += self.weights
                else:
                    d_weights += self.lambda_param * self.weights - y[j] * X[j]
                    d_bias += -y[j]
            
            d_weights /= n_samples
            d_bias /= n_samples
            
            # Update the weights and bias
            self.weights -= self.learning_rate * d_weights
            self.bias -= self.learning_rate * d_bias
            
    def predict(self, X):
        activation = np.dot(X, self.weights) + self.bias
        return np.sign(activation)


    
def compute_error(y_true, y_pred):
    try:
        n = len(y_true)
        return np.sum(np.abs(y_true-y_pred))/n
    except:
        raise Exception('Function not yet implemented!')
    
if __name__ == '__main__':

    Xtrn, ytrn = get_batch_1_2() # change the batch values fucntion from 1-1 to 1-2... etc

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
    # svm = SVM(lr=0.01, C=10, epochs=1000, tol=0.001)
    svm = LinearSVM()


    svm.fit(X_train_scaled, ytrn)
    y_pred = svm.predict(X_test_scaled)
    print("x train scaled",X_train_scaled)
    print("x test scaled",X_test_scaled)

    print("y predicted", y_pred)
    
    test_error = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(test_error*100))
    
    print("Accuracy score from scikit: ",accuracy_score(ytst, y_pred))



    # # Train the SVM model
    svm = LinearSVC(C=1, max_iter=10000)
    svm.fit(X_train_scaled, ytrn)

    # Make predictions on the test set
    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(ytst, y_pred)
    print("Test Accuracy from SVM SCKIT MODEL: ", acc)
 
