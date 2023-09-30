import numpy as np
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle



# class SVM:
#     def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001):
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs
#         self.tol = tol
#         self.w = None
#         self.b = None
    
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.w = np.zeros(n_features)
#         self.b = 0
        
#         for epoch in range(self.epochs):
#             y_pred = np.dot(X, self.w) + self.b
#             loss = np.maximum(0, 1 - y * y_pred)
#             dW = np.zeros(n_features)
#             dB = 0
            
#             for i in range(n_samples):
#                 if loss[i] == 0:
#                     dW += self.w
#                     dB += 0
#                 else:
#                     dW += self.w - self.C * y[i] * X[i]
#                     dB += -self.C * y[i]
            
#             self.w -= self.lr * (dW + 2 * self.C * self.w) / n_samples
#             self.b -= self.lr * dB / n_samples
#             # print("W AND BIAS", self.w , self.b)
            

    
#     def predict(self, X):
#         # Compute the predicted values for the input data using the learned weight and bias parameters
#         # print(" x, w, b ",X, self.w,  self.b)
#         # print("dot prod", np.dot(X, self.w)+ self.b)
#         # print("bias... ",self.b)
#         y_pred = np.sign(np.dot(X, self.w) + self.b)
#         # print( "predicted prior is ",y_pred[0:3])
#         # Convert -1 to 0 and leave 1 as is
#         y_pred[y_pred == -1] = 0
#         # print( "predicted output is ",y_pred[0:3])
        
#         return y_pred



## linear kernel with l2 
class SVM:
    def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001, l2_reg=0.1):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.tol = tol
        self.l2_reg = l2_reg
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            loss = np.maximum(0, 1 - y * y_pred)
            dW = np.zeros(n_features)
            dB = 0
            
            for i in range(n_samples):
                if loss[i] == 0:
                    dW += self.w
                    dB += 0
                else:
                    dW += self.w - self.C * y[i] * X[i]
                    dB += -self.C * y[i]
            
            dW += 2 * self.l2_reg * self.w
            
            self.w -= self.lr * dW / n_samples
            self.b -= self.lr * dB / n_samples
            

    
    def predict(self, X):
        # Compute the predicted values for the input data using the learned weight and bias parameters
        y_pred = np.sign(np.dot(X, self.w) + self.b)
        print(y_pred)
        # Convert -1 to 0 and leave 1 as is
        y_pred[y_pred == -1] = 0
        print("again  ",y_pred)
        
        return y_pred


# linear kernel, l2 with PCA 
# from sklearn.decomposition import PCA
# class SVM:
#     def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001, l2_reg=0.1, n_components=19):
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs
#         self.tol = tol
#         self.l2_reg = l2_reg
#         self.w = None
#         self.b = None
#         self.n_components = n_components
#         self.pca = None
        
#     def fit(self, X, y):
#         # Apply PCA to reduce dimensionality if n_components is given
#         if self.n_components:
#             self.pca = PCA(n_components=self.n_components)
#             X = self.pca.fit_transform(X)
            
#         n_samples, n_features = X.shape
#         self.w = np.zeros(n_features)
#         self.b = 0
        
#         for epoch in range(self.epochs):
#             y_pred = np.dot(X, self.w) + self.b
#             loss = np.maximum(0, 1 - y * y_pred)
#             dW = np.zeros(n_features)
#             dB = 0
            
#             for i in range(n_samples):
#                 if loss[i] == 0:
#                     dW += self.w
#                     dB += 0
#                 else:
#                     dW += self.w - self.C * y[i] * X[i]
#                     dB += -self.C * y[i]
            
#             dW += 2 * self.l2_reg * self.w
            
#             self.w -= self.lr * dW / n_samples
#             self.b -= self.lr * dB / n_samples
            
    
#     def predict(self, X):
#         # Apply PCA to reduce dimensionality if n_components is given
#         if self.n_components:
#             X = self.pca.transform(X)
        
#         # Compute the predicted values for the input data using the learned weight and bias parameters
#         y_pred = np.sign(np.dot(X, self.w) + self.b)
#         # Convert -1 to 0 and leave 1 as is
#         y_pred[y_pred == -1] = 0
        
#         return y_pred



# #poly kernel with l2
# import numpy as np
# class SVM:
#     def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001, degree=2, gamma=0.1):
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs
#         self.tol = tol
#         self.degree = degree
#         self.gamma = gamma
#         self.w = None
#         self.b = None
    
#     def poly_kernel(self, X1, X2):
#         return (self.gamma * np.dot(X1, X2.T) + 1) ** self.degree
    
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.w = np.zeros(n_samples)
#         self.b = 0
        
#         K = np.zeros((n_samples, n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 K[i,j] = self.poly_kernel(X[i], X[j])
        
#         for epoch in range(self.epochs):
#             y_pred = np.dot(K, self.w) + self.b
#             loss = np.maximum(0, 1 - y * y_pred)
#             dW = np.zeros(n_samples)
#             dB = 0
            
#             for i in range(n_samples):
#                 if loss[i] == 0:
#                     dW += self.w
#                     dB += 0
#                 else:
#                     dW += self.w - self.C * y[i] * K[:,i]
#                     dB += -self.C * y[i]
            
#             self.w -= self.lr * (dW + 2 * self.C * np.dot(K, self.w)) / n_samples
#             self.b -= self.lr * dB / n_samples

#     def predict(self, X):
#         n_samples, _ = X.shape
#         y_pred = np.zeros(n_samples)
        
#         for i in range(n_samples):
#             s = 0
#             for j in range(self.w.shape[0]):
#                 s += self.w[j] * self.poly_kernel(X[i], X[j])
#             y_pred[i] = np.sign(s + self.b)
        
#         y_pred[y_pred == -1] = 0
        
#         return y_pred

#     def score(self, X, y):
#         y_pred = self.predict(X)
#         acc = np.sum(y == y_pred) / len(y)
#         return acc

# #rbf kernel with l2
# import numpy as np
# import numpy as np

# class SVM:
#     def __init__(self, lr=0.01, C=1.0, epochs=1000, tol=0.001, l2_reg=0.1, gamma=1.0):
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs
#         self.tol = tol
#         self.l2_reg = l2_reg
#         self.gamma = gamma
#         self.alpha = None
#         self.b = None
#         self.K = None
    
#     def rbf_kernel(self, X1, X2):
#         dist = np.linalg.norm(X1 - X2)
#         return np.exp(-self.gamma * dist**2)
    
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.support_vectors = X
#         self.target_values = y
#         self.alpha = np.zeros(n_samples)
#         self.b = 0

#         self.K = np.zeros((n_samples, n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 self.K[i, j] = self.rbf_kernel(X[i], X[j])

#         for epoch in range(self.epochs):
#             for i in range(n_samples):
#                 if self._decision_function(X[i]) * y[i] < 1:
#                     self.alpha[i] += self.lr
#             self.b = np.mean(y - np.dot(self.alpha * y, self.K))

#         self.support_vectors = X[self.alpha != 0]
#         self.alpha = self.alpha[self.alpha != 0]
#         self.target_values = y[self.alpha != 0]

#     def _decision_function(self, X):
#         K_pred = np.zeros(len(self.support_vectors))
#         for i in range(len(self.support_vectors)):
#             K_pred[i] = self.rbf_kernel(X, self.support_vectors[i])

#         return np.dot(self.alpha * self.target_values, K_pred) + self.b

        
#     def predict(self, X):
#         K_pred = np.zeros((X.shape[0], self.K.shape[1]))
        
#         for i in range(X.shape[0]):
#             for j in range(self.K.shape[1]):
#                 K_pred[i, j] = self.rbf_kernel(X[i], X[j])
        
#         y_pred = np.dot(self.alpha * self.y, K_pred) + self.b
#         y_pred = np.sign(y_pred)
#         y_pred[y_pred == -1] = 0
        
#         return y_pred


def compute_error(y_true, y_pred):
    try:
        n = len(y_true)
        y_true = np.array([int(i) for i in y_true])
        y_pred = np.array([int(i) for i in y_pred])
        return np.sum(y_true^y_pred) / n
    except Exception as e:
        print(e)
        # raise Exception('Function not yet implemented!')

    
if __name__ == '__main__':

    Xtrn, ytrn = get_batch_1_4() # change the batch values fucntion from 1-1 to 1-2... etc
    Xtst, ytst = get_test_data()

    Xtrn, ytrn = shuffle(Xtrn, ytrn, random_state=42)

    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt

    # # Load the data
    # X, y = get_batch_1_4()

    # # Perform PCA
    # pca = PCA()
    # pca.fit(X)

    # # Plot the cumulative explained variance ratio
    # cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    # plt.plot(cumulative_explained_variance_ratio)
    # plt.xlabel('Number of components')
    # plt.ylabel('Cumulative explained variance ratio')
    # plt.show()


    # print("x train ",Xtrn[0::15])
    # print("y train ",ytrn[0::15])
    # print("x test ",Xtst[0::15])
    # print("y test ",ytst[0::15])

    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)
    X_test_scaled = scaler.transform(Xtst)

    # print("x train scaled",X_train_scaled)
    # print("x test scaled",X_test_scaled)

    # train the SVM
    # svm = SVM(lr=0.01, C=10, epochs=1000, tol=0.001)
    # svm = SVM()
    lr_list = [0.0001]
    C_list = [0.1]
    epochs_list = [10]
    tol_list = [0.001]

    for lr in lr_list:
        for C in C_list:
            for epochs in epochs_list:
                for tol in tol_list:
                    svm = SVM(lr=lr, C=C, epochs=epochs, tol=tol)
                
                    svm.fit(X_train_scaled, ytrn)
                    # y_train_pred = svm.predict(X_train_scaled)
                    y_pred = svm.predict(X_test_scaled)
                    # print("y predicted ", y_pred[1:25])
                    # train_error = compute_error(ytrn,y_train_pred)
                    test_error = compute_error(ytst, y_pred)
                    print("Accuracy score : ",accuracy_score(ytst, y_pred))
                    print(f"lr={lr}, C={C}, epochs={epochs}, tol={tol}, Test Error={test_error}")

    # svm.fit(X_train_scaled, ytrn)
    # y_pred = svm.predict(X_train_scaled)
    # print("x train scaled",X_train_scaled)
    # print("x test scaled",X_test_scaled)

    # print("y predicted", y_pred)
    
    # test_error = compute_error(ytrn, y_pred)

    # print('Test Error = {0:4.2f}%.'.format(test_error*100))
    
    # print("Accuracy score from scikit: ",accuracy_score(ytrn, y_pred))



                    # # Train the SVM model
                    svmSk = LinearSVC(C=1, max_iter=10000)
                    svmSk.fit(X_train_scaled, ytrn)

                    y_train_pred = svmSk.predict(X_train_scaled)
                    # Make predictions on the test set
                    y_pred = svmSk.predict(X_test_scaled)

                    acc = accuracy_score(ytst, y_pred)
                    print("Test Accuracy from SVM SCKIT MODEL: ", acc)
                    print("trained pred", accuracy_score(ytrn, y_train_pred))

                    # Calculate precision score
                    precision = precision_score(y_true, y_pred)

                    # Calculate recall score
                    recall = recall_score(y_true, y_pred)

                    print("Precision:", precision)
                    print("Recall:", recall)
 
