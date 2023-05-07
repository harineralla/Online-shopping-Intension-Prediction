import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score

class SupportVectorMachine:
    def __init__(self, C=10, features=2, width=0.1, kernel="None"):
        self.C = C
        self.features = features
        self.width = width
        self.kernel = kernel
        self.weights = np.zeros(features)
        self.bias = 0.0
    
    def gaussian_kernel(self, x1, x):
        n = x1.shape[0]
        m = x.shape[0]
        op = []
        for i in range(n):
            row = []
            for j in range(m):
                #K(x, y) = exp(-||x - y||^2 / (2 * σ^2))
                cos = np.exp(-np.sum((x1[i] - x[j]) ** 2) / (2 * self.width))
                row.append(cos)
            op.append(row)
        return np.array(op)

    
    def fit(self, x_train, y_train, epochs=1000, print_interval=100, learning_rate=0.01):
        y = y_train[:]
        x = x_train[:]
        self.initial = x[:]

        if self.kernel == "gaussian":
            x_transformed = self.gaussian_kernel(x, x_train)
            m = x_transformed.shape[0]
            self.weights = np.zeros(m)
        else:
            x_transformed = x
        
        n = x_transformed.shape[0]
        for epoch in range(epochs):
            y_hat = np.dot(x_transformed, self.weights) + self.bias
            grad_weights = (-self.C * np.multiply(y, x_transformed.T).T + self.weights).T
            
            for weight in range(self.weights.shape[0]):
                cond=1 - y_hat <= 0
                grad_weights[weight] = np.where(cond, self.weights[weight], grad_weights[weight])
            
            grad_weights = np.sum(grad_weights, axis=1)
            self.weights -= learning_rate * grad_weights / n
            grad_bias = -y * self.bias
            cond1=1 - y_hat <= 0
            grad_bias = np.where(cond1, 0, grad_bias)
            grad_bias = np.sum(grad_bias)
            self.bias -= grad_bias * learning_rate / n
            
            if (epoch + 1) % print_interval == 0:
                loss = self.loss_function(y, y_hat)
                print(f" Epoch {epoch+1} --> Loss = {loss} ")
    
    def loss_function(self, y, y_hat):
        sum_terms = np.maximum(0, 1 - y * y_hat)
        sum_terms = np.maximum(sum_terms, 0)
        return (self.C * np.sum(sum_terms) / len(y) + np.sum(self.weights ** 2) / 2)
    
    def predict(self, x, y):
        if self.kernel == "gaussian":
            x = self.gaussian_kernel(x, self.initial)
        y_hat = np.where(np.dot(x, self.weights) + self.bias > 0, 1, -1)
        y_hat = np.array([0 if i == -1 else 1 for i in y_hat])
        print("prediction vals", y_hat[:30])
        diff = np.abs(np.array([0 if i == -1 else 1 for i in y]) - y_hat)
        
        return (len(diff) - sum(diff)) / len(diff)


if __name__ == '__main__':
    Xtrn, ytrn = get_batch_1_1() # change the batch values fucntion from 1-1 to 1-2... etc
    Xtst, ytst = get_test_data()
    Xtrn, ytrn = shuffle(Xtrn, ytrn, random_state=42)

    x1tst,y1tst = get_batch_1_3()

    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)
    X_test_scaled = scaler.transform(Xtst)
    x1tstscaled= scaler.transform(x1tst)


    # #Model
    model=SupportVectorMachine(C=20,features=X_train_scaled.shape[1],width=0.01,kernel="gaussian")
    model.fit(X_train_scaled,ytrn,epochs=10,print_interval=1,learning_rate=0.01)
    print("Training Accuracy = {}".format(model.predict(X_train_scaled,ytrn)))
    print("Testing Accuracy = {}".format(model.predict(X_test_scaled,ytst)))
    print("Testing new Accuracy = {}".format(model.predict(x1tstscaled,y1tst)))