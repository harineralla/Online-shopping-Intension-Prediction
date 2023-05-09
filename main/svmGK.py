
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import KFold



class SupportVectorMachine:
    def __init__(self, C=20, features=29, width=0.01, kernel="None"):
        self.C = C
        self.features = features
        self.width = width
        self.kernel = kernel
        self.weights = np.zeros(features)
        self.bias = 0.0
        self.loss_values=[]
        self.high_dim_space = None
    
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
    
    def fit(self, x_train, y_train, epochs=20, print_interval=1, learning_rate=0.1):
        y = y_train[:]
        x = x_train[:]
        self.initial = x[:]

        if self.kernel == "gaussian":
            x_transformed = self.gaussian_kernel(x, x_train)
            self.high_dim_space = x_transformed
            m = x_transformed.shape[0]
            self.weights = np.zeros(m)
            # print("modif weights",self.weights)
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
                self.loss_values.append(loss)
                print(f" Epoch {epoch+1} --> Loss = {loss} ")
    
    def loss_function(self, y, y_hat):
        sum_terms = np.maximum(0, 1 - y * y_hat)
        sum_terms = np.maximum(sum_terms, 0)
        return (self.C * np.sum(sum_terms) / len(y) + np.sum(self.weights ** 2) / 2)

    def evaluate(self,x,y):
        y_hat=self.predict(x)
        # y_hat = np.array([0 if i == -1 else 1 for i in y_hat])
        # print("prediction vals", y_hat[:50])
        # diff = np.abs(np.array([0 if i == -1 else 1 for i in y]) - y_hat)
        n = len(y)
        y_true = np.array([0 if i == -1 else 1 for i in y])
        y_pred = np.array([0 if i == -1 else 1 for i in y_hat])
        # print("in eval y_pred",y_pred[:50])
        return [1-np.sum(y_true^y_pred) / n, y_hat]
        # corr_pred = len(diff) - sum(diff)
        # return [(corr_pred) / len(diff), y_hat]

    def predict(self,x):
        if(self.kernel=="gaussian"):
            x=self.gaussian_kernel(x,self.initial)
            self.test_high_dim = x
        return np.where(np.dot(x,self.weights)+self.bias>0,1,-1)

# Function to plot the loss function
def plot_loss(loss_values):
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.show()

# Function to plot the decision boundary

def plot_decision_boundary(model, X, y):
    # Reduce the dimensionality of the data to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    # Generate mesh grid for 2D plot
    h = 0.1  # step size in the mesh
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Transform the mesh points to the original high-dimensional space
    mesh_points_transformed = pca.inverse_transform(mesh_points)
    
    # Predict the class labels for the mesh points
    Z = model.predict(mesh_points_transformed)
    
    # Reshape the predictions and plot the decision boundary
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Decision Boundary')
    plt.show()


## Plot vernoi diagram
def plot_decision_boundary(model, X, y):
    # Reduce the dimensionality of the data to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    vor = Voronoi(X_reduced)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='gray', line_width=0.5, line_alpha=0.8)
    y_pred = model.predict(X)

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap=plt.cm.Paired)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Voronoi Diagram - proximity boundary')
    plt.show()

def scatter_plot_2d_features(features, labels):
    # Create a dataframe with features and labels
    data = pd.DataFrame({'Feature 1': features[:, 0], 'Feature 2': features[:, 1],'Label': labels})
    sns.set(style="ticks")
    sns.scatterplot(x='Feature 1', y='Feature 2',hue='Label', data=data, palette='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of 2-Dimensional Features')
    plt.show()

def scatter_plot_3d_features(features, labels):
    data = pd.DataFrame({'Feature 1': features[:, 0], 'Feature 2': features[:, 1], 'Feature 3': features[:, 2], 'Label': labels})
    sns.set(style="ticks")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['Feature 1'], data['Feature 2'], data['Feature 3'], c=data['Label'], cmap='viridis')
    ax.set_xlabel('f 1')
    ax.set_ylabel('f 2')
    ax.set_zlabel('f 3')
    ax.set_title('Scatter Plot of 3-Dimensional Features')
    fig.colorbar(scatter)
    plt.show()

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


if __name__ == '__main__':
    Xtrn, ytrn = get_batch_1_3() # change the batch values fucntion from 1-1 to 1-2... etc
    
    q=Xtrn.shape[1]
    print("cols before feature selection:", q)
    # Xtst, ytst = get_test_data()
    Xtst, ytst = get_batch_1_1()
    print("Data preprocessing in progress..")
    Xtrn, ytrn = shuffle(Xtrn, ytrn, random_state=12)
    # x1tst,y1tst = get_batch_1_3()

    #feature selection
    k = 12  # Number of features to select
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(Xtrn, ytrn)
    selected_feature_indices = selector.get_support(indices=True)
    Xtrn = Xtrn[:, selected_feature_indices]
    w=Xtrn.shape[1]
    print("cols after feaure selection:", w)
    Xtst = Xtst[:, selected_feature_indices]


    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)
    X_test_scaled = scaler.transform(Xtst)
    # x1tstscaled= scaler.transform(x1tst)
    
    lr_list = [0.1]
    C_list = [0.01]
    epochs_list = [5]
    width_list = [0.01]


    #Model
    for lr in lr_list:
        for C in C_list:
            for epochs in epochs_list:
                for width in width_list:
                    model=SupportVectorMachine(C=C,features=X_train_scaled.shape[1],width=width,kernel="gaussian")
                    print("Training the model...")
                    model.fit(X_train_scaled,ytrn,epochs=epochs,print_interval=1,learning_rate=lr)
                    print(f"_________lr={lr}, C={C}, epochs={epochs}, width={width}_____________")
                    train_eval=(model.evaluate(X_train_scaled,ytrn))
                    test_eval = (model.evaluate(X_test_scaled,ytst))
                    print(f"Training Accuracy of our model= {train_eval[0]}")
                    print(f"Testing Accuracy of our model= {test_eval[0]}")
                    # print("Testing new Accuracy = {}".format(model.predict(x1tstscaled,y1tst)))

                    # Calculate precision score
                    precision = precision_score(ytst, test_eval[1], average='macro', zero_division=1)

                    # Calculate recall score
                    recall = recall_score(ytst, test_eval[1],  average='macro', zero_division=1)

                    print("Precision of our model:", precision)
                    print("Recall of our model:", recall)

   

                    #______________________________________________________________________
                    #plotting decision boundary for the model
                    plot_decision_boundary(model,  X_train_scaled,  model.predict(X_train_scaled))
                    #___________________________________________________________________

                    #visualize
                     
                    # Plot the loss function
                    
                    plt.plot(model.loss_values)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Loss Function')
                    plt.show()
                    #--------------------------------------------------------

                    # Plotting true and predicted labels
                    # # Reduce the dimensionality of X_test_scaled using PCA
                    pca = PCA(n_components=2)
                    X_test_reduced = pca.fit_transform(X_test_scaled)

                    # Plotting true and predicted labels after PCA
                    plt.figure(figsize=(12, 6))

                    # True Labels
                    plt.subplot(1, 2, 1)
                    plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=ytst, cmap=plt.cm.Paired)
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    plt.title('True Labels')

                    # Predicted Labels
                    plt.subplot(1, 2, 2)
                    plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=model.predict(X_test_scaled), cmap=plt.cm.Paired)
                    plt.xlabel('Component 1')
                    plt.ylabel('Component 2')
                    plt.title('Predicted Labels')

                    plt.tight_layout()
                    plt.show()


#________________________________ SCIKIT SVM MODEL ______________________________________________
#________________________________ SCIKIT SVM MODEL ______________________________________________
#   # Train the SVM model
    # for c in [0.1,1,30,50]:
    svmSk = LinearSVC(C=0.1, max_iter=10)
    svmSk.fit(X_train_scaled, ytrn)

    # y_train_pred = svmSk.predict(X_train_scaled)
    # Make predictions on the test set
    y_pred = svmSk.predict(X_test_scaled)

    acc = accuracy_score(ytst, y_pred)
    precision = precision_score(ytst, y_pred)
    recall = recall_score(ytst, y_pred)
    f1_score = f1_score(ytst,y_pred)
    print("Accuracy of SCKIT model for SVM-Linear: ", acc)
    print("Precision of scikit model for SVM-Linear:", precision)
    print("Recall of scikit model for SVM-Linear:", recall)
    print("F1 Score of scikit model for SVM-Linear", f1_score)

    # Plot confusion matrix
    fig1 = plt.figure(1)
    confusion_matrix(ytst, y_pred, fig1)
    fig1.suptitle("SVM-Linear Confusion Matrix - C {0}".format(0.1))

    # Plot ROC curve
    y_pred_proba = svmSk.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(ytst, y_pred_proba)
    roc_auc = roc_auc_score(ytst, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', label='ROC Curve (area = %0.2f)' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for SVM Linear')
    plt.legend(loc="lower right")
    plt.show()

    #_____________cross validation-----------------------------------------

    # # Get the data from each batch
    Xtrn1, ytrn1 = get_batch_1_1()
    Xtrn2, ytrn2 = get_batch_1_2()
    Xtrn3, ytrn3 = get_batch_1_3()
    # Xtrn4, ytrn4 = get_batch_1_4()

    # # Concatenate the data horizontally
    Xtrn_combined = np.concatenate((Xtrn1, Xtrn2, Xtrn3), axis=0)
    # # Xtrn_combined = np.concatenate((Xtrn1, Xtrn2), axis=0)
    # Xtrn_combined = Xtrn1

    ytrn_combined = np.concatenate((ytrn1, ytrn2, ytrn3), axis=0)
    # # ytrn_combined = np.concatenate((ytrn1, ytrn2), axis=0)
    # ytrn_combined = ytrn1

    print("Data preprocessing in progress..")
    Xtrn, ytrn = shuffle(Xtrn_combined, ytrn_combined, random_state=3)

    #feature selection
    k = 12  # Number of features to select
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(Xtrn, ytrn)
    selected_feature_indices = selector.get_support(indices=True)
    Xtrn = Xtrn[:, selected_feature_indices]
    w=Xtrn.shape[1]
    print("cols after feaure selection:", w)


    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)

    

    # # Print the combined data
    # print("Combined Xtrn shape:", Xtrn_combined.shape)
    # print("Combined ytrn shape:", ytrn_combined.shape)

    # Assuming X and y are your input features and labels
    X = X_train_scaled  # Your input features
    y = ytrn_combined # Your labels

    # X = scaler.fit_transform(X)
    
    model = SupportVectorMachine(C=0.1, features=X.shape[1], width=0.01, kernel="gaussian")

    # Perform cross-validation
    num_folds = 4
    # kf = StratifiedKFold(n_splits=num_folds, shuffle = True, random_state=42)
    kf = KFold(n_splits=num_folds)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for train_index, val_index in kf.split(X,y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit the model on the training data
        model.fit(X_train, y_train,epochs=10,print_interval=1,learning_rate=0.01)

        # Predict on the validation data
        y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_scores.append(accuracy)

        # Calculate precision and recall (assuming binary classification)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Print the scores for each fold
    for i in range(num_folds):
        print(f"Fold {i+1} Accuracy: {accuracy_scores[i]}")
        print(f"Fold {i+1} Precision: {precision_scores[i]}")
        print(f"Fold {i+1} Recall: {recall_scores[i]}")

    # Calculate the average scores across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")

####################### Final ###########################
#Please comment necessary file for necessary actions