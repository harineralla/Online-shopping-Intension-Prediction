
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class SupportVectorMachine:
    def __init__(self, C=10, features=2, width=0.1, kernel="None"):
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
    
    def fit(self, x_train, y_train, epochs=1000, print_interval=100, learning_rate=0.01):
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
        y_hat = np.array([0 if i == -1 else 1 for i in y_hat])
        print("prediction vals", y_hat[:30])
        diff = np.abs(np.array([0 if i == -1 else 1 for i in y]) - y_hat)
        corr_pred = len(diff) - sum(diff)
        return [(corr_pred) / len(diff), y_hat]

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


if __name__ == '__main__':
    Xtrn, ytrn = get_batch_1_4() # change the batch values fucntion from 1-1 to 1-2... etc
    
    q=Xtrn.shape[1]
    print("shape before", q)
    Xtst, ytst = get_test_data()
    print("Data preprocessing in progress..")
    Xtrn, ytrn = shuffle(Xtrn, ytrn, random_state=42)
    # x1tst,y1tst = get_batch_1_3()


    from sklearn.feature_selection import SelectKBest, f_classif

    # Assuming X_train and y_train are your training data and labels, respectively

    # Initialize SelectKBest with the desired number of features to select
    k = 15  # Number of features to select
    selector = SelectKBest(score_func=f_classif, k=k)

    # Fit the selector on the training data
    selector.fit(Xtrn, ytrn)

    # Get the indices of the selected features
    selected_feature_indices = selector.get_support(indices=True)

    # Subset the training data with the selected features
    Xtrn = Xtrn[:, selected_feature_indices]
    w=Xtrn.shape[1]
    print("shape after", w)


    # Similarly, subset the test data (if applicable)
    Xtst = Xtst[:, selected_feature_indices]


    # Feature scaling for SVM and Gradient Boosting
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xtrn)
    X_test_scaled = scaler.transform(Xtst)
    # x1tstscaled= scaler.transform(x1tst)
    
    lr_list = [0.01]
    C_list = [20]
    epochs_list = [10]
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
                    precision = precision_score(ytst, test_eval[1])

                    # Calculate recall score
                    recall = recall_score(ytst, test_eval[1])

                    print("Precision of our model:", precision)
                    print("Recall of our model:", recall)

    #TSEN
    # Apply TCEN for dimensionality reduction
    # ypred = model.predict(X_test_scaled)
    # tcen = TSNE(n_components=3, random_state=42)

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)

    # y_test_pred = model.predict(X_train_scaled)
    # X_test_reduced = pca.fit_transform(model.high_dim_space)
    # weights_reduced = pca.transform([model.weights])
    # print("weights:",weights_reduced)
    # print("reduced x values")
    # scatter_plot_2d_features(X_test_reduced,y_test_pred)
    # scatter_plot_3d_features(X_test_reduced,ypred)

                    #______________________________________________________________________
                    #plotting decision boundary for the model
                    # plot_decision_boundary(model,  X_train_scaled,  model.predict(X_train_scaled))
                    #___________________________________________________________________

                    #visualize
                     
                    # Plot the loss function
                    
                    # plt.plot(model.loss_values)
                    # plt.xlabel('Epoch')
                    # plt.ylabel('Loss')
                    # plt.title('Loss Function')
                    # plt.show()
                    #--------------------------------------------------------

                    # # Plotting true and predicted labels
                    # # Reduce the dimensionality of X_test_scaled using PCA
                    # pca = PCA(n_components=2)
                    # X_test_reduced = pca.fit_transform(X_test_scaled)

                    # # Plotting true and predicted labels after PCA
                    # plt.figure(figsize=(12, 6))

                    # # True Labels
                    # plt.subplot(1, 2, 1)
                    # plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=ytst, cmap=plt.cm.Paired)
                    # plt.xlabel('Component 1')
                    # plt.ylabel('Component 2')
                    # plt.title('True Labels')

                    # # Predicted Labels
                    # plt.subplot(1, 2, 2)
                    # plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=model.predict(X_test_scaled), cmap=plt.cm.Paired)
                    # plt.xlabel('Component 1')
                    # plt.ylabel('Component 2')
                    # plt.title('Predicted Labels')

                    # plt.tight_layout()
                    # plt.show()

                    # ---------------------------------------------------------
                    # XXXXXXXXXX
                    # # Get the dataset using get_batch_1_3()
                    # X, y = get_batch_1_3()

                    # # Select two features for SVM
                    # feature1 = 4  # Index of the first feature
                    # feature2 = 5  # Index of the second feature

                    # # Extract the selected features and the target variable
                    # selected_features = X[:, [feature1, feature2]]

                    # # Shuffle the data
                    # X_shuffled, y_shuffled = shuffle(selected_features, y, random_state=42)

                    # # Split the data into training and test sets
                    # X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)

                    # # Feature scaling
                    # scaler = StandardScaler()
                    # X_train_scaled = scaler.fit_transform(X_train)
                    # X_test_scaled = scaler.transform(X_test)

                  
                    # Z = Z.reshape(xx.shape)
                    # print("almost..")
                    # plt.contourf(xx, yy, Z, alpha=0.8)
                    # sns.scatterplot(x=X_train_scaled[:, 0], y=X_train_scaled[:, 1], hue=y_train, cmap=plt.cm.Paired)
                    # plt.xlabel(f"Feature {feature1+1}")
                    # plt.ylabel(f"Feature {feature2+1}")
                    # plt.title('SVM Decision Boundary')
                    # plt.show()


#________________________________ SCIKIT SVM MODEL ______________________________________________
#   # Train the SVM model
    svmSk = LinearSVC(C=10, max_iter=10000)
    svmSk.fit(X_train_scaled, ytrn)

    # y_train_pred = svmSk.predict(X_train_scaled)
    # Make predictions on the test set
    y_pred = svmSk.predict(X_test_scaled)

    acc = accuracy_score(ytst, y_pred)
    print("Test Accuracy of SCKIT model for SVM: ", acc)
    # print("trained pred", accuracy_score(ytrn, y_train_pred))

    # Calculate precision score
    precision = precision_score(ytst, y_pred)

    # Calculate recall score
    recall = recall_score(ytst, y_pred)

    print("Precision of scikit model:", precision)
    print("Recall of scikit model:", recall)
