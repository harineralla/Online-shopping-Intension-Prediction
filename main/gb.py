import numpy as np
from data_utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.trees = []

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(y) < self.min_samples_split:
            return Leaf(y)

        best_feature, best_threshold = self._best_split(X, y)
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(best_feature, best_threshold, left_tree, right_tree)

    def _best_split(self, X, y):
        best_loss = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                loss = self._split_loss(X, y, feature, threshold)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split_loss(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_y = y[left_mask]
        right_y = y[right_mask]
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        left_loss = self.loss.gradient(left_y, self._predict(X[left_mask]))
        right_loss = self.loss.gradient(right_y, self._predict(X[right_mask]))
        return left_weight * left_loss + right_weight * right_loss

    def _predict(self, X):
        return sum(self.learning_rate * tree.predict(X) for tree in self.trees)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.loss = GradientBoostingMSE(n_estimators=800, learning_rate=0.01, max_depth=6, random_state=5)
        self.trees = []
        r = y
        for i in range(self.n_estimators):
            tree = self._build_tree(X, r, depth=0)
            r -= self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        return self._predict(X)

class GradientBoostingMSE(GradientBoosting):

    def __init__(self, n_estimators, learning_rate, max_depth, random_state):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        
    def negative_gradient(self, y, pred):
        return y - pred
        
    def predict_value(self, X, trees):
        pred = np.zeros(len(X))
        for tree in trees:
            pred += self.learning_rate * tree.predict(X)
        return pred

class DecisionNode:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict(self, X):
        mask = X[:, self.feature] <= self.threshold
        y = np.zeros(len(X))
        y[mask] = self.left.predict(X[mask])
        y[~mask] = self.right.predict(X[~mask])
        return y


class LeafNode:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


# change the batch values fucntion from 1-1 to 1-2... etc
Xtrn, ytrn = get_batch_1_4()

Xtst, ytst = get_test_data()

gb = GradientBoosting(n_estimators=800, learning_rate=0.01,
                      max_depth=6, random_state=5)
gb.fit(Xtrn, ytrn)

y_pred = gb.predict(Xtst)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", round(accuracy_score(ytst, y_pred), 4))
# Precision Score on the test data
print("Precision Score:", round(precision_score(ytst, y_pred), 4))
# Precision Score on the test data
print("F-1 Score:", round(f1_score(ytst, y_pred), 4))
