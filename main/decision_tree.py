import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import graphviz


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    try:
        unique_values, indices_by_value = np.unique(x), {}
        for value in unique_values:
            indices_by_value[value] = np.where(x == value)[0]
        return indices_by_value
    except Exception as e:
        print("An exception occurred:", e)


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    try:
        probabilities, y_entropy_value = [], 0
        val_indices = partition(y)
        num_samples = len(y)
        for val in val_indices.keys():
            count = len(val_indices[val])
            probabilities.append(float(count/num_samples))
        for prob in probabilities:
            y_entropy_value = y_entropy_value - prob * np.log2(prob)
        return y_entropy_value
    except Exception as e:
        print("An exception occurred:", e)


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    try:
        y_entropy = entropy(y)
        y_entropy_given_x = 0
        num_samples = len(x)
        val_indices = partition(x)

        for val in val_indices.keys():
            probability = float(len(val_indices[val]) / num_samples)
            y_given_x = [y[i] for i in val_indices[val]]
            y_entropy_given_x += probability * entropy(y_given_x)

        return y_entropy - y_entropy_given_x
    except Exception as e:
        print("An exception occurred:", e)


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    try:
        if attribute_value_pairs is None:
            attribute_value_pairs = []
            for attr_idx in range(len(x[0])):
                all_values = np.array([item[attr_idx] for item in x])
                for attr_value in np.unique(all_values):
                    attribute_value_pairs.append((attr_idx, attr_value))

        attribute_value_pairs = np.array(attribute_value_pairs)

        unique_vals_y = np.unique(y)
        if len(unique_vals_y) == 1:
            return unique_vals_y[0]

        unique_vals_y, count_y = np.unique(y, return_counts=True)
        if len(attribute_value_pairs) == 0:
            return unique_vals_y[np.argmax(count_y)]

        if max_depth == depth:
            return unique_vals_y[np.argmax(count_y)]

        mutual_info_pairs = []
        for attr, val in attribute_value_pairs:
            attr_val_arr = np.array((x[:, attr] == val).astype(int))
            mutual_info = mutual_information(attr_val_arr, y)
            mutual_info_pairs.append(mutual_info)
        
        mutual_info_pairs = np.array(mutual_info_pairs)
        chosen_attr, chosen_val = attribute_value_pairs[np.argmax(mutual_info_pairs)]

        part = partition(np.array((x[:, chosen_attr] == chosen_val).astype(int)))
        attribute_value_pairs = np.delete(attribute_value_pairs, np.argmax(mutual_info_pairs), 0)

        decision_tree = {}
        for val, ele_indices in part.items():
            out_label = bool(val)
            x_after_part = x.take(np.array(ele_indices), axis=0)
            y_after_part = y.take(np.array(ele_indices), axis=0)
            decision_tree[(chosen_attr, chosen_val, out_label)] = id3(x_after_part, y_after_part,
                                                                      attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth)

        return decision_tree
    except Exception as e:
        raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    try:
        for node, child in tree.items():
            feature_idx = node[0]
            feature_val = node[1]
            decision = node[2]

            if decision == (x[feature_idx] == feature_val):
                if type(child) is not dict:
                    predicted_label = child
                else:
                    predicted_label = predict_example(x, child)

        return predicted_label
    except Exception as e:
        print("An exception occurred:", e)


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    try:
        n = len(y_true)
        return np.sum(np.abs(y_true-y_pred))/n
    except:
        raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if tree is None:
        raise ValueError("The decision tree is empty")

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


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


def predict_example_proba(x, tree):
    """
    Predicts the probability of each class label for a single example x using tree by recursively descending the tree
    until a label/leaf node is reached.

    Returns a dictionary of the form {class_label: probability} for the predicted probabilities of x according to tree
    """
    try:
        for node, child in tree.items():
            feature_idx = node[0]
            feature_val = node[1]
            decision = node[2]

            if decision == (x[feature_idx] == feature_val):
                if type(child) is not dict:
                    # If we've reached a leaf node, return the probability distribution for the class labels
                    return {0: 1 - child, 1: child}
                else:
                    # Recursively descend the tree
                    return predict_example_proba(x, child)
    except Exception as e:
        print("An exception occurred:", e)


if __name__ == '__main__':
    # PART-1 training the model using four batches 
    # BATCH 1:1
    # Load batch-1 
    Xtrn, ytrn = get_batch_1_1() # change the batch values fucntion from 1-1 to 1-2... etc
    # Xtrn, ytrn = get_batch_1_2()
    # Xtrn, ytrn = get_batch_1_3()
    # Xtrn, ytrn = get_batch_1_4() # please uncomment the line when you choose which batch you want to run
    Xtst, ytst = get_test_data()

    """
    Part- A Training model using our model above for each batch
    """
    train_errors = []
    test_errors = []
    depths = []
    decision_trees = []
    final_ypred = 0
    for depth in range (1,11):
        decision_tree = id3(Xtrn, ytrn, max_depth=depth)
        decision_trees.append(decision_tree)
        YPred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        YPred_tst = [predict_example(x, decision_tree) for x in Xtst]
        train_error = compute_error(ytrn, YPred_trn)
        test_error = compute_error(ytst, YPred_tst)
        depths.append(depth)
        train_errors.append(train_error)
        test_errors.append(test_error)
        final_ypred = YPred_tst
        print("For depth=", depth)
        print('Test Error= {0:4.2f}%.'.format(test_error*100))
        print("Accuracy score",accuracy_score(ytst, YPred_tst))
        print("Precision score from scikit", precision_score(ytst,YPred_tst))
        print("Recall score from scikit", recall_score(ytst,YPred_tst))
        print("F1 score from scikit", f1_score(ytst,YPred_tst))
        print()


    fig1 = plt.figure(1)
    confusion_matrix(ytst, final_ypred, fig1)
    fig1.suptitle("Decision Tree after 10 depths training Confusion Matrix-Batch 1:4")

    splt_idx = 130
    plt.title("Training and Test errors for Batch 1:1 for depths 10")
    plt.xlabel("Max Depth")
    plt.ylabel("Error")
    plt.grid()
    plt.plot(depths, train_errors, 'o-', color='r', label='Training Examples')
    plt.plot(depths, test_errors, 'o-', color='b', label='Testing Examples')
    plt.legend(loc="best")
    # plt.show()

    y_scores = []
    for x in Xtst:
        y_scores.append(predict_example_proba(x, decision_trees[7])[1])
    fpr, tpr, thresholds = roc_curve(ytst, y_scores)
    roc_auc = auc(fpr, tpr)

    # plot ROC Curve
    plt.figure(3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


    """
    Part- B Training model using Scikit Learn and visualizing using graphviz and confusion Matrix
    """
    train_errors = []
    test_errors = []
    depths = []
    decision_trees = []
    scikitDescTree = tree.DecisionTreeClassifier(criterion="entropy")
    scikitDescTree.fit(Xtrn, ytrn)
    predictedYScikit = [scikitDescTree.predict(
        np.array(x).reshape(1, -1))[0] for x in Xtst]
    print("Accuracy score from scikit for batch 1:1: ",accuracy_score(ytst, predictedYScikit))
    print("Precision score from scikit for batch 1:1:", precision_score(ytst,predictedYScikit))
    print("Recall score from scikit for batch 1:1:", recall_score(ytst,predictedYScikit))
    print("F1 score from scikit for batch 1:1:", f1_score(ytst,predictedYScikit))


    fig3 = plt.figure(5)
    confusion_matrix(ytst, predictedYScikit, fig3)
    fig3.suptitle(
        "Confusion Matrix for Batch 1 Test Set with Scikit Decision tree")
    plt.show()
    splt_idx = 130

    # Generate the graphviz visualization of the decision tree
    dot_data = export_graphviz(scikitDescTree, out_file=None,
                            feature_names=['Administrative', 'Administrative_Duration', 'Informational',
                                          'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                                          'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
                                          'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
                                          'Weekend', 'OperatingSystems_1', 'OperatingSystems_2',
                                          'OperatingSystems_3', 'OperatingSystems_4', 'OperatingSystems_5', 
                                          'OperatingSystems_6', 'OperatingSystems_7', 'Browser_1', 'Browser_2',
                                          'Browser_3', 'Browser_4', 'Browser_5'],
                            filled=True, rounded=True,
                            special_characters=True)
    graph = graphviz.Source(dot_data)

    # Display the graphviz visualization of the decision tree
    graph.view()