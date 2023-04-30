import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
import graphviz
from data_utils import *
from sklearn.metrics import accuracy_score


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
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

        # print("attr pairs", attribute_value_pairs)
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
        
        
        # print("DT   >>>>   ",decision_tree)

        return decision_tree
    except Exception as e:
        print("erorrr hereee ")
        print(e)
        # raise Exception('Function not yet implemented!')



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

    # INSERT YOUR CODE HERE
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


# def confusion_matrix_multiclass(y, y_pred, classes, fig):
#     confusion_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))))
#     rows = []
#     columns = []
#     for cl in classes.tolist():
#         rows.append("Actual " + str(cl))
#         columns.append("Predicted " + str(cl))
#     for i, j in zip(y, y_pred):
#         # breakpoint()
#         confusion_matrix[i][j] += 1
#     fig.subplots_adjust(left=0.3, top=0.8, wspace=2)
#     ax = plt.subplot2grid((1, 1), (0, 0), colspan=2, rowspan=2)
#     table = ax.table(cellText=confusion_matrix.tolist(),
#                      rowLabels=rows,
#                      colLabels=columns, loc="upper center")
#     table.set_fontsize(14)
#     table.scale(1, 2)
#     ax.axis("off")


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
    # BATCH 1:1
    # Load batch-1 
    Xtrn, ytrn = get_batch_1_1() # change the batch values fucntion from 1-1 to 1-2... etc

    Xtst, ytst = get_test_data()
    # print("X Test DATA IN DT",Xtst[0])
    print(ytst[0:9])
    # print(Xtst[0])

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    # visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    print(y_pred[0:9])
    test_error = compute_error(ytst, y_pred)

    print('Test Error = {0:4.2f}%.'.format(test_error*100))
    
    print("Accuracy score from scikit: ",accuracy_score(ytst, y_pred))