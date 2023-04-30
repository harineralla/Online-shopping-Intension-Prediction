import numpy as np
import pandas as pd


def get_batch_1_1():
    # Load the training data of batch 1:1
    M = np.genfromtxt('./data/batch-1-1.data', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)

    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    return Xtrn, ytrn
    # Read Excel file
    # pass


def get_batch_1_2():
    # Load the training data of batch 1:2
    M = np.genfromtxt('./data/batch-1-2.data', skip_header=0,
                      delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    return Xtrn, ytrn


def get_batch_1_3():
    # Load the training data of batch 1:3
    M = np.genfromtxt('./data/batch-1-3.data', skip_header=0,
                      delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    return Xtrn, ytrn


def get_batch_1_4():
    # Load the training data of batch 1:4
    M = np.genfromtxt('./data/batch-1-4.data', skip_header=0,
                      delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    return Xtrn, ytrn


def get_test_data():
    # Load test data
    M = np.genfromtxt('./data/test.data', skip_header=0,
                      delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    return Xtst, ytst


# def get():
#     df = pd.read_csv('./data/test_data.csv')
#     np.savetxt('./data/test.data', df, delimiter=',')

# if __name__ == '__main__':
#     get()
