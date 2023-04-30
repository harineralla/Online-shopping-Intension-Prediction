import numpy as np
import pandas as pd


def get_batch_1_1():
    # Load the training data of batch 1:1
    M = np.genfromtxt('main/data/batch-1-1.data',
                      skip_header=0, delimiter=',')
    M = M.astype(int)
    # print(M)

    # ytrn = M[:, 0]
    # Xtrn = M[:, 1:]
    # print(Xtrn)

    Xtrn = M[:, :-1]
    ytrn = M[:, -1]

    return Xtrn, ytrn
    # Read Excel file
    # pass


def get_batch_1_2():
    # Load the training data of batch 1:2
    M = np.genfromtxt('main/data/batch-1-2.data', skip_header=0,
                      delimiter=',')
    M = M.astype(int)
    Xtrn = M[:, :-1]
    ytrn = M[:, -1]

    return Xtrn, ytrn


def get_batch_1_3():
    # Load the training data of batch 1:3
    M = np.genfromtxt('main/data/batch-1-3.data', skip_header=0,
                      delimiter=',')
    M = M.astype(int)
    Xtrn = M[:, :-1]
    ytrn = M[:, -1]

    return Xtrn, ytrn


def get_batch_1_4():
    # Load the training data of batch 1:4
    M = np.genfromtxt('main/data/batch-1-4.data', skip_header=0,
                      delimiter=',')
    M = M.astype(int)
    Xtrn = M[:, :-1]
    ytrn = M[:, -1]

    return Xtrn, ytrn


def get_test_data():
    # Load test data main/data/test.data
    M = np.genfromtxt('main/data/test.data', skip_header=0,
                      delimiter=',')
    M = M.astype(int)
    # ytst = M[:, 0]
    # Xtst = M[:, 1:]


    Xtst = M[:, :-1]
    ytst = M[:, -1]

    return Xtst, ytst


# def get():
#     df = pd.read_csv('./data/test_data.csv')
#     np.savetxt('./data/test.data', df, delimiter=',')

# if __name__ == '__main__':
#     get()

# with open('main/data/batch-1-1.data', 'r') as f:
#     for line in f:
#         print(line)
        # break

# get_batch_1_1()