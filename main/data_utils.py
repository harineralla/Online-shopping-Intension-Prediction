import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def get_oversampled_data():
    # Load the training data of batch 1:4
    M = np.genfromtxt('main/data/oversampled.data', skip_header=0,
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

if __name__ == '__main__':
    x1, y1 = get_batch_1_1()
    x2, y2 = get_batch_1_2()
    x3, y3 = get_batch_1_3()
    x4, y4 = get_batch_1_4()
    x5, y5 = get_oversampled_data()

    # unique, counts = np.unique(y1, return_counts=True)
    # print("Count in Batch 1 class 0: ", counts[unique == 0][0])
    # print("Count in Batch 1 class 1: ", counts[unique == 1][0])
    # print()

    # unique, counts = np.unique(y2, return_counts=True)
    # print("Count in Batch 2 class 0: ", counts[unique == 0][0])
    # print("Count in Batch 2 class 1: ", counts[unique == 1][0])
    # print()

    # unique, counts = np.unique(y3, return_counts=True)
    # print("Count in Batch 3 class 0: ", counts[unique == 0][0])
    # print("Count in Batch 3 class 1: ", counts[unique == 1][0])
    # print()

    # unique, counts = np.unique(y4, return_counts=True)
    # print("Count in Batch 4 class 0: ", counts[unique == 0][0])
    # print("Count in Batch 4 class 1: ", counts[unique == 1][0])
    # print()

    # # Calculate class counts
    # class_0_counts = [np.sum(y1 == 0), np.sum(y2 == 0), np.sum(y3 == 0), np.sum(y4 == 0)]
    # class_1_counts = [np.sum(y1 == 1), np.sum(y2 == 1), np.sum(y3 == 1), np.sum(y4 == 1)]

    # # Create histogram
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.bar(np.arange(4), class_0_counts, label='Class 0')
    # ax.bar(np.arange(4), class_1_counts, bottom=class_0_counts, label='Class 1')
    # ax.set_xticks(np.arange(4))
    # ax.set_xticklabels(['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4'])
    # ax.set_xlabel('Batch')
    # ax.set_ylabel('Count')
    # ax.set_title('Class Counts by Batch')
    # ax.legend()
    # plt.show()

    # Checking Oversampling data
    
    # unique, counts = np.unique(y5, return_counts=True)
    # print("Count in Oversampled Batch class 0: ", counts[unique == 0][0])
    # print("Count in Oversampled Batch class 1: ", counts[unique == 1][0])
    # print()

    # Calculate class counts
    # class_0_counts = [np.sum(y5 == 0)]
    # class_1_counts = [np.sum(y5 == 1)]

    # # Create histogram
    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.bar(np.arange(4), class_0_counts, label='Class 0')
    # ax.bar(np.arange(4), class_1_counts, bottom=class_0_counts, label='Class 1')
    # ax.set_xticks(np.arange(4))
    # ax.set_xticklabels(['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4'])
    # ax.set_xlabel('Batch')
    # ax.set_ylabel('Count')
    # ax.set_title('Class Counts by Batch')
    # ax.legend()
    # plt.show()