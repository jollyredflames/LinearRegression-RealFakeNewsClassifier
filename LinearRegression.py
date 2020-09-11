from numpy import ndarray
from random import randint
import matplotlib.pyplot as plt
import numpy as np


def createDataTrainTest(dataTrainX: str, dataTrainY: str, dataTestX: str, dataTestY: str) -> (ndarray, ndarray):
    """Just loads the data into a dictionary with data and targets"""
    data_train = {'X': np.genfromtxt(dataTrainX, delimiter=','), 't': np.genfromtxt(dataTrainY, delimiter=',')}
    data_test = {'X': np.genfromtxt(dataTestX, delimiter=','), 't': np.genfromtxt(dataTestY, delimiter=',')}

    return (data_train, data_test)


def shuffle_data(dataDict: dict) -> dict:
    """shuffles data and returns it. Does not manipulate original entries.
    Shuffling is identical for data and its respective target."""
    data = dataDict['X'].copy()
    targets = dataDict['t'].copy()
    data: ndarray
    targets: ndarray

    lenConst = len(data)
    for _ in range(3 * lenConst):
        do1 = randint(0, lenConst - 1)
        do2 = randint(0, lenConst - 1)
        if do1 != do2:
            data[do1], data[do2] = data[do2], data[do1]
            targets[do1], targets[do2] = targets[do2], targets[do1]
    return {'X': data, 't': targets}


def split_data(dataDict: dict, numFolds: int, fold: int) -> (dict, dict):
    """fold must be an int < len(numFolds).
    Fold is indexed like an array, so fold 0 are the "top" elements in the dataset"""
    numInEachPartition = len(dataDict['X']) // numFolds

    start = fold * numInEachPartition
    stop = start + numInEachPartition

    foldItem = {'X': dataDict['X'][start:stop], 't': dataDict['t'][start: stop]}

    restData = {'X': np.concatenate((dataDict['X'][0:start], dataDict['X'][stop:]), axis=0),
                't': np.concatenate((dataDict['t'][0:start], dataDict['t'][stop:]), axis=0)}

    return (foldItem, restData)


def train_model(data: dict, lambd: float) -> ndarray:
    """Given hyperparameter lambda, compute best fit and return the matrix for best fit on this set of data"""

    X = data['X']
    t = data['t']
    X: ndarray
    t: ndarray
    Xtranspose = X.transpose()

    term1 = np.dot(Xtranspose, X) + lambd * np.identity(len(Xtranspose))
    term2 = np.dot(np.linalg.inv(term1), Xtranspose)

    return np.dot(term2, t)

def predict(data: dict, model: ndarray) -> []:
    """Return an array containing the image made by applying the regression model on the data"""
    predictions = []

    for line in data['X']:
        predictions.append(np.dot(model, line))

    return predictions

def loss(dataDict: dict, model: ndarray) -> float:
    """Computes mean squared error of the model when applied to data"""
    target = dataDict['t']
    n = len(target)
    target: ndarray
    predictions = predict(dataDict, model)

    term1 = target - predictions
    numerator = np.linalg.norm(term1, 2) ** 2

    return numerator / n


def cross_validation(data: dict, num_folds: int, lambd_seq: ndarray) -> [float]:
    """Perform a num_folds corss validation on the data"""
    data = shuffle_data(data)
    cvErrorList = []
    for num in lambd_seq:
        cv_loss_at_num = 0
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, num)
            cv_loss_at_num += loss(val_cv, model)
        cvErrorList.append(cv_loss_at_num / num_folds)
    return cvErrorList


def createSeq(start: float, stop: float, len_arr: int) -> ndarray:
    """Creates a sequence of evenly spaced floats which start at start and stop at stop. length of return is len_arr"""
    return np.linspace(start, stop, len_arr)


if __name__ == "__main__":

    data_train, data_test = createDataTrainTest("data_train_X.csv", "data_train_Y.csv", "data_test_X.csv",
                                                "data_test_Y.csv")
    lambd_seq = createSeq(0.02, 1.5, 50)

    errorTrain = []
    # modelTest = []
    errorTest = []

    for num in lambd_seq:
        model = train_model(data_train, num)
        # modelTest.append(train_model(data_test, num))
        errorTrain.append(loss(data_train, model))
        errorTest.append(loss(data_test, model))

    fold5Results = cross_validation(data_train, 5, lambd_seq)
    fold10Results = cross_validation(data_train, 10, lambd_seq)

    print([round(x, 4) for x in errorTrain])
    print([round(y, 4) for y in errorTest])

    print("Best Lambda 5Fold: ", lambd_seq[fold5Results.index(min(fold5Results))], " With Error: ", min(fold5Results))
    print("Best Lambda 10Fold: ", lambd_seq[fold10Results.index(min(fold10Results))], " With Error: ", min(fold10Results))
    print("Best Lambda test: ", lambd_seq[errorTest.index(min(errorTest))], " With Error: ", min(errorTest))
    print("Best Lambda train: ", lambd_seq[errorTrain.index(min(errorTrain))], " With Error: ", min(errorTrain))

    plt.plot(lambd_seq, errorTrain, '-o', label="Train Error")
    plt.plot(lambd_seq, errorTest, '-o', label="Test Error")
    plt.plot(lambd_seq, fold5Results, '-o', label="5 Fold Error")
    plt.plot(lambd_seq, fold10Results, '-o', label="10 Fold Error")
    plt.xlabel("Hyperparameter value")
    plt.ylabel("Mean Squared Error")

    plt.legend()
    plt.show()

    # Best Lambda 5Fold: 1.5 WithError: 4.556
    # Best Lambda 10Fold: 1.5 With Error: 4.891
    # Best Lambda test: 0.47306122448979593 With Error: 2.191
    # Best Lambda train: 0.02 With Error: 0.0497
    # Error Train = [0.05, 0.106, 0.154, 0.198, 0.238, 0.277, 0.313, 0.349, 0.383, 0.417, 0.45, 0.482, 0.513, 0.544,
    #                0.574, 0.604, 0.633, 0.661, 0.69, 0.717, 0.744, 0.771, 0.798, 0.824, 0.849, 0.875, 0.899, 0.924,
    #                0.948, 0.972, 0.995, 1.019, 1.041, 1.064, 1.086, 1.108, 1.13, 1.151, 1.172, 1.193, 1.213, 1.233,
    #                1.253, 1.273, 1.293, 1.312, 1.331, 1.349, 1.368, 1.386]
    # Error Test = [5.107, 3.631, 3.07, 2.771, 2.587, 2.466, 2.382, 2.323, 2.28, 2.249, 2.227, 2.211, 2.201, 2.195,
    #               2.191, 2.191, 2.192, 2.195, 2.2, 2.206, 2.212, 2.22, 2.228, 2.237, 2.246, 2.255, 2.265, 2.275,
    #               2.285, 2.296, 2.306, 2.317, 2.327, 2.338, 2.349, 2.36, 2.37, 2.381, 2.392, 2.402, 2.413, 2.423,
    #               2.434, 2.444, 2.455, 2.465, 2.475, 2.485, 2.495, 2.505]
