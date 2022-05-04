import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (208608802, 309205219)
        self.label = []
        self.weight = {}
        self.features = []



    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here
        self.X = X
        self.y = y
        self.features=X.shape[1]#number of col=number of feature
        self.label=np.unique(y)
        number_of_sample=len(self.X)
        train_set=np.insert(X, 0, 1, axis=1)#add 1 to the vector x for the bias
        for label in self.label:#crate the weight vector with bias
                self.weight[label]=np.array([float(0) for i in range(self.features+1)])
        change=True
        while change:##stop runing after all the train set classification is correct
            change=False

            for i in range (number_of_sample):
                y_predict=self.label[0]##random prediction at the begining of the itreation
                max_predict=0
                for label in self.label:
                    predicted_scalar=np.dot(self.weight[label], train_set[i, :])
                    if predicted_scalar > max_predict:##look for the best fit
                        max_predict=predicted_scalar
                        y_predict=label

                if y_predict != y[i]:#condition of the algorithem
                    self.weight[y_predict]-= train_set[i, :]
                    self.weight[y[i]]+= train_set[i, :]
                    change=True


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        predicted_class=np.array([int(0) for i in range(X.shape[0])])
        test_set=np.insert(X, 0, 1, axis=1)

        for i in range (X.shape[0]):
            y_predict = self.label[0]
            max_predict = 0
            for label in self.label:
                predicted_scalar = np.dot(self.weight[label],test_set[i, :])
                if predicted_scalar > max_predict:
                    max_predict = predicted_scalar
                    y_predict = label
            predicted_class[i]=y_predict

        return predicted_class
        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":

    print("*" * 20)
    print("Started HW2_208608802_309205219.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)
    np.random.seed(42)
    acc_list = []
    for i in range(10, 150, 10):
        indecies = np.arange(150)
        np.random.shuffle(indecies)
        indecies_train = indecies[:i]
        indecies_test = indecies[i:150]
        X1 = X[indecies_train]
        y1 = y[indecies_train]
        X2 = X[indecies_test]
        y2 = y[indecies_test]
        is_separable = model.fit(X1, y1)
        y_pred = model.predict(X2)
        accuracy = np.sum(y_pred == y2.ravel()) / y.shape[0]
        acc_list.append(accuracy)
    print(acc_list)