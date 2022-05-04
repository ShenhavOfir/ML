import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (208608802, 309205219)
        self.X = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        self.classes = np.array([], dtype=np.float64)


    #Minkowski distance calc func
    @staticmethod
    def minkowski_distance(v1, v2, p):
        import math
        if p < 1:
            raise ValueError("p must be greater than one for minkowski metric")
        # Getting vector 1 size and initializing summing variable
        size, sum = len(v1), 0
        # Adding the p exponent of the difference of the values of the two vectors
        for i in range(size):
            sum += math.pow(abs(v1[i] - v2[i]), p)
        sum_total_a = math.pow(sum, 1/p)

        return sum_total_a


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        # TODO - your code here
        # Initializing x, y & num_of_classes
        self.X = X
        self.y = y
        self.classes = np.unique(y)

    def getNeighbors(self, new_set):
        # Initializing dict of distances and variable with size of training set
        distances, train_length = {}, len(self.X)

        # Calculating the Minkowski distance between the new
        # sample and the values of the training sample
        for i in range(train_length):
            d = self.minkowski_distance(self.X[i], new_set, self.p)
            distances[i] = d

        # Sorting the dict's keys by their values
        k_neighbors = sorted(distances, key=distances.get)

        #intitalizing neighbors list, distances list and a counter
        neighbors = []
        distances_list = []
        count = 0
        for i in k_neighbors:
            if count < self.k:
                neighbors.append(self.y[i])
                distances_list.append(distances[i])
                count+=1
            else:
                break
        return np.array(neighbors),np.array(distances_list)


    def predict_class(self, y: np.ndarray,distances):
        taple_class_and_occurrence = (-1, -1) #class number, occurrence

        for cl in self.classes:
            class_occurrence = len(np.where(y == cl)[0]) #count time of occurrence in the y array
            if class_occurrence > taple_class_and_occurrence[1]:
                taple_class_and_occurrence = (cl, class_occurrence)

            #in a case of a tiebreak between classes with the same number of occurrence
            #choose class with the min distance
            elif class_occurrence >= taple_class_and_occurrence[1] and cl in y:
                info_array = np.array([y, distances])
                d_new_cl = [distances[i] for i in np.where(info_array == cl)][1]
                d_old_cl = [distances[i] for i in np.where(info_array == taple_class_and_occurrence[0])][1]
                if min(d_old_cl) > min(d_new_cl):
                    taple_class_and_occurrence = (cl, class_occurrence)
                #if the 2 distance equal take by lexicographic distance
                elif min(d_old_cl) == min(d_new_cl):
                    taple_class_and_occurrence = (cl, class_occurrence) if cl < taple_class_and_occurrence[0] else taple_class_and_occurrence

        return taple_class_and_occurrence[0]


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        # TODO - your code here
        y_pred = []
        for instance in X:
            instance_neighbor,distances = self.getNeighbors(instance)
            instance_class              = self.predict_class(instance_neighbor,distances)
            y_pred.append(instance_class)
        return np.asarray(y_pred)


        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")


    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
