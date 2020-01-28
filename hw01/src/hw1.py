'''
hw1.py
Author: TODO

Tufts COMP 135 Intro ML

'''

import numpy as np
import numpy.random as rd
import math
import queue as Q


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    # Get number of rows for the input Matirx
    L = len(x_all_LF)
    # Get number of columns for the input Matrix
    F = len(x_all_LF[0])

    # print(x_all_LF)
    # Create a deep copy of the input Matrix
    x_copy = rd.permutation(x_all_LF.copy()) if (random_state == None) else rd.RandomState(seed=random_state).permutation(x_all_LF.copy());
    # print("There are :", L, "Instances of", F, "dimensional data", "\nShuffle: \n", x_copy);

    # Determine the number of entries needed for testing
    x_test_N = math.ceil(frac_test * L)
    # Determien the number of entries needed for training
    x_train_M = L - x_test_N

    # print("N is:", x_test_N, "M is:", x_train_M);

    x_test_NF = (x_copy[0: x_test_N, :])
    # print("test_NF: \n", x_test_NF)
    x_train_MF = (x_copy[x_test_N:, :])
    # print("train_NF: \n", x_train_MF)
    return x_train_MF, x_test_NF

class Neighbor(object):
    def __init__(self, distance, data):
        self.distance = distance
        self.data = data

    def __cmp__(self, other):
        return cmp(self.distance, other.distance)

    def __lt__(self, other):
        return self.distance < other.distance

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    F_dim = len(query_QF[0])
    Q_dim = len(query_QF)
    N_dim = len(data_NF)

    Three_Dim_List = []
    for i in query_QF:
        pq = Q.PriorityQueue()
        for j in data_NF:
            dist = calc_distance(i, j)
            print("Distance:", dist, "data: ", j )
            pq.put(Neighbor(dist, j))

        K_F_Lis = []
        for i in range(K):
            K_F_Lis.append(pq.get().data)
            K_F_Dim_Array = np.array(K_F_Lis)
        K_F_Dim_Array.reshape(K, F_dim)
        Three_Dim_List.append(K_F_Dim_Array)
    Three_Dim_Array = np.array(Three_Dim_List)
    Three_Dim_Array = np.array(Three_Dim_List)
    print(Three_Dim_Array)
    print(Three_Dim_Array.shape)
    return Three_Dim_Array

def calc_distance(X, Y):
    sum = 0
    for x, y in zip(X,Y
        sum += (x - y)**2
    return math.sqrt(sum);
