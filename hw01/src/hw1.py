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

    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''

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
    # = np.arange(Q_dim * K * F_dim).reshape(Q_dim, K, F_dim)
    # print("Dimension of 3D array: ", Three_Dim_Array.shape)

    # pq = Q.PriorityQueue(K)
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
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''



def calc_distance(X, Y):
    sum = 0
    for x, y in zip(X,Y):
        sum += (x - y)**2
    return math.sqrt(sum);
