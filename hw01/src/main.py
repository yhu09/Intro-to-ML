from hw1 import *
import heapq

# x_LF = np.arange(35).reshape(5,7)
# train_MF, test_NF = split_into_train_and_test(
# x_LF, frac_test=0.3, random_state=None)
# print(train_MF.shape)
# print(test_NF.shape)
# print("train_MF:\n", train_MF)
# print("test_NF:\n", test_NF)


data_NF = np.array([[2.13, 5.47, 4.34, 5.23, 2.34],
                    [5.13, 6.47, 7.34, 9.23, 9.34],
                    [3.13, 1.47, 7.34, 9.23, 2.34]])
#
query_QF = np.array([[2.12, 5.42, 4.32, 5.22, 2.32], [2.12, 5.42, 4.32, 5.22, 2.32], [2.12, 5.42, 4.32, 5.22, 2.32]])

NxF = np.array([[2,2], [4,4], [6,6]])
QxF = np.array([[7,7], [2,1], [4,4]])

result = calc_k_nearest_neighbors(NxF, QxF, K = 3)
# print(result)
# li = [[2.13, 5.47, 4.34, 5.23, 2.34],
#  [3.13, 1.47, 7.34, 9.23, 2.34],
#  [3.13, 1.47, 7.34, 9.23, 2.34]]
#
# calc_distance(X, Y)
# heapq.heapify(li);
# print("heapified list: \n",data_NF)
