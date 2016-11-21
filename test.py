import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
#
# # print(a[:, 0])
# b = np.array([0.697, 0.774, 0.634, 608, 0.556, 0.403, 0.481, 0.437])
# print(np.var(b))


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(np.std(x, ddof=1))


