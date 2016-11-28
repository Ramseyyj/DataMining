import numpy as np
import math

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
#
# # print(a[:, 0])
# b = np.array([0.697, 0.774, 0.634, 608, 0.556, 0.403, 0.481, 0.437])
# print(np.var(b))


# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(np.std(x, ddof=1))

a = math.log(0.001, 0.5)
b = a / math.log(2)
print(b)

