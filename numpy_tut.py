import numpy as np

normRands = np.random.normal(loc=1., scale=10,
                             size=(1000, 50))
print normRands

# print normRands[:,0 ]

m = np.mean(normRands, axis=0)
std = np.std(normRands, axis=0)

normRands = (normRands - m) / std

Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])

a = Z.sum(axis=1)
b = np.nonzero(a > 10)
print b

E = np.eye(3)
print E

I = np.eye(3)
EI = np.vstack([E, I])
print EI


