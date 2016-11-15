import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale

fname = 'wine.data'
data = pd.read_csv(fname)

Y = data.values[:, 0]
X = data.values[:, 1:]

cv = KFold(n=X.shape[0], n_folds=5, random_state=42, shuffle=True)
scores = []
for n_neighbors in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores.append(
        np.mean(cross_val_score(estimator=neigh,
                                X=X, y=Y, cv=cv, scoring='accuracy')))
result = round(max(scores), 2)
k_max = scores.index(max(scores)) + 1

X1 = scale(X)
scores1 = []
for n_neighbors in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores1.append(
        np.mean(cross_val_score(estimator=neigh,
                                X=X1, y=Y, cv=cv, scoring='accuracy')))
result1 = round(max(scores1), 2)
k_max1 = scores1.index(max(scores1)) + 1

fname1 = '1.txt'
fname2 = '2.txt'
fname3 = '3.txt'
fname4 = '4.txt'

f1 = open(fname1, 'wb')
f1.write(str(k_max))
f1.close()

f2 = open(fname2, 'wb')
f2.write(str(result))
f2.close()

f3 = open(fname3, 'wb')
f3.write(str(k_max1))
f3.close()

f4 = open(fname4, 'wb')
f4.write(str(result1))
f4.close()