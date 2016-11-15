import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import scale
import sklearn.datasets as dset

boston = dset.load_boston()
X = boston.data
y = boston.target

X = scale(X)

p_list = np.linspace(1, 10, 200)

cv = KFold(n=X.shape[0], n_folds=5, shuffle=True, random_state=42)
scores = []
for p in p_list:
    neigh = KNeighborsRegressor(n_neighbors=5, metric='minkowski', weights='distance', p=p)
    scores.append(np.mean(cross_val_score(neigh, X, y, scoring='neg_mean_squared_error', cv=cv)))

p_max = round(p_list[scores.index(max(scores))], 1)
# fname = 'boston.txt'
# f = open(ans_fname,'wb')
# f.write(str(p_max))
# f.close()
