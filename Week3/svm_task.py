import pandas as pd
from sklearn.svm import SVC

fname = 'svm-data.csv'
data = pd.read_csv(fname, header=None)

y = data[0]
X = data[[1, 2]]
clf = SVC(kernel='linear', C=10000, random_state=241)
clf.fit(X, y)
print clf.support_


cols = []
for yi in y:
    if yi:
        cols.append('r')
    else:
        cols.append('b')
import matplotlib.pyplot as plt

X = X.get_values()
plt.scatter(X[:,0], X[:,1], c = cols)
plt.show()