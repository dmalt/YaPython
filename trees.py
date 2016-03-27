import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('titanic.csv', index_col='PassengerId')
features = ['Pclass', 'Fare', 'Age', 'Sex']
Pclasses = data[features[0]]
Fares = data[features[1]]
Ages = data[features[2]]
Sexes = (data[features[3]] == 'male').astype(int)
X = np.array([Pclasses, Fares, Ages, Sexes]).T

Y = data['Survived']
nanMask =  ~np.isnan(X).any(axis=1)
X = X[nanMask]
Y = Y[nanMask]

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X,Y)

importances = clf.feature_importances_
print importances
sortIndex = [i[0] for i in sorted(enumerate(importances), reverse=True, key = lambda x: x[1])]
topFeatures = [features[i] for i in sortIndex[:2]]
fname = 'trees.txt'
f = open(fname, 'w')
for item in topFeatures:
  f.write(item + ' ')
f.close()