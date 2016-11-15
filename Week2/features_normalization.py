from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Example of Perceptron usage
# X = np.array([[1,2], [3,4], [5,6]])
# y = np.array([0, 1, 0])
# clf = Perceptron()
# clf.fit(X,y)
# predictions = clf.predict(X)

# Standard scaler example:
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
# X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

fname_train = 'perceptron-train.csv'
fname_test = 'perceptron-test.csv'
data_train = pd.read_csv(fname_train, header=None)
data_test = pd.read_csv(fname_test, header=None)

X_train = data_train[[1, 2]]
y_train = data_train[0]

X_test = data_test[[1, 2]]
y_test = data_test[0]

clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score_1 = accuracy_score(y_test, y_pred)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
y_pred_scaled = clf.predict(X_test_scaled)
score_2 = accuracy_score(y_test, y_pred_scaled)

answer = round(score_2 - score_1, 3)

ans_fname = 'features_normalization.txt'
f = open(ans_fname, 'w')
f.write(str(answer))
f.close()
