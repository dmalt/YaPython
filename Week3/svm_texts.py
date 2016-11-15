import numpy as np
import sklearn
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
# from sklearn.model_selection import train_test_split


newsgroups = datasets.fetch_20newsgroups(
        subset = 'all',
        categories = ['alt.atheism', 'sci.space']
        )

data = newsgroups.data
target = newsgroups.target

# data_train, data_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# ----------- Create features from texts ------- #
tt = TfidfVectorizer()

# X_train = tt.fit_transform(data_train)
# X_test = tt.transform(data_test)
X = tt.fit_transform(data)
y = target
# ----------------------------------------------- #

# ---- Train a classifier and do gridsearch ----------- #
grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X,y)

# Train a classifier with the best performance
# parameter found by the grid search
C_max = max(gs.grid_scores_).parameters['C']
clf_max = SVC(kernel='linear', random_state=241, C=C_max)
clf_max.fit(X,y)

# print(clf_max.coef_)

# Find 10 most relevant words for the classification
ff = np.array(tt.get_feature_names())
idx = np.argpartition(abs(clf_max.coef_.data), -10)[-10:]
idx1 = clf_max.coef_.indices[idx]
words = ff[idx1]
words = sorted(words)
print('10 words that make the biggest difference between space travelling and religion')
print(' ----------------------------------------------------------------------------- ')

for i, word in enumerate(words):
    print('{i}. {word}'.format(i=i + 1, word=word))

# Output answer to a file
ans_fname = 'answer_svm_texts.txt'
f = open(ans_fname, 'w')
f.write(' '.join(words))
f.close()

