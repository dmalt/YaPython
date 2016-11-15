import numpy as np
from numpy.linalg import norm


def sigmoid_f(w,X):
    ''' Compute sigmoid function '''

    n_samples = X.shape[0]
    A = np.zeros([n_samples, ])

    for i in range(n_samples):
        A[i] =  1 / (1 + np.exp(-np.inner(w, X[i,:])))
    return A


def logistic_regression_step(X, y, w_orig, k, C):
    ''' Logistic regression step '''
    l = len(y)
    w = np.copy(w_orig)
    sum_w = np.zeros(w.shape)

    for i in range(l):
        scalar = np.inner(w_orig, X[i,:])
        expon = 1 - 1 /(1 + np.exp(-y[i] * scalar))
        delta = y[i] * X[i,:].T * expon 
        sum_w += delta

        # print(delta, scalar, expon, sum_w)

    w += k * (1. / l) * sum_w   - k * C * w_orig
    # print(w, k)
    return w


def train_logistic_regression(X, y, C=0, k=0.1, tol=1e-5, n_iter=10000):
    ''' Logistic regression '''
    w_n = np.zeros([X.shape[1],])
    w_p = np.zeros([X.shape[1],])
    # print(C)
    for i in range(n_iter):
        w_n = logistic_regression_step(X, y, w_p, k, C)
        n = norm(w_n - w_p)
        print(n)
        if n < tol:
            print('Success! (n_iterations = {})'.format(n))
            return w_n, i
        w_p = w_n

    print('Error: timed out. {}'.format(n))
    return None, i

        

if __name__ == '__main__':
    from sklearn.metrics import roc_auc_score
    import pandas as pd

    input_fname = 'data-logistic.csv'
    data = pd.read_csv(input_fname)
    y = data.values[:,0]
    X = data.values[:,1:]
    w_no_reg, n_iter = train_logistic_regression(X, y, C=0)
    probs_no_reg = sigmoid_f(w_no_reg, X)
    AUROC_no_reg = round(roc_auc_score(y,probs_no_reg),3)
    

    w_reg, n_iter = train_logistic_regression(X, y, C=10)
    probs_reg = sigmoid_f(w_reg, X)
    AUROC_reg = round(roc_auc_score(y,probs_reg), 3)
    
    print(AUROC_no_reg)
    print(AUROC_reg)
    
    ans_fname = 'answer_log_reg.txt'
    f = open(ans_fname, 'w')
    f.write('{} {}'.format(AUROC_no_reg, AUROC_reg))
    f.close()
    # print(probs)


