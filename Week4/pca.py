import pandas as pd
from sklearn.decomposition import PCA
from numpy import corrcoef


pr_fname = 'close_prices.csv'
pr_dataframe = pd.read_csv(pr_fname)
pca = PCA(n_components=10)
data = pr_dataframe.values[:,1:]

pca.fit(data)

data_transf = pca.transform(data)

dj_fname = 'djia_index.csv'
dj_dataframe = pd.read_csv(dj_fname)
dj_data = dj_dataframe.values[:,1].astype(float)

cor_mat = corrcoef(dj_data, data_transf[:,0])
cor_coef = round(cor_mat[0,1], 2)
print(cor_coef)

