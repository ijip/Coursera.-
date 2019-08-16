import pandas as pd
from sklearn.decomposition import PCA
import numpy

data = pd.read_csv('close_prices.csv')
X = data.iloc[:, 1:]
pca = PCA(n_components=10)
pca.fit(X.values)
sum = 0
n = 0

for i in pca.explained_variance_ratio_:
    n += 1
    sum += i
    if sum >= 0.9:
        break

print(n)
df = pd.DataFrame(pca.transform(X))
component1 = df[0]
df_dj = pd.read_csv('djia_index.csv')
y = df_dj['^DJI']
print(round(numpy.corrcoef(component1, y)[1, 0], 2))
component1 = pd.Series(pca.components_[0])
company1 = component1.sort_values(ascending=False).head(1).index[0]
print(X.columns[company1])