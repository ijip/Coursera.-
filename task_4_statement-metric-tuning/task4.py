import sklearn
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

boston = load_boston()
X = boston.data
y = boston.target
X = sklearn.preprocessing.scale(X)
kf = KFold(n_splits=5, random_state=42, shuffle=True)

def test_pParameter(kf, X, y):
    scores = list()
    p_range = np.linspace(1, 10, 200)
    for p in p_range:
        model = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance')
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error'))

    return pd.DataFrame(scores, p_range).mean(axis=1).sort_values(ascending=False)

pParameter = test_pParameter(kf, X, y)
print(pParameter.head(1))