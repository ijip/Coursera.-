import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import sys
sys.path.append("..")

data = pd.read_csv('wine.data', header=None)
y = data[0]
X = data.iloc[:, 1:]
kf = KFold(n_splits=5, random_state=42, shuffle=True)


def test_accuracy(kf, X, y):
    scores = list()
    k_range = range(1, 51)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='accuracy'))

    return pd.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)


accuracy = test_accuracy(kf, X, y)
top_accuracy = accuracy.head(1)
print(1, top_accuracy.index[0])
print(2, round(top_accuracy.values[0], 2))

X = sklearn.preprocessing.scale(X)
accuracy = test_accuracy(kf, X, y)
top_accuracy = accuracy.head(1)
print(3, top_accuracy.index[0])
print(4, round(top_accuracy.values[0], 2))