import pandas as pd
from sklearn.svm import SVC


data = pd.read_csv('svm-data.csv', header=None)
y = data[0]
X = data.iloc[:, 1:]
clf = SVC(C=1000, random_state=241, kernel='linear')
clf.fit(X, y)
print(clf.support_)

