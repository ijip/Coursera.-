import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('titanic.csv')
data = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data = data.dropna()
y = data['Survived']
data = data.drop(['Survived'], axis=1)
data = data.replace(['male', 'female'], [0, 1])
print(y.head())
print(data.head())
clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, y)
importances = clf.feature_importances_
print(importances)
