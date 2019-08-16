import pandas as pd
import math
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('gbm-data.csv').values
y = df[:, 0]
X = df[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
lrange = [1, 0.5, 0.3, 0.2, 0.1]
results = {}

def logLoss(clf, X, y):
    results = []
    for pred in clf.staged_decision_function(X):
        results.append(log_loss(y, [1.0 / (1.0 + math.exp(-y_pred)) for y_pred in pred]))

    return results

def plotLoss(learning_rate, test_loss, train_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig('plotsrate_' + str(learning_rate) + '.png')

    min_lvalue = min(test_loss)
    min_lindex = test_loss.index(min_lvalue)
    return min_lvalue, min_lindex

for lr in lrange:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    clf.fit(X_train, y_train)
    train_loss = logLoss(clf, X_train, y_train)
    test_loss = logLoss(clf, X_test, y_test)
    results[lr] = plotLoss(lr, test_loss, train_loss)

print('overfitting')
min_lvalue, min_lindex = results[0.2]
print(round(min_lvalue, 2), min_lindex)
clf = RandomForestClassifier(n_estimators=min_lindex, random_state=241)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)[:, 1]
tree_loss = log_loss(y_test, y_pred)
print(round(tree_loss, 2))


