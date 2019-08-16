import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('perceptron-train.csv', header=None)
data_test = pd.read_csv('perceptron-test.csv', header=None)
y_train = data_train[0]
y_test = data_test[0]
X_train = data_train.iloc[:, 1:]
X_test = data_test.iloc[:, 1:]
model = Perceptron(random_state=241)
model.fit(X_train, y_train)

accuracy_before = accuracy_score(y_test, model.predict(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
accuracy_after = accuracy_score(y_test, model.predict(X_test_scaled))

print(accuracy_after - accuracy_before)