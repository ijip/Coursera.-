import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data = pd.read_csv('salary-train.csv')
data['FullDescription'] = data['FullDescription'].map(lambda x: x.lower())
data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
vectorizer = TfidfVectorizer(min_df=5)
X_fidV = vectorizer.fit_transform(data['FullDescription'])
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_dictV = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X = hstack([X_fidV, X_dictV])
y = data['SalaryNormalized']
clf = Ridge(alpha=1, random_state=241)
clf.fit(X, y)
data_pred = pd.read_csv('salary-test-mini.csv')
data_pred['FullDescription'] = data_pred['FullDescription'].map(lambda x: x.lower())
data_pred['FullDescription'] = data_pred['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
X_fidV = vectorizer.transform(data_pred['FullDescription'])
X_dictV = enc.transform(data_pred[['LocationNormalized', 'ContractTime']].to_dict('records'))
X = hstack([X_fidV, X_dictV])
y = clf.predict(X)
print(round(y[0], 2), round(y[1], 2))