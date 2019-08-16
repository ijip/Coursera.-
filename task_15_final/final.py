import pandas
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

features = pandas.read_csv('features.csv', index_col='match_id') #файл с признаками (train выборка)
test = pandas.read_csv('features_test.csv', index_col='match_id') #test выборка
rows = len(features)
counts = features.count()
skips = counts[counts < rows]
print(skips.sort_values().apply(lambda c: (rows - c) / rows)) #процентное содержание пропусков в столбцах

#обработка данных
features.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1, inplace=True) #удаляем избыточные признаки

#разделяем данные на признаки и целевую переменную
X = features
y = features['radiant_win'].to_frame()
train_y = np.array(y).astype(int)
del features['radiant_win']

#заполняем пропуски нулями
X = X.fillna(0)

#функция очистки 11-ти категориальных признаков в данных
def clean_category(X):
    del X['lobby_type']
    for n in range(1, 6):
        del X['r{}_hero'.format(n)]
        del X['d{}_hero'.format(n)]
    return X

#функция формирования мешка слов
heroes = pandas.read_csv('./data/dictionaries/heroes.csv')
def words_bag(X):
    X_pick = np.zeros((X.shape[0], len(heroes)))
    for i, match_id in enumerate(X.index):
        for p in range(5):
            X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return pandas.DataFrame(X_pick, index=X.index)

#функция построения графика с различными параметрами регуляризации
def plt_plot_C(C_pow_range, scores):
    plt.plot(C_pow_range, scores)
    plt.xlabel('log(C)')
    plt.ylabel('score')
    plt.show()

#функция тестирования логистической регрессии при различных параметрах регуляризации
def test_model(X, y):
    scores = []
    C_pow_range = range(-5, 6)
    C_range = [10.0 ** i for i in C_pow_range]
    for C in C_range:
        start_time = datetime.datetime.now()
        print('C=', str(C))
        model = LogisticRegression(C=C, random_state=42, n_jobs=-1)
        model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
        print(model_scores)
        print('Time elapsed:', datetime.datetime.now() - start_time)
        scores.append(np.mean(model_scores))

    plt_plot_C(C_pow_range, scores)
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    return C_range[max_score_index], max_score

#метод 1: градиентный бустинг
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
nums = [10, 20, 30, 50, 100, 250, 500]
for n in nums:
    print('#', str(n))
    model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    start_time = datetime.datetime.now()
    model_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    print(model_scores)
    scores.append(np.mean(model_scores))
plt.plot(nums, scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()

#метод 2: логистическая регрессия
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
C, score = test_model(X, y)
print('#Тест 1:')
print('Лучший параметр регуляризации: ' + str(C) + '\nПоказатель: ' + str(score))

#удаляем категориальные признаки и повторяем тест
X = pandas.read_csv('features.csv', index_col='match_id')
X.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire', 'radiant_win'], axis=1, inplace=True) #удаляем избыточные признаки
X = X.fillna(0)
X = clean_category(X)
X = scaler.fit_transform(X)
C, score = test_model(X, y)
print('#Тест 2:')
print('Лучший параметр регуляризации: ' + str(C) + '\nПоказатель: ' + str(score))

#тесты с добавлением мешка слов героев
X = pandas.read_csv('features.csv', index_col='match_id')
X.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire', 'radiant_win'], axis=1, inplace=True) #удаляем избыточные признаки
X = X.fillna(0)
test = test.fillna(0)
X_h, test_h = words_bag(X), words_bag(test)
X = pandas.DataFrame(scaler.fit_transform(X), index = X.index)
test = pandas.DataFrame(scaler.fit_transform(test), index = test.index)
X = pandas.concat([X, X_h], axis=1)
test = pandas.concat([test, test_h], axis=1)
C, score = test_model(X, y)
print('#Тест 3:')
print('Лучший параметр регуляризации: ' + str(C) + '\nПоказатель: ' + str(score))
model = LogisticRegression(C=0.1, random_state=42, n_jobs=-1)
model.fit(X, y)
y_test = model.predict_proba(test)[:, 1]
max_score = max(y_test)
min_score = min(y_test)
print('Максимальный показатель на тестовой выборке:')
print('reasult: ' + str(max_score))
print('Минимальный показатель на тестовой выборке:')
print('reasult: ' + str(min_score))
results = pandas.DataFrame({'radiant_win': y_test}, index=test.index)
results.index.name = 'match_id'
results.to_csv('./results.csv')