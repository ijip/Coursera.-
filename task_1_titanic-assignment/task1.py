import pandas as pd

data = pd.read_csv('titanic.csv')
print(data.groupby(['Sex'])['PassengerId'].count()['male'], data.groupby(['Sex'])['PassengerId'].count()['female'])
live = data.groupby(['Survived'])['PassengerId'].count()[1]
all = data.groupby(['Survived'])['PassengerId'].count().sum()
pclass = data.groupby(['Pclass'])['PassengerId'].count()[1]
print(round(live*100/all, 2))
print(round(pclass*100/all, 2))
print(round(data['Age'].mean(), 2), data['Age'].median())
x, y = data['Parch'], data['SibSp']
print(round(y.corr(x), 2))
data2 = data[data.Sex == 'female']['Name']
C = []
for i in data2:
  if '(' in i:
    if ')' in i.split('(')[1].split(' ')[0]:
      C.append(i.split('(')[1].split(' ')[0].split(')')[0])
    else:
      C.append(i.split('(')[1].split(' ')[0])
  else:
    C.append(i.split('. ')[1].split(' ')[0])
print(pd.DataFrame.from_dict(C)[0].value_counts())


