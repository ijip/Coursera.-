import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data_classes = pd.read_csv('classification.csv')
y_true = data_classes['true']
y_pred = data_classes['pred']
conf_matrix = confusion_matrix(y_true, y_pred)
TP, FP, FN, TN = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]
print(TP, FP, FN, TN)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2))

data_scores = pd.read_csv('scores.csv')
y_strue = data_scores['true']
scores_logreg = data_scores['score_logreg']
scores_svm = data_scores['score_svm']
scores_knn = data_scores['score_knn']
scores_tree = data_scores['score_tree']
roc = dict.fromkeys(['score_logreg', 'score_svm', 'score_knn', 'score_tree'], 0)
roc['score_logreg'] = roc_auc_score(y_strue, scores_logreg)
roc['score_svm'] = roc_auc_score(y_strue, scores_svm)
roc['score_knn'] = roc_auc_score(y_strue, scores_knn)
roc['score_tree'] = roc_auc_score(y_strue, scores_tree)
max_value = max(roc.values())
final_roc = {k: v for k, v in roc.items() if v == max_value}
print(final_roc.keys())
pr_scrs = {}
for i in data_scores.columns[1:]:
    pr_crv = precision_recall_curve(data_scores['true'], data_scores[i])
    pr_crv_df = pd.DataFrame({'precision': pr_crv[0], 'recall': pr_crv[1]})
    pr_scrs[i] = pr_crv_df[pr_crv_df['recall'] >= 0.7]['precision'].max()

print(pd.Series(pr_scrs).sort_values(ascending=False).head(1).index[0])
