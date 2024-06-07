import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
df = pd.read_excel('wine.xlsx', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
catboost_model = CatBoostClassifier(verbose=0)
skf = StratifiedKFold(n_splits=5)
acc_scores = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    catboost_model.fit(X_train, y_train)
    y_pred = catboost_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_scores.append(acc)
avg_acc = sum(acc_scores) / len(acc_scores)
print(f'Average Accuracy: {avg_acc}')
y_pred = catboost_model.predict(X)
for index, label in enumerate(y_pred):
    print(f'Sample {index + 1} predicted class: {label}')
dump(catboost_model, 'catboost_model.joblib')