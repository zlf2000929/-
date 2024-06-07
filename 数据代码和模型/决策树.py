import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 读取Excel文件
df = pd.read_excel('wine.xlsx', header=None)

# 分离特征和标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 初始化决策树分类器
dtc = DecisionTreeClassifier()

# 初始化五折交叉验证
skf = StratifiedKFold(n_splits=5)

# 存储每次迭代的评估指标
acc_scores = []
acc_ = -1
a = 0
# 执行五折交叉验证
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 训练决策树分类器
    dtc.fit(X_train, y_train)
    # 进行预测
    y_pred = dtc.predict(X_test)
    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    # 将评估指标添加到相应的列表中
    acc_scores.append(acc)
    if acc_ <= acc:
        a = dtc
# 计算平均评估指标
avg_acc = sum(acc_scores) / len(acc_scores)
# 输出平均评估指标
print(f'Average Accuracy: {avg_acc}')

# 标准化整个数据集
X_scaled = scaler.transform(X)
# 进行预测
y_pred = a.predict(X_scaled)
# 输出每条数据的预测类别
for index, label in enumerate(y_pred):
    print(f'Sample {index + 1} predicted class: {label}')

from joblib import dump
# 保存模型和标准化器到本地
dump(dtc, 'dtc_model.joblib')