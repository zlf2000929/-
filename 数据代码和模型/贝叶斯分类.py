import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
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

# 初始化高斯贝叶斯分类器
gnb = GaussianNB()

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
    # 训练贝叶斯分类器
    print(X_train)
    print(y_train)
    print("TEXT",y_test)

    gnb.fit(X_train, y_train)
    # 进行预测
    y_pred = gnb.predict(X_test)
    print("PRED", y_pred)
    # 计算评估指标
    acc = accuracy_score(y_test, y_pred)
    # 将评估指标添加到相应的列表中
    acc_scores.append(acc)
    print(acc)
    if acc_ <= acc:
        a = gnb
# 计算平均评估指标
avg_acc = sum(acc_scores) / len(acc_scores)
# 输出平均评估指标
print(f'Average Accuracy: {avg_acc}')
# 使用原始数据集（未标准化）进行预测，因为高斯贝叶斯假设输入是符合高斯分布的
X_original = df.iloc[:, :-1].values
y_pred = a.predict(X_original)
# 输出每条数据的预测类别
for index, label in enumerate(y_pred):
    print(f'Sample {index + 1} predicted class: {label}')

from joblib import dump
# 保存模型到本地
dump(a, 'gnb_model.joblib')