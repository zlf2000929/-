import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# 从Excel文件加载数据
df = pd.read_excel('wine.xlsx', header=None)

# 分离特征和标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 将类别标签从 [1 2 3] 转换为 [0 1 2]
y = y - 1

# 初始化LightGBM分类器
lgbm_model = LGBMClassifier()

# 初始化分层交叉验证
skf = StratifiedKFold(n_splits=5)

# 用于存储准确度的列表
acc_scores = []

# 分层交叉验证循环
for train_index, test_index in skf.split(X, y):
    # 分割训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    lgbm_model.fit(X_train, y_train)

    # 预测测试集
    y_pred = lgbm_model.predict(X_test)

    # 计算并保存准确度
    acc = accuracy_score(y_test, y_pred)
    acc_scores.append(acc)

# 计算平均准确度
avg_acc = sum(acc_scores) / len(acc_scores)
print(f'平均准确度: {avg_acc}')

# 使用整个数据集进行最终预测
y_pred = lgbm_model.predict(X)

# 输出每个样本的预测类别
for index, label in enumerate(y_pred):
    print(f'样本 {index + 1} 的预测类别: {label + 1}')  # 转换回原始的类别标签

# 保存模型
dump(lgbm_model, 'lgbm_model.joblib')