import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码问题
# 清理内存
import gc
import seaborn as sns
import random

# # from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# from sklearn import model_selection
# from sklearn.neighbors import KNeighborsRegressor

df_train = pd.read_csv(r'data/data_format1/train_format1.csv')
df_test = pd.read_csv(r'data/data_format1/test_format1.csv')
user_info = pd.read_csv(r'data/data_format1/user_info_format1.csv')
user_log = pd.read_csv(r'data/data_format1/user_log_format1.csv')

print(df_test.shape, df_train.shape)
print(user_info.shape, user_log.shape)

# 填充缺失值
user_info['age_range'].replace(0.0, np.nan, inplace=True)
user_info['gender'].replace(2.0, np.nan, inplace=True)
user_info['age_range'].replace(np.nan, -1, inplace=True)
user_info['gender'].replace(np.nan, -1, inplace=True)

# 聚合特征
seller_group = user_log.groupby(["seller_id", "action_type"]).count()[["user_id"]].reset_index().rename(
    columns={'user_id': 'count'})

# del user_log
# gc.collect()

# age_range,gender特征添加
df_train = pd.merge(df_train, user_info, on="user_id", how="left")
df_train.head()

total_logs_temp = user_log.groupby([user_log["user_id"], user_log["seller_id"]]).count().reset_index()[
    ["user_id", "seller_id", "item_id"]]
total_logs_temp.rename(columns={"seller_id": "merchant_id", "item_id": "total_logs"}, inplace=True)
df_train = pd.merge(df_train, total_logs_temp, on=["user_id", "merchant_id"], how="left")

# 添加unique_item_ids特征
unique_item_ids_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["item_id"]]).count().reset_index()[
        ["user_id", "seller_id", "item_id"]]
unique_item_ids_temp1 = unique_item_ids_temp.groupby(
    [unique_item_ids_temp["user_id"], unique_item_ids_temp["seller_id"]]).count().reset_index()
unique_item_ids_temp1.rename(columns={"seller_id": "merchant_id", "item_id": "unique_item_ids"}, inplace=True)
df_train = pd.merge(df_train, unique_item_ids_temp1, on=["user_id", "merchant_id"], how="left")

#
categories_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["cat_id"]]).count().reset_index()[
        ["user_id", "seller_id", "cat_id"]]
categories_temp1 = categories_temp.groupby(
    [categories_temp["user_id"], categories_temp["seller_id"]]).count().reset_index()
categories_temp1.rename(columns={"seller_id": "merchant_id", "cat_id": "categories"}, inplace=True)
df_train = pd.merge(df_train, categories_temp1, on=["user_id", "merchant_id"], how="left")

df_train.head(10)
# 添加browse_days特征
browse_days_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["time_stamp"]]).count().reset_index()[
        ["user_id", "seller_id", "time_stamp"]]
browse_days_temp1 = browse_days_temp.groupby(
    [browse_days_temp["user_id"], browse_days_temp["seller_id"]]).count().reset_index()
browse_days_temp1.rename(columns={"seller_id": "merchant_id", "time_stamp": "browse_days"}, inplace=True)
df_train = pd.merge(df_train, browse_days_temp1, on=["user_id", "merchant_id"], how="left")

# 添加one_clicks、shopping_carts、purchase_times、favourite_times特征
one_clicks_temp = \
    user_log.groupby([user_log["user_id"], user_log["seller_id"], user_log["action_type"]]).count().reset_index()[
        ["user_id", "seller_id", "action_type", "item_id"]]
one_clicks_temp.rename(columns={"seller_id": "merchant_id", "item_id": "times"}, inplace=True)
one_clicks_temp["one_clicks"] = one_clicks_temp["action_type"] == 0
one_clicks_temp["one_clicks"] = one_clicks_temp["one_clicks"] * one_clicks_temp["times"]
one_clicks_temp["shopping_carts"] = one_clicks_temp["action_type"] == 1
one_clicks_temp["shopping_carts"] = one_clicks_temp["shopping_carts"] * one_clicks_temp["times"]
one_clicks_temp["purchase_times"] = one_clicks_temp["action_type"] == 2
one_clicks_temp["purchase_times"] = one_clicks_temp["purchase_times"] * one_clicks_temp["times"]
one_clicks_temp["favourite_times"] = one_clicks_temp["action_type"] == 3
one_clicks_temp["favourite_times"] = one_clicks_temp["favourite_times"] * one_clicks_temp["times"]
four_features = one_clicks_temp.groupby(
    [one_clicks_temp["user_id"], one_clicks_temp["merchant_id"]]).sum().reset_index()
four_features = four_features.drop(["action_type", "times"], axis=1)
df_train = pd.merge(df_train, four_features, on=["user_id", "merchant_id"], how="left")

# 缺失值处理
df_train.isnull().sum(axis=0)
df_train = df_train.fillna(method='ffill')

# 模型构建与调参
Y = df_train['label']
X = df_train.drop(['user_id', 'merchant_id', 'label'], axis=1)
# df_train = pd.merge(df_train, browse_days_temp1, on=["user_id", "merchant_id"], how="left")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.55, random_state=10)

# 初始化梯度提升树分类器
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)  # 这里 n_estimators 表示基学习器的数量

# 训练梯度提升树模型
gradient_boosting.fit(X_train, y_train)

# 特征选择
select_features = SelectFromModel(gradient_boosting, threshold='median')  # 这里使用梯度提升树模型进行特征选择
select_features.fit(X_train, y_train)

# 提取重要特征
X_train_selected = select_features.transform(X_train)
X_test_selected = select_features.transform(X_test)

# 使用选择的重要特征重新训练模型
gradient_boosting_selected = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boosting_selected.fit(X_train_selected, y_train)

# 定义参数网格
param_grid = {
    'n_estimators': [15, 25, 50],  # 调整基学习器的数量
    'learning_rate': [0.003, 0.005, 0.01],  # 学习率
    'max_depth': [1, 3, 5]  # 调整树的深度
    # 其他需要调整的参数
}

# 使用 GridSearchCV 进行参数调优
grid_search = GridSearchCV(estimator=gradient_boosting_selected, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

# 输出最佳参数组合和对应的准确率
print("Best Parameters:", grid_search.best_params_)

# 使用最佳参数组合重新训练模型
best_gradient_boosting = grid_search.best_estimator_
best_gradient_boosting.fit(X_train_selected, y_train)

# 对测试集进行预测
predictions = best_gradient_boosting.predict(X_test_selected)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("Gradient Boosting Accuracy after Parameter Tuning:", accuracy)

# choose = ["user_id", "merchant_id", "mlp_prob"]
# res = df_test[choose]
# res.rename(columns={"mlp_prob": "prob"}, inplace=True)
# print(res.head(10))
# res.to_csv(path_or_buf=r"data/prediction.csv", index=False)

pX = df_test.drop(['user_id', 'merchant_id'], axis=1)
pX_selected = select_features.transform(pX)
pPredictions = best_gradient_boosting.predict_proba(pX_selected)
df_test['prob'] = pPredictions
choose = ["user_id", "merchant_id", "label"]
res = df_test[choose]
print(res.head(10))
res.to_csv(path_or_buf=r"data/prediction.csv", index=False)