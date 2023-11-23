import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

plt.rcParams["font.sans-serif"] = "SimHei"  # 解决中文乱码问题

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

# 检查缺失值
user_info['gender'].replace(2.0, np.nan, inplace=True)
missing_values = user_info.isnull().sum()
print("缺失值统计：")
print(missing_values)

missing_values = user_log.isnull().sum();
print(missing_values)

user_log = user_log.fillna(method='ffill')

# 缺失值处理:使用众数填充
user_info['gender'].replace(2.0, np.nan, inplace=True)
user_info['age_range'].replace(0.0, np.nan, inplace=True)
user_info.fillna(user_info.mode(), inplace=True)

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=10)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# 创建一个决策树基分类器
base_classifier = DecisionTreeClassifier(max_depth=1)

mlp_model = MLPClassifier(solver='lbfgs', activation='relu', alpha=0.1, random_state=0, hidden_layer_sizes=[10, 10])
logit_model = LogisticRegression(solver='liblinear')
tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)
forest_model = RandomForestClassifier(n_estimators=10, random_state=2)
gbrt_model = GradientBoostingClassifier(random_state=0)

# 元模型
meta_model = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, learning_rate=0.1, random_state=0)
# 划分数据集
X_train_base, X_stack, y_train_base, y_stack = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
# 使用基础模型在一部分训练集上训练
mlp_model.fit(X_train_base, y_train_base)
logit_model.fit(X_train_base, y_train_base)
tree_model.fit(X_train_base, y_train_base)
forest_model.fit(X_train_base, y_train_base)
gbrt_model.fit(X_train_base, y_train_base)

# 使用另一部分训练集来预测，得到用于训练元模型的新特征
#stacking_features = stacking_model.predict(X_stack)
mlp_prediction = mlp_model.predict(X_stack)
logit_prediction = logit_model.predict(X_stack)
tree_prediction = tree_model.predict(X_stack)
forest_prediction = forest_model.predict(X_stack)
gbrt_prediction = gbrt_model.predict(X_stack)

predictions_list = [mlp_prediction, logit_prediction, tree_prediction, forest_prediction, gbrt_prediction]  # 代表每个基模型的预测结果
num_models = len(predictions_list)
num_samples = predictions_list[0].shape[0]  # 假设每个模型的预测结果具有相同数量的样本

# 创建一个空的元特征矩阵
stacking_features = np.zeros((num_samples, num_models))

# 填充元特征矩阵的每一列
for i, predictions in enumerate(predictions_list):
    stacking_features[:, i] = predictions  # 将每个模型的预测结果放入相应的列

# 使用元模型进行训练
meta_model.fit(stacking_features, y_stack)

# 在测试集上进行堆叠预测
mlp_prediction = mlp_model.predict(X_test)
logit_prediction = logit_model.predict(X_test)
tree_prediction = tree_model.predict(X_test)
forest_prediction = forest_model.predict(X_test)
gbrt_prediction = gbrt_model.predict(X_test)

predictions_list = [mlp_prediction, logit_prediction, tree_prediction, forest_prediction, gbrt_prediction]  # 代表每个基模型的预测结果
num_models = len(predictions_list)
num_samples = predictions_list[0].shape[0]  # 假设每个模型的预测结果具有相同数量的样本

# 创建一个空的元特征矩阵
stacking_features = np.zeros((num_samples, num_models))

# 填充元特征矩阵的每一列
for i, predictions in enumerate(predictions_list):
    stacking_features[:, i] = predictions  # 将每个模型的预测结果放入相应的列
stacking_predictions = meta_model.predict(stacking_features)

accuracy = accuracy_score(y_test, stacking_predictions)
print(stacking_predictions[:])
print(f"Accuracy of Stacking Model on test set: {accuracy:.8f}")
