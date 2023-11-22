import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=10)



mlp = MLPClassifier(solver='lbfgs', activation='relu', alpha=0.1, random_state=0, hidden_layer_sizes=[10, 10]).fit(
    X_train, y_train)
Predict = mlp.predict(X_test)
Predict_proba = mlp.predict_proba(X_test)
print(Predict_proba[:])
Score = accuracy_score(y_test, Predict)
print(Score)
# df_train = pd.merge(df_train, browse_days_temp1, on=["user_id", "merchant_id"], how="left")
