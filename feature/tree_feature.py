import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from libs.common import my_agg, reduce_mem_usage, clean_error_month

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_data')

train_path = os.path.join(data_dir, 'used_car_train_20200313.csv')
test_path = os.path.join(data_dir, 'used_car_testB_20200421.csv')

output_train = os.path.join(output_dir, 'train_tree.csv')
output_test = os.path.join(output_dir, 'test_tree.csv')

df_train = pd.read_csv(train_path, sep=' ')
df_train['is_train'] = 1
df_test = pd.read_csv(test_path, sep=' ')
df_test['is_train'] = 0

date_feature = ['regDate', 'creatDate']
for feature in date_feature:
    df_train[feature] = df_train[feature].apply(clean_error_month)
    df_test[feature] = df_test[feature].apply(clean_error_month)

# 缺失值填充
df_train['model'] = df_train['model'].fillna(0)
df_train['bodyType'] = df_train['bodyType'].fillna(0)
df_train['fuelType'] = df_train['fuelType'].fillna(0)
df_train['gearbox'] = df_train['gearbox'].fillna(0)
df_train['power'] = df_train['power'].where(df_train['power'] <= 600, 600)
df_train['notRepairedDamage'] = df_train['notRepairedDamage'].replace('-', '0').astype('float')

df_test['bodyType'] = df_test['bodyType'].fillna(0)
df_test['fuelType'] = df_test['fuelType'].fillna(0)
df_test['gearbox'] = df_test['gearbox'].fillna(0)
df_test['power'] = df_test['power'].where(df_test['power'] <= 600, 600)
df_test['notRepairedDamage'] = df_test['notRepairedDamage'].replace('-', '0').astype('float')

# 删除异常数据
# 使用【Q1-1.5IQR,Q3+1.5IQR】
# 数值型变量
numeric_features = ['power', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4',
                    'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14']

# 类别型变量
categorical_features = ['name', 'kilometer', 'model', 'brand', 'bodyType',
                        'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode']
# 日期型变量
date_features = ['regDate', 'creatDate']

# 异常数据处理
rules = []
for feature in numeric_features:
    Q1 = df_train[feature].quantile(0.25)
    Q3 = df_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    min_border = Q1 - 3 * IQR
    max_border = Q3 + 3 * IQR
    rules.append((feature, min_border, max_border))

for rule in rules:
    feature, min_border, max_border = rule
    df_train = df_train[(df_train[feature] <= max_border)
                        & (df_train[feature] >= min_border)]
print(rules)
print(df_train.shape)

# 合并数据集
df = pd.concat([df_train, df_test], axis=0)
df.head()

# 租金要做log变换
df['price'] = np.log(df['price'])
# 开始特征工程处理
df['name_count'] = df.groupby('name')['SaleID'].transform('count')
name_price_df = my_agg(df[df['is_train'] == 1], 'name')
df = pd.merge(df, name_price_df, on='name', how='left')

df['region_count'] = df.groupby('regionCode')['SaleID'].transform('count')
region_price_df = my_agg(df[df['is_train'] == 1], 'regionCode')
df = pd.merge(df, region_price_df, on='regionCode', how='left')
# df.head()

df['model_count'] = df.groupby('model')['SaleID'].transform('count')
model_price_df = my_agg(df[df['is_train'] == 1], 'model')
df = pd.merge(df, model_price_df, on='model', how='left')
# df.head()


df['brand_count'] = df.groupby('brand')['SaleID'].transform('count')
brand_price_df = my_agg(df[df['is_train'] == 1], 'brand')
df = pd.merge(df, brand_price_df, on='brand', how='left')

df['bodyType_count'] = df.groupby('bodyType')['SaleID'].transform('count')
bodyType_price_df = my_agg(df[df['is_train'] == 1], 'bodyType')
df = pd.merge(df, bodyType_price_df, on='bodyType', how='left')
tmp_df = pd.get_dummies(df['bodyType'], prefix='bodyType')
df = pd.concat([df, tmp_df], axis=1)

df['fuelType_count'] = df.groupby('fuelType')['SaleID'].transform('count')
fuelType_price_df = my_agg(df[df['is_train'] == 1], 'fuelType')
df = pd.merge(df, fuelType_price_df, on='fuelType', how='left')
tmp_df = pd.get_dummies(df['fuelType'], prefix='fuelType')
df = pd.concat([df, tmp_df], axis=1)

df['gearbox_count'] = df.groupby('gearbox')['SaleID'].transform('count')
gearbox_price_df = my_agg(df[df['is_train'] == 1], 'gearbox')
df = pd.merge(df, gearbox_price_df, on='gearbox', how='left')
tmp_df = pd.get_dummies(df['gearbox'], prefix='gearbox')
df = pd.concat([df, tmp_df], axis=1)

df['kilometer_count'] = df.groupby('kilometer')['SaleID'].transform('count')
kilometer_price_df = my_agg(df[df['is_train'] == 1], 'kilometer')
df = pd.merge(df, kilometer_price_df, on='kilometer', how='left')

df['notRepairedDamage_count'] = df.groupby('notRepairedDamage')['SaleID'].transform('count')
notRepairedDamage_price_df = my_agg(df[df['is_train'] == 1], 'notRepairedDamage')
df = pd.merge(df, notRepairedDamage_price_df, on='notRepairedDamage', how='left')
tmp_df = pd.get_dummies(df['notRepairedDamage'], prefix='notRepairedDamage')
df = pd.concat([df, tmp_df], axis=1)

# 日期处理
df['used_days'] = (pd.to_datetime(df['creatDate'], format='%Y%m%d') -
                   pd.to_datetime(df['regDate'], format='%Y%m%d')).dt.days
df['used_years'] = round(df['used_days'] / 365, 1)
df['kilometer_div_years'] = df['kilometer'] / df['used_years']
df['kilometer_div_days'] = df['kilometer'] / df['used_days']

# 对使用天数进行分箱
df['use_days_bin_20'] = pd.qcut(df['used_days'], 20, labels=False)
use_days_bin_20_price_df = my_agg(df[df['is_train'] == 1], 'use_days_bin_20')
df = pd.merge(df, use_days_bin_20_price_df, on='use_days_bin_20', how='left')

# 匿名特征处理
features_list = ['v_0', 'v_2', 'v_8', 'v_12', 'v_3']
for i in features_list:
    feature = f'box_{i}'
    df[f'box_{i}'] = pd.qcut(df[i], 20, duplicates='drop', labels=False)
    tmp_df = my_agg(df[df['is_train'] == 1], feature)
    df = pd.merge(df, tmp_df, on=feature, how='left')
    del df[f'box_{i}']

# 匿名特征组合
for i in range(15):
    for j in range(i + 1, 15):
        df[f'v_{i}_add_v_{j}'] = df[f'v_{i}'] + df[f'v_{j}']
        df[f'v_{i}_minus_v_{j}'] = df[f'v_{i}'] - df[f'v_{j}']
        df[f'v_{i}_multiply_v_{j}'] = df[f'v_{i}'] * df[f'v_{j}']
        df[f'v_{i}_div_v_{j}'] = df[f'v_{i}'] / df[f'v_{j}']

df_train = df[df['is_train'] == 1]
df_test = df[df['is_train'] == 0]
y_train = df_train['price']

enc = TargetEncoder(cols=['name', 'regionCode', 'model', 'brand'])

# 高基离散特征编码
df_train = enc.fit_transform(df_train, y_train)
df_test = enc.transform(df_test)

# 删除无效编码
delete_features = ['SaleID', 'regDate', 'offerType', 'seller', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage',
                   'creatDate', 'is_train']
for feature in delete_features:
    del df_train[feature]
    del df_test[feature]
del df_test['price']

for column in df_train.columns:
    print(column)

# 节省内存
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

# 输出特征
df_train.to_csv(output_train, index=False)
df_test.to_csv(output_test, index=False)
print('树模型特征工程已完成')
