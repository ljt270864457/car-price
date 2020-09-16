import os

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost.sklearn import XGBRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_data')
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prediction_result')

data_path = os.path.join(data_dir, 'train_tree.csv')
test_data_path = os.path.join(data_dir, 'test_tree.csv')

valid_output_path = os.path.join(output_dir, 'xgb_valid.csv')
test_output_path = os.path.join(output_dir, 'xgb_test.csv')

df_train = pd.read_csv(data_path)
df_test = pd.read_csv(test_data_path)
print(df_train.shape)
print(df_test.shape)
y = df_train.pop('price').values
X = df_train.values
X_test = df_test.values


def log_transfer(func):  # 定义一个将数据转为log的闭包函数
    def wrapper(y, yhat):
        result = func(np.exp(y), np.nan_to_num(np.exp(yhat)))
        return result

    return wrapper


# 调参
def xgb_cv(n_estimators, learning_rate, gamma, min_child_weight, max_depth, colsample_bytree, subsample):
    param = {
        'objective': 'reg:squarederror',
        'random_state': 2020,
        #         "early_stopping_rounds": 50,
        "tree_method": "gpu_hist",
        "gpu_id": 1
    }
    param['n_estimators'] = int(n_estimators)
    param['learning_rate'] = float(learning_rate)
    param['gamma'] = float(gamma)
    param['min_child_weight'] = float(min_child_weight)
    param['max_depth'] = int(max_depth)
    param['colsample_bytree'] = float(colsample_bytree)
    val = cross_val_score(XGBRegressor(**param),
                          X, y, scoring=make_scorer(log_transfer(mean_absolute_error), greater_is_better=False),
                          cv=5).mean()
    return val


xgb_bo = BayesianOptimization(
    xgb_cv,
    {'n_estimators': (150, 1000),
     'learning_rate': (0.03, 0.3),
     'gamma': (0, 0.5),
     'min_child_weight': (10, 200),
     'max_depth': (4, 10),
     'colsample_bytree': (0.1, 1),
     'subsample': (0.5, 1),
     }
)


# 获取调参结果
# xgb_bo.maximize()
# print(xgb_bo.res)
# print(xgb_bo.max['params'])


def my_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'exp_score', mean_absolute_error(np.exp(labels), np.nan_to_num(np.exp(preds)))


param_dist = {
    "n_estimators": 100000,
    "objective": "reg:squarederror",
    "learning_rate": 0.03,  # [0.03~0.3]
    "gamma": 0.04782946032469271,  # 损失值大于gamma才能继续分割[0~0.5]
    "min_child_weight": 116,  # 最小叶子结点权重 [20~200]
    "max_depth": 8,  # [4~10]
    "colsample_bytree": 0.7237044081916191,  # [0.3~1]
    "subsample": 0.985,  # [0.3~1]
    "tree_method": "gpu_hist",
    "gpu_id": 1
}

folds = KFold(n_splits=5, shuffle=True, random_state=2020)
oof_xgb = np.zeros(len(X))
predictions_xgb = np.zeros(len(X_test))
predictions_train_xgb = np.zeros(len(X))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_ + 1))
    clf = XGBRegressor(**param_dist)
    clf.fit(X[trn_idx], y[trn_idx], eval_set=[(X[val_idx], y[val_idx])], eval_metric=my_score, early_stopping_rounds=50)
    oof_xgb[val_idx] = clf.predict(X[val_idx], ntree_limit=clf.best_iteration)
    predictions_xgb += clf.predict(X_test, ntree_limit=clf.best_iteration) / folds.n_splits
    predictions_train_xgb += clf.predict(X, ntree_limit=clf.best_iteration) / folds.n_splits

print("xgb score: {:<8.8f}".format(mean_absolute_error(np.exp(oof_xgb), np.expm1(y))))

# 验证集输出
SaleID = range(150000)
y_true = y
y_predict = oof_xgb
df = pd.DataFrame({'SaleID': SaleID, 'y_true': y_true, 'y_predict': y_predict})
df.to_csv(valid_output_path, index=False)

# 测试集输出
saleID = np.array(range(200000, 250000))
price = np.exp(predictions_xgb)
df = pd.DataFrame({'SaleID': saleID, 'price': price})
df.to_csv(test_output_path, index=False)

# 绘制特征重要度
# clf.importance_type='gain'
# im = pd.DataFrame({'feature':df_train.columns,'importance':clf.feature_importances_})
# im=im.sort_values('importance',ascending=False)
# import seaborn as sns
# f, ax = plt.subplots(figsize=(16, 16))
# sns.barplot(y='feature',x='importance',data=im.head(50))

from lightgbm import cv
