import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost.sklearn import XGBRegressor
from matplotlib import pyplot
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_data')
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prediction_result')

data_path = os.path.join(data_dir, 'train_tree.csv')
test_data_path = os.path.join(data_dir, 'test_tree.csv')
model_path = os.path.join(model_dir, 'xgb.pkl')
output_path = os.path.join(output_dir, 'xgb.csv')


def my_score(y_predict, y_true):
    '''
    自定义评估指标
    :return:
    '''
    score = - mean_absolute_error(np.exp(y_true), np.exp(y_predict))
    return 'exp_mae', score


def train():
    df = pd.read_csv(data_path)
    print(df.head())
    y = df.pop('price').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2020)

    param_dist = {
        "n_estimators": 50000,
        "objective": "reg:squarederror",
        # "eval_metric": "mae",
        "learning_rate": 0.25,  # [0.03~0.3]
        "gamma": 0,  # 损失值大于gamma才能继续分割[0~0.5]
        "min_child_weight": 1,  # 最小叶子结点权重 [20~200]
        "max_depth": 6,  # [4~10]
        "colsample_bytree": 0.6,  # [0.3~1]
        "subsample": 0.6,  # [0.3~1]
        "early_stopping_rounds": 50,
        "tree_method": "gpu_hist",
        "gpu_id": 0
    }

    clf = XGBRegressor(**param_dist)

    print('开始训练。。。')
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=my_score)
    print(clf.best_score)
    print(clf.best_iteration)

    eval_result = clf.evals_result()
    print(eval_result)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print('模型保存成功。。。')


def load_model():
    '''
    加载模型
    :return:
    '''
    with open(model_path, 'rb') as f:
        xgb = pickle.load(f)
        return xgb


def learning_curve_plot():
    model = load_model()
    eval_results = model.evals_result()
    # retrieve performance metrics
    epochs = len(eval_results['validation_0']['mae'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(x_axis[50:], eval_results['validation_0']['mae'][50:], label='Train')
    ax.plot(x_axis[50:], eval_results['validation_1']['mae'][50:], label='Test')
    ax.legend()
    pyplot.ylabel('mae')
    pyplot.title('XGBoost mae')
    pyplot.show()


if __name__ == '__main__':
    # learning_curve_plot()
    model = load_model()
    print(model.feature_importances_)
    df_test = pd.read_csv(test_data_path)
    price = np.exp(model.predict(df_test.values, ntree_limit=model.best_ntree_limit))
    # saleID = np.array(range(200000, 250000))
    # df = pd.DataFrame({'SaleID': saleID, 'price': price})
    # print(df.head())
    # df.to_csv(output_path, index=False)
