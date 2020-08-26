#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 6:56 下午
# @Author  : liujiatian
# @File    : by_opt.py

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

# 产生随机分类数据集，10个特征， 2个类别
x, y = make_classification(n_samples=100, n_features=5, n_classes=2)
rf = RandomForestClassifier()
print(np.mean(cross_val_score(rf, x, y, cv=20, scoring='roc_auc')))


def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),  # float
                               max_depth=int(max_depth),
                               random_state=2
                               ),
        x, y, scoring='roc_auc', cv=5
    ).mean()
    return val


rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (10, 250),
     'min_samples_split': (2, 25),
     'max_features': (0.1, 0.999),
     'max_depth': (5, 15)}
)
rf_bo.maximize()
print(rf_bo.res)
print(rf_bo.max)
'''
[{'target': 0.944, 'params': {'max_depth': 14.060846063050885, 'max_features': 0.4542043954517685, 'min_samples_split': 24.645266967639877, 'n_estimators': 48.531147732900166}}, {'target': 0.944, 'params': {'max_depth': 11.734678303011993, 'max_features': 0.4422087065280813, 'min_samples_split': 10.435858855098768, 'n_estimators': 129.77654934599548}}, {'target': 0.95, 'params': {'max_depth': 14.574713222919275, 'max_features': 0.16941149208471296, 'min_samples_split': 3.8800117925445012, 'n_estimators': 158.60251134415705}}, {'target': 0.942, 'params': {'max_depth': 8.499685061591133, 'max_features': 0.11000365797959963, 'min_samples_split': 18.384005401558635, 'n_estimators': 237.21842955067484}}, {'target': 0.9400000000000001, 'params': {'max_depth': 9.811977939335335, 'max_features': 0.46374081955329904, 'min_samples_split': 13.625245823188417, 'n_estimators': 223.79030834556562}}, {'target': 0.9359999999999999, 'params': {'max_depth': 7.401703742914555, 'max_features': 0.12179474883130588, 'min_samples_split': 2.2343147479227854, 'n_estimators': 10.511832843254885}}, {'target': 0.9460000000000001, 'params': {'max_depth': 14.97491916782571, 'max_features': 0.13979792086643547, 'min_samples_split': 3.2015838920118678, 'n_estimators': 244.6474881698322}}, {'target': 0.9400000000000001, 'params': {'max_depth': 14.957423213568502, 'max_features': 0.9545818150035623, 'min_samples_split': 24.57894231441473, 'n_estimators': 193.82333795654557}}, {'target': 0.9349999999999999, 'params': {'max_depth': 14.966956267389001, 'max_features': 0.40456763662057926, 'min_samples_split': 2.8015765984363425, 'n_estimators': 73.21296506106995}}, {'target': 0.942, 'params': {'max_depth': 14.733154692126677, 'max_features': 0.3043485537091408, 'min_samples_split': 24.42011484854261, 'n_estimators': 11.573880221476234}}, {'target': 0.933, 'params': {'max_depth': 5.180256138405675, 'max_features': 0.8731472128098345, 'min_samples_split': 24.247270796606504, 'n_estimators': 13.08601864997926}}, {'target': 0.9480000000000001, 'params': {'max_depth': 5.310867175311841, 'max_features': 0.44179165947599774, 'min_samples_split': 2.2068260966785234, 'n_estimators': 248.97890684219382}}, {'target': 0.95, 'params': {'max_depth': 5.198380951677092, 'max_features': 0.1152449922709869, 'min_samples_split': 5.547149185902824, 'n_estimators': 163.78235937229672}}, {'target': 0.932, 'params': {'max_depth': 14.613989614725146, 'max_features': 0.1366830604208306, 'min_samples_split': 24.768205424328553, 'n_estimators': 147.5486196303545}}, {'target': 0.9390000000000001, 'params': {'max_depth': 12.977570154451953, 'max_features': 0.9598191655340023, 'min_samples_split': 23.31526085761592, 'n_estimators': 249.11407111413735}}, {'target': 0.9440000000000002, 'params': {'max_depth': 7.097356297293786, 'max_features': 0.1200048526014765, 'min_samples_split': 2.0956886636537817, 'n_estimators': 183.7226792953449}}, {'target': 0.9410000000000001, 'params': {'max_depth': 5.110366318527633, 'max_features': 0.49203178088034305, 'min_samples_split': 2.812635439902597, 'n_estimators': 139.35406895165252}}, {'target': 0.943, 'params': {'max_depth': 14.97924756843138, 'max_features': 0.943403906859195, 'min_samples_split': 4.655504568754322, 'n_estimators': 32.97772129982721}}, {'target': 0.942, 'params': {'max_depth': 5.329686284790447, 'max_features': 0.9963099962105252, 'min_samples_split': 14.984256553445858, 'n_estimators': 174.97250320498677}}, {'target': 0.9470000000000001, 'params': {'max_depth': 14.859624314148453, 'max_features': 0.7686230298428456, 'min_samples_split': 23.47640641136372, 'n_estimators': 89.50596535601827}}, {'target': 0.9460000000000001, 'params': {'max_depth': 13.178253573706582, 'max_features': 0.36681836939236223, 'min_samples_split': 5.513338437252996, 'n_estimators': 162.25558916350022}}, {'target': 0.9359999999999999, 'params': {'max_depth': 5.060571422114091, 'max_features': 0.1783758600458792, 'min_samples_split': 23.522630689877648, 'n_estimators': 90.04741667495243}}, {'target': 0.9480000000000001, 'params': {'max_depth': 14.8388123044917, 'max_features': 0.26844900707113994, 'min_samples_split': 4.2268360257955155, 'n_estimators': 117.78251159910724}}, {'target': 0.9400000000000001, 'params': {'max_depth': 5.163534781703158, 'max_features': 0.13407797892586623, 'min_samples_split': 24.591542519191805, 'n_estimators': 199.46816568827364}}, {'target': 0.9390000000000001, 'params': {'max_depth': 8.247423163415952, 'max_features': 0.5977310822013809, 'min_samples_split': 3.7502607820861913, 'n_estimators': 161.60189619070832}}, {'target': 0.946, 'params': {'max_depth': 12.193755022272931, 'max_features': 0.1, 'min_samples_split': 7.280729552486767, 'n_estimators': 158.3247722357097}}, {'target': 0.9400000000000001, 'params': {'max_depth': 14.792367190276748, 'max_features': 0.13560259230850583, 'min_samples_split': 20.05158351172034, 'n_estimators': 104.53894451138957}}, {'target': 0.9400000000000001, 'params': {'max_depth': 14.677476612177706, 'max_features': 0.23318777089027168, 'min_samples_split': 23.863219544405695, 'n_estimators': 70.34693973523854}}, {'target': 0.9359999999999999, 'params': {'max_depth': 14.770369768338158, 'max_features': 0.3042182876596729, 'min_samples_split': 2.2004697073085686, 'n_estimators': 10.368942925663216}}, {'target': 0.932, 'params': {'max_depth': 14.992135424762695, 'max_features': 0.10068924492512678, 'min_samples_split': 21.436388603388405, 'n_estimators': 29.792606753243092}}]
'''
print(123)
