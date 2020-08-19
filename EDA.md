## 数据字典
|  Field   | Description  |
|  ----  | ----  |
|SaleID	|交易ID，唯一编码|
|name	|汽车交易名称，已脱敏|
|regDate	|汽车注册日期，例如20160101，2016年01月01日|
|model	|车型编码，已脱敏|
|brand	|汽车品牌，已脱敏|
|bodyType	|车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7|
|fuelType	|燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6|
|gearbox	|变速箱：手动：0，自动：1|
|power	|发动机功率：范围 [ 0, 600 ]|
|kilometer	|汽车已行驶公里，单位万km|
|notRepairedDamage	|汽车有尚未修复的损坏：是：0，否：1|
|regionCode	|地区编码，已脱敏|
|seller	|销售方：个体：0，非个体：1|
|offerType	|报价类型：提供：0，请求：1|
|creatDate	|汽车上线时间，即开始售卖时间|
|price	|二手车交易价格（预测目标）|
|v系列特征	|匿名特征，包含v0-14在内15个匿名特征|

## 数据量
训练集数据量15万，测试集数据量5万

## 缺失情况
tips 可以使用missingno包对于缺失值进行可视化
```
import missingno as msno
# 无效矩阵的数据密集显示
msno.matrix(df, labels=True)
# 条形图
msno.bar(df)
# 相关性热力图
msno.heatmap(df)
# 树状图
msno.dendrogram(df)
```
========train_set========
- 【model】 车型编码 miss_count:1,miss_rate:1e-05
- 【bodyType】 车身类型 miss_count:4506,miss_rate:0.03004
- 【fuelType】 燃油类型 miss_count:8680,miss_rate:0.05787
- 【gearbox】 变速箱 miss_count:5981,miss_rate:0.03987

========test_set========
- 【bodyType】 车身类型 miss_count:1413,miss_rate:0.02826
- 【fuelType】 燃油类型 miss_count:2893,miss_rate:0.05786
- 【gearbox】 变速箱 miss_count:1910,miss_rate:0.0382

## 异常数据
tips:注意pandas中object类型的数据，需要重点看一下。
对于本数据集，【notRepairedDamage】（汽车有尚未修复的损坏）字段将 - 转换为null
```
df_train['notRepairedDamage'].value_counts()
0.0    111361
-       24324
1.0     14315
Name: notRepairedDamage, dtype: int64

df_test['notRepairedDamage'].value_counts()
0.0    37249
-       8031
1.0     4720

df_train['notRepairedDamage'].replace('-', np.nan, inplace=True)
df_test['notRepairedDamage'].replace('-', np.nan, inplace=True)
```
处理完之后再次观察缺失值情况
========train_set========
- 【model】 miss_count:1,miss_rate:1e-05
- 【bodyType】 miss_count:4506,miss_rate:0.03004
- 【fuelType】 miss_count:8680,miss_rate:0.05787
- 【gearbox】 miss_count:5981,miss_rate:0.03987
- 【notRepairedDamage】 miss_count:24324,miss_rate:0.16216

========test_set========
- 【bodyType】 miss_count:1413,miss_rate:0.02826
- 【fuelType】 miss_count:2893,miss_rate:0.05786
- 【gearbox】 miss_count:1910,miss_rate:0.0382
- 【notRepairedDamage】 miss_count:8031,miss_rate:0.16062

## 删除异常特征
由于训练集和测试集的【seller】【offerType】字段只有1个枚举值，所以可以直接删除
```
# value_counts枚举值只有1
del df_train["seller"]
del df_train["offerType"]
del df_test["seller"]
del df_test["offerType"]
```

## 观测预测值分布
预测值的不是标准正态分布，偏度: 3.346487，峰度: 18.995183。是一个左偏高尖型的分布，通常预测这类问题需要将预测值进行log变换


## 各个特征观测
|  Field   | Description  |Type|
|  ----  | ----  | ----  |
|SaleID	|交易ID，唯一编码|ID|
|name	|汽车交易名称，已脱敏|类别（类别数较大）|
|regDate	|汽车注册日期，例如20160101，2016年01月01日|日期|
|model	|车型编码，已脱敏|类别|
|brand	|汽车品牌，已脱敏|类别|
|bodyType	|车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7|类别|
|fuelType	|燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6|类别|
|gearbox	|变速箱：手动：0，自动：1|类别|
|power	|发动机功率：范围 [ 0, 600 ]|数值|
|kilometer	|汽车已行驶公里，单位万km|数值|
|notRepairedDamage	|汽车有尚未修复的损坏：是：0，否：1|类别|
|regionCode	|地区编码，已脱敏|类别|
|seller	|销售方：个体：0，非个体：1|【删除】|
|offerType	|报价类型：提供：0，请求：1|【删除】|
|creatDate	|汽车上线时间，即开始售卖时间|日期|
|v系列特征	|匿名特征，包含v0-14在内15个匿名特征|数值|
|price	|二手车交易价格（预测目标）|数值|


### 离散型特征

#### 维度的基

========================================

训练集【name】特征有99662个不同的值

预测集【name】特征有37453个不同的值

========================================

训练集【model】特征有248个不同的值

预测集【model】特征有247个不同的值

========================================

训练集【brand】特征有40个不同的值

预测集【brand】特征有40个不同的值

========================================

训练集【bodyType】特征有8个不同的值

预测集【bodyType】特征有8个不同的值

========================================

训练集【fuelType】特征有7个不同的值

预测集【fuelType】特征有7个不同的值

========================================

训练集【gearbox】特征有2个不同的值

预测集【gearbox】特征有2个不同的值

========================================

训练集【notRepairedDamage】特征有2个不同的值

预测集【notRepairedDamage】特征有2个不同的值

========================================

训练集【regionCode】特征有7905个不同的值

预测集【regionCode】特征有6971个不同的值

#### 类别对应的数量
- brand=0的销量最好
- body_type=0销量最好
- fuelType=0销量最好
- gearbox=0销量最好
- notRepairedDamage=0销量最好

#### 类别特征与价格的影响关系
- brand=24与brand=37的价格中位数最高(汽车品牌)
- bodyType=5,6,4的价格中位数较高，缺失值的价格相对较低(车身类型)
- fuelType=1,4,6的价格中位数较高(燃油类型) 更倾向电动，混动
- gearbox=1较高(自动挡车更保值)
- notRepairedDamage=0较高（必然是未修理过的车更保值）


#### 连续型特征单变量分布
tips 
- 行驶里程已经被离散化了 半数以上二手车行驶里程是15万公里
- v_5，v_6,v_8,v_9,列存在大量0的情况

#### 连续型特征与价格相关性分析
```python
# v12，v8,v0 正相关；kilometer，v_3负相关
[('v_12', 0.693),
 ('v_8', 0.686),
 ('v_0', 0.628),
 ('power', 0.22),
 ('v_5', 0.164),
 ('v_2', 0.085),
 ('v_6', 0.069),
 ('v_1', 0.061),
 ('v_14', 0.036),
 ('v_13', -0.014),
 ('v_7', -0.053),
 ('v_4', -0.147),
 ('v_9', -0.206),
 ('v_10', -0.246),
 ('v_11', -0.275),
 ('kilometer', -0.441),
 ('v_3', -0.731)]
```

#### 特征与特征之间的相关性
- 超强正相关(v1,v6,0.999415),(v2,v7,0.973689),(v4,v9,0.962928),(v4,v13,0.934580)
- 超强负相关(v1,v10,-0.921904),(v2,v5,-0.921857),(v5,v7,-0.939385,v6,v10)
- v1与v6超强相关，相关系数0.999415

### 日期类特征
- 车龄大部分在2000天到8000天
- 总体来看，随着车龄的增长钱在降低，但是也有价格坚挺的车
- 销售数据主要集中在2016年，月份主要集中在3，4月份
