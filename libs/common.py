#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/19 7:40 下午
# @Author  : liujiatian
# @File    : common.py

import numpy as np
import pandas as pd


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem / 1024 / 1024))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem / 1024 / 1024))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def my_agg(data, dim, measure='price'):
    '''
    按照指定维度聚合并重命名,暂时维度只有1
    '''
    index = ['min', 'max', 'mean', 'median', 'sum', 'std', 'kurt', 'skew']
    new_columns = [dim] + list(map(lambda x: f'{dim}_{measure}_{x}', index))
    new_df = data.groupby(dim).agg({
        measure: ['min', 'max', 'mean', 'median', 'sum', 'std', pd.DataFrame.kurt, pd.DataFrame.skew]
    }).reset_index()
    new_df.columns = new_columns
    return new_df


def clean_error_month(x):
    '''
    清洗日期中月份出现错误的数据 将00->01
    '''
    x = str(x)
    if len(x) != 8:
        return pd.NaT
    if x[4:6] == '00':
        x = x[:4] + '01' + x[6:]
    return x
