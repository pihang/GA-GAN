#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 19:03
# @Author  : Chenchen Wei
# @Description:

import os
import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def input_sequnece(data, n_in=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i - 1))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def data_process_pre(data_path, adj_index, obj_index, lags, test_rate=0.2):
    # 转换成滞后格式
    df = pd.read_pickle(data_path).values
    adj_vals, obj_vals = df[:, adj_index], df[:, obj_index]

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    adj_vals = scaler1.fit_transform(adj_vals)

    scaler2 = MinMaxScaler(feature_range=(0, 1))
    obj_vals = scaler2.fit_transform(obj_vals)

    data_x = input_sequnece(adj_vals, n_in=lags).values
    data_y = input_sequnece(obj_vals, n_in=lags).values
    length = int(data_x.shape[0] * (1 - test_rate))
    train_x, test_x = data_x[:length, :], data_x[length:, :]
    train_y, test_y = data_y[:length, :], data_y[length:, :]
    return scaler2, train_x, train_y, test_x, test_y


def get_adj_obj_index(road_path, obj_index):
    road = np.load(road_path, allow_pickle=True)
    if len(obj_index) == 1:
        return road[obj_index[0]][1]
    else:
        all_index = []
        for i in range(len(obj_index)):
            all_index.append(road[obj_index[i]][1])
        adj_index = list(set(all_index[0]).intersection(*all_index[1:]))
        if len(adj_index) == 0:
            print('The obj_index %s Don\'t have the intersection' % (obj_index))
            return []
        else:
            return adj_index


def Construction_matrix(path, scale):
    cor = abs(pd.read_csv(path, header=None).values)
    num = int(scale * cor.shape[1])
    for i in range(cor.shape[0]):
        sort = np.argsort(-cor[i])
        cor[i][sort[:num]] = 1
        cor[i][sort[num:]] = 0
    return np.asarray(cor)


def gpu_set(gpu_id, memory):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = memory
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)


def weights_get(names, input_dim, output_dim, use_regular=False):
    """
    If regularization is used, the model loss function should be changed
    """
    if use_regular:
        w = tf.get_variable(names,
                            [input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4))
    else:
        w = tf.get_variable(names,
                            [input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
    return w


def bias_get(names, dim):
    b = tf.get_variable(names,
                        [dim],
                        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    return b


def count_time(func):
    """
    Statistical function runtime decorator
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('Function：<' + str(func.__name__) + '>TimeCost：{:.2f} Minute'.format((end_time - start_time) / 60))
        return ret

    return wrapper


def xavier_init(size):  # 初始化参数时使用的xavier_init函数
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)  # 初始化标准差
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):  # 生成维度为[m, n]的随机噪声作为生成器G的输入
    return np.random.uniform(-1, 1., size=[m, n])
