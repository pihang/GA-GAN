#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 19:00
# @Author  : Chenchen Wei
# @Description:

from .utils import *


class GraphSAGE_Mean:
    def __init__(self, name, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu):
        self.dropout = dropout
        self.act = act
        for i in range(len(adj)): adj[i][i] = 0  # delete self connection
        self.adj = adj / np.sum(adj, axis=1).reshape(-1, 1)  # mean aggregate
        self.adj = tf.cast(self.adj, tf.float32)

        self.w1 = weights_get(name + 'self', input_dim, output_dim)
        self.w2 = weights_get(name + 'neg', input_dim, output_dim)

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, rate=self.dropout)
        _self = tf.matmul(inputs, self.w1)
        _neg = tf.matmul(tf.matmul(self.adj, inputs), self.w2)
        _concat = self.act(tf.concat([_self, _neg], axis=2, ))
        return _concat

    def __call__(self, inputs):
        return self._call(inputs)


class Linear:
    def __init__(self, name, input_dim, output_dim, use_bias=True, dropout=0., act=tf.nn.relu):
        self.dropout = dropout
        self.act = act
        self.w = weights_get(name + '_w', input_dim, output_dim)
        self.use_bias = use_bias
        if self.use_bias:
            self.b = bias_get(name + '_b', output_dim)

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, rate=self.dropout)
        if self.use_bias:
            x = tf.matmul(x, self.w) + self.b
        else:
            x = tf.matmul(x, self.w)
        x = self.act(x)
        return x

    def __call__(self, inputs):
        return self._call(inputs)
