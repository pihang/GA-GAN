#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 10:13
# @Author  : Chenchen Wei
# @Description:

import matplotlib

matplotlib.use('Agg')  # Server mapping, the order cannot be wrong
import matplotlib.pyplot as plt


def plt_figures(vals, labels, save_path, save_names, title='', figsize=(12, 8)):
    """
    Drawing multiple curves in one picture, and list different curves
    :param vals: list
    :param labels: list
    :param title:
    :param save_names:
    :return:
    """
    plt.figure(figsize=figsize)
    for val, label in zip(vals, labels):
        plt.plot(val, label=label)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path + '/' + save_names)
    plt.close()
