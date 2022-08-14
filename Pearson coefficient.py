#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 10:41
# @Author  : Chenchen Wei
# @Description:

import os

import numpy as np
import pandas as pd

data_params = {'seattle_data': {'data_path': os.path.abspath(os.path.join(os.getcwd(), "./data/seattle")),
                                'file_names': 'Speed2.csv',
                                'adj_name': 'AA.csv',
                                'corr_path': 'seattle_corr_dis.csv'},
               'pems-bay': {'data_path': os.path.abspath(os.path.join(os.getcwd(), "./data/pems/pems-bay")),
                            'file_names': 'pems-bay_speed.csv',
                            'adj_name': '',
                            'corr_path': 'pems-bay-corr.csv'}}

flags = ['pems-bay']
data_flag = flags[0]
data_path, file_name, adj_name, corr_path = data_params[data_flag].values()
data = pd.read_csv(os.path.join(data_path, file_name)).values[:288 * 7, :]
cor = np.corrcoef(data, rowvar=0)
pd.DataFrame(cor).to_csv(os.path.join(data_path, corr_path), index=None, header=None)
