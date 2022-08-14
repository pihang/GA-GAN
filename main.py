#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 10:29
# @Author  : Chenchen Wei
# @Description:
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product
from models import *
from models_2 import *
from result_save import *
from visualization import *
from utils import *
from load_data import *
# from gain_GAGAN import *


data_params = {'seattle_data': {'data_path': os.path.abspath(os.path.join(os.getcwd(), "./data/seattle")),
                                'file_names': 'Speed2.csv',
                                'ori_adj_path': 'AA.csv',
                                'corr_path': 'seattle_corr_dis.csv'},
               'pems-bay': {'data_path': os.path.abspath(os.path.join(os.getcwd(), "./data/pems-bay")),
                            'file_names': 'pems-bay_speed.csv',
                            'ori_adj_path': '',
                            'corr_path': 'pems-bay_corr_dis.csv'}, }
model_params = {'corr_rates': [0.01],
                # 'loss_categorys': ['mcar', 'tmcar'],
                'loss_categorys': ['smcar'],
                'loss_rates': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, ],
                # 'loss_rates': [0.1],
                'lags': 12,
                'epoch': 200,
                'batch_size': 32,
                'sage_units': [64, 128],
                'ge_units': [64, 128],
                'dis_units': [128, 64], }

flags = ['seattle_data', 'pems-bay']
data_flag = flags[0]
data_path, file_name, ori_adj_path, corr_path = data_params[data_flag].values()
corr_rates = model_params['corr_rates']
loss_categorys = model_params['loss_categorys']
loss_rates = model_params['loss_rates']
lags = model_params['lags']
epoch = model_params['epoch']
batch_size = model_params['batch_size']
sage_units = model_params['sage_units']
ge_units = model_params['ge_units']
dis_units = model_params['dis_units']
loop_val = [corr_rates, loss_categorys, loss_rates]
model_name = 'GA_GAN_concat'
gpu_set(0, 0.5)
for loops in product(*loop_val):
    tf.reset_default_graph()
    corr_rate, loss_category, loss_rate = loops
    save_path = 'Results/{}/{}/cor={} cate={} rate={} ep={}'.format(model_name,
                                                                    file_name[:-4],
                                                                    corr_rate,
                                                                    loss_category,
                                                                    loss_rate,
                                                                    epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    adj = Construction_matrix(os.path.join(data_path, corr_path), corr_rate)
    ori_adj = pd.read_csv(os.path.join(data_path,ori_adj_path), header=None).values
    data = get_data(path=data_path,
                    file_name=file_name,
                    loss_category=loss_category,
                    loss_rate=loss_rate,
                    lags=lags).main()
    train_x, train_y, test_x, test_y, train_mask, test_mask, scaler = data


    with tf.Session() as sess:
        model = GA_GAN_concat(sess=sess,
                       all_data=data,
                       adj=adj,
                       sgae_hidden_list=sage_units,
                       ge_hidden_lists=ge_units,
                       dis_hidden_lists=dis_units,
                       epoch=epoch,
                       batch_size=batch_size,
                       save_model=True,
                       save_model_path=save_path,
                       ori_adj=ori_adj,
                       )   # ori_adj=ori_adj
        rec, true = model.train()

    params = {'lo_cat': loss_category,
              'lo_rt': loss_rate,
              'sel_ra': corr_rate,
              'ep': epoch,
              'bs': batch_size,
              'sg_h': sage_units,
              'g_hi': ge_units,
              'd_hi': dis_units}

    each_rec, each_true = save_results(save_path=save_path,
                                       model_name=model_name,
                                       data_name=file_name[:-4],
                                       params=params,
                                       rec=rec,
                                       true=true,
                                       mask=test_mask).main()
    #
    # plt_figures(vals=[each_rec[0], each_true[0]],
    #             labels=['rec', 'true'],
    #             save_path=save_path,
    #             title='only_loss',
    #             save_names='road_0.png')
