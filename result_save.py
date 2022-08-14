#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/12 16:03
# @Author  : Chenchen Wei
# @Description:
import csv
import os
import time
import numpy as np
import pandas as pd
from evalutation import *




class save_results(object):
    def __init__(self, save_path, model_name, data_name, params, rec, true, mask,
                 csv_path='result.csv'):
        self.save_path = save_path
        self.model_name = model_name
        self.data_name = data_name
        self.params = params
        self.rec = rec
        self.true = true
        self.mask = mask
        self.csv_path = csv_path

    def _call(self, ):
        self.mask = np.transpose(self.mask, (0, 2, 1)).reshape(-1, self.mask.shape[-2])
        loss_col, loss_column = np.where(self.mask == 0)[0], np.where(self.mask == 0)[1]

        self.rec_pre, self.rec_true = [], []
        for i, j in zip(loss_col, loss_column):  # The impute and true values of all segments
            self.rec_pre.append(self.rec[i][j])
            self.rec_true.append(self.true[i][j])

        # Imputing and true values of each road segment
        lo_vals, true_vals = [], []
        for i in range(self.mask.shape[-1]):
            temp = self.mask[:, i]
            zeros_index = np.where(temp == 0)[0]
            lo_vals.append(self.rec[zeros_index, i])
            true_vals.append(self.true[zeros_index, i])
        self.each_rec = np.asarray(lo_vals)
        self.each_true = np.asarray(true_vals)

    def main(self):
        self._call()
        mae, rmse, mape = evalute(self.rec_pre, self.rec_true)
        now_times = time.strftime("%m-%d %H:%M:%S", time.localtime())
        self.save_csv([self.rec, self.true], ['rec.csv', 'true.csv'])
        rows = [self.model_name, self.data_name, mae, rmse, mape] + list(self.params.values()) + [now_times]
        print(rows)
        self.write_csv(rows)
        return self.each_rec, self.each_true

    def save_csv(self, vals, names):
        for val, name in zip(vals, names):
            pd.DataFrame(val).to_csv(os.path.join(self.save_path, name),
                                     header=None, index=None)

    def write_csv(self, rows):
        with open(self.csv_path, 'a+', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(rows)
