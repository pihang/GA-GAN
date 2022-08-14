#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 19:08
# @Author  : Chenchen Wei
# @Description:
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class get_data(object):
    """
    Return missing data in different missing categories and missing rates
    Return: Samples * node * lags
    """

    def __init__(self, path,
                 file_name,
                 loss_category='mcar',
                 loss_rate=0.1,
                 lags=12,
                 split_rate=0.8,
                 ):
        self.path = path
        self.file_name = file_name
        self.loss_category = loss_category
        self.loss_rate = loss_rate
        self.lags = lags
        self.split_rate = split_rate
        self.data_names = '{}_{}_lrt{}_sl{}.npz'.format(self.file_name[:-4],
                                                        self.loss_category,
                                                        self.loss_rate,
                                                        self.lags)
        self.scalerfile = os.path.join(self.path, self.file_name[:-4] + 'rec_scaler.sav')

    def data_process(self):
        data = pd.read_csv(os.path.join(self.path, self.file_name)).values
        self.road_num = data.shape[1]
        data_num = data.shape[0] // self.lags
        split_num = int(0.8 * data_num)
        train, test = data[:split_num * self.lags], data[split_num * self.lags:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Standardization operation, standardize the data of each detector by column
        train = scaler.fit_transform(train)
        test = scaler.transform(test)  # Standardize the test set with the training set scale

        train = np.array(np.split(train, train.shape[0] // self.lags, axis=0))
        test = np.array(np.split(test, test.shape[0] // self.lags, axis=0))
        train, test = np.transpose(train, (0, 2, 1)), np.transpose(test, (0, 2, 1))

        train_mask, test_mask = self.get_mask(train.shape), self.get_mask(test.shape)
        train_x, train_y = train * train_mask, train
        test_x, test_y = test * test_mask, test

        np.savez(os.path.join(self.path, self.data_names),
                 train_x, train_y, test_x,
                 test_y, train_mask, test_mask)  # data file pre_save
        pickle.dump(scaler, open(self.scalerfile, 'wb'))  # save scaler

        return train_x, train_y, test_x, test_y, scaler, train_mask, test_mask

    def main(self):
        if os.path.exists(os.path.join(self.path, self.data_names)):  # If it exists, load directly
            a = time.time()
            print('#' + self.data_names + ' are Exists, Begin Loading')
            all_data = np.load(os.path.join(self.path, self.data_names))
            train_x, train_y, test_x = all_data['arr_0'], all_data['arr_1'], all_data['arr_2']
            test_y, train_mask, test_mask = all_data['arr_3'], all_data['arr_4'], all_data['arr_5']
            scaler = pickle.load(open(self.scalerfile, 'rb'))
            use_time = time.time() - a
            print('#Loading Success! Use time= {:.2f}s'.format(use_time))
        else:  # If it does not exist, call the function to read and save data
            a = time.time()
            print('#' + self.data_names + ' are Not exists, Begin Loading and Saving')
            train_x, train_y, test_x, test_y, scaler, train_mask, test_mask = self.data_process()
            use_time = time.time() - a
            print('#Save and Loading Success Use time= {:.2f}s'.format(use_time))
        return train_x, train_y, test_x, test_y, train_mask, test_mask, scaler

    def get_mask(self, shape):
        if self.loss_category == 'mcar':
            return self.get_mcar_mask(shape)

        elif self.loss_category == 'tmcar':
            return self.get_time_mask(shape)

        elif self.loss_category == 'smcar':
            return self.get_space_mask(shape)

        else:
            raise ValueError('No loss_category')

    def get_time_mask(self, shape):
        """
        tmcar
        Missing in time, no data at certain moments
        :param shape:
        :return:
        """
        loss_nums = int(shape[-1] * self.loss_rate)
        mask = np.ones(shape=shape)
        for num in range(shape[0]):
            indexs = np.arange(shape[-1]).astype(np.int)
            np.random.shuffle(indexs)
            index = indexs[:loss_nums]
            mask[num, :, index] = 0
        return mask

    def get_space_mask(self, shape):
        """
        smcar
        Spatially missing, some detectors have no data
        :param shape:
        :return:
        """
        loss_nums = int(shape[1] * self.loss_rate)
        mask = np.ones(shape=shape)
        for num in range(shape[0]):
            indexs = np.arange(shape[1]).astype(np.int)
            np.random.shuffle(indexs)
            index = indexs[:loss_nums]
            mask[num, index, :] = 0
        return mask

    def get_mcar_mask(self, shape):
        """
        mcar
        All missing at random
        :param shape:
        :return:
        """
        array = []
        flag = int(shape[1] * shape[2] * (self.loss_rate))
        for n in range(0, shape[0]):
            array_first = []
            array_mid = []
            array_fianl = np.ones([shape[1], shape[2]])
            for i in range(0, shape[1]):
                for j in range(0, shape[2]):
                    array_mid.append(int(array_fianl[i][j]))
            for k in range(0, flag):
                array_mid[k] = 0
            np.random.shuffle(array_mid)  # 得到随机排序后的array_mid
            for z in range(0, shape[2] * shape[1], shape[2]):
                array_first.append(array_mid[z:z + shape[2]])
            array.append(array_first)

        return np.asarray(array)
