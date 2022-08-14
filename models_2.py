#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 18:58
# @Author  : Chenchen Wei
# @Description:

from .layers import *
from .load_data import *
from tqdm import tqdm

class GA_GAN_concat(object):
    """
    原始路网，重建路网经SAGE聚合后融合特征
    """
    def __init__(self,
                 sess,
                 all_data,
                 adj,
                 ori_adj,
                 sgae_hidden_list,
                 ge_hidden_lists,
                 dis_hidden_lists,
                 reconstruction_coefficient=100,
                 epoch=1,
                 batch_size=2,
                 drop_rate=0,
                 learning_rate=1e-3,
                 save_model=False,
                 save_model_path=None):
        self.all_data = all_data
        self.adj = adj
        self.epoch = epoch
        self.batch_size = batch_size
        self.sess = sess
        self.drop_rate = drop_rate
        self.sgae_hidden_list = sgae_hidden_list
        self.ge_hidden_lists = ge_hidden_lists
        self.dis_hidden_lists = dis_hidden_lists
        self.learning_rate = learning_rate
        self.reconstruction_coefficient = reconstruction_coefficient
        self.save_model = save_model
        self.ori_adj = ori_adj
        self.save_model_path = save_model_path + '/model'

    def _call(self):
        self.train_x, self.train_y, self.test_x, self.test_y, = self.all_data[:4]
        self.train_mask, self.test_mask, self.scaler = self.all_data[4:]
        self.batch_nums = int(self.train_x.shape[0] / self.batch_size + 1)
        self.tr_bat_nums = self.batch_size - self.train_x.shape[0] % self.batch_size
        # Make up the number of 0 for insufficient batch
        self.in_dim, self.out_dim = self.train_x.shape[-1], self.train_y.shape[-1]
        self.node_num = self.train_x.shape[1]

    def GraphSAGE(self, inputs, reuse=False):
        with tf.variable_scope('sage', reuse=reuse):
            concat1 = GraphSAGE_Mean('sage1',
                                     input_dim=self.train_x.shape[-1],
                                     output_dim=self.sgae_hidden_list[0],
                                     adj=self.adj)(inputs)
            concat2 = GraphSAGE_Mean('sage2',
                                     input_dim=self.sgae_hidden_list[0] * 2,
                                     output_dim=self.sgae_hidden_list[0],
                                     adj=self.adj)(concat1)
            return concat2

    def GraphSAGE_ori(self, inputs, reuse=False):
        with tf.variable_scope('sage_ori', reuse=reuse):
            concat1 = GraphSAGE_Mean('sage1_ori',
                                     input_dim=self.train_x.shape[-1],
                                     output_dim=self.sgae_hidden_list[0],
                                     adj=self.ori_adj)(inputs)
            concat2 = GraphSAGE_Mean('sage2_ori',
                                     input_dim=self.sgae_hidden_list[0] * 2,
                                     output_dim=self.sgae_hidden_list[0],
                                     adj=self.ori_adj)(concat1)
            return concat2


    def generate(self, inputs, reuse=False):
        with tf.variable_scope('generate', reuse=reuse):
            h1 = Linear('ge_h1',
                        input_dim=inputs.shape[-1],
                        output_dim=self.ge_hidden_lists[0])(inputs)
            h2 = Linear('ge_h2',
                        input_dim=self.ge_hidden_lists[0],
                        output_dim=self.ge_hidden_lists[1])(h1)
            out = Linear('ge_out',
                         input_dim=self.ge_hidden_lists[1],
                         output_dim=self.train_x.shape[-1],
                         act=tf.nn.sigmoid)(h2)
            return out

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            h1 = Linear('dis_h1',
                        input_dim=inputs.shape[-1],
                        output_dim=self.dis_hidden_lists[0])(inputs)
            h2 = Linear('dis_h2',
                        input_dim=self.dis_hidden_lists[0],
                        output_dim=self.dis_hidden_lists[1])(h1)
            h2_logit = Linear('dis_logit',
                              input_dim=self.dis_hidden_lists[1],
                              output_dim=1,
                              act=lambda x: x)(h2)
            h2_prob = tf.nn.sigmoid(h2_logit)
            return h2_prob, h2_logit

    def get_batch(self, i, batch_nums, data):
        if i != batch_nums - 1:
            return data[i * self.batch_size:(i + 1) * self.batch_size, :]
        else:
            nums = self.tr_bat_nums
            temp = data[i * self.batch_size:, :]
            tm_zeros = np.zeros(shape=(nums, self.node_num, self.out_dim))
            concats = np.concatenate((temp, tm_zeros), axis=0)
            return concats

    def _build(self):
        self._call()
        self.x_real = tf.placeholder(tf.float32, [None, self.node_num, self.out_dim])
        self.x = tf.placeholder(tf.float32, [None, self.node_num, self.in_dim])
        self.concat_features_1 = self.GraphSAGE(self.x)
        self.concat_features_2 = self.GraphSAGE_ori(self.x)
        self.concat_features = tf.concat([self.concat_features_1, self.concat_features_2], axis=-1)

        D_real, D_logit_real = self.discriminator(self.x_real, reuse=False)
        G_samples = self.generate(self.concat_features, reuse=False)
        D_fake, D_logit_fake = self.discriminator(G_samples, reuse=True)
        d_loss_real = -tf.reduce_mean(D_logit_real)
        d_loss_fake = tf.reduce_mean(D_logit_fake)

        mse_loss = tf.reduce_mean(tf.square(G_samples - self.x_real))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = - d_loss_fake + self.reconstruction_coefficient * mse_loss

        self.t_vars = tf.trainable_variables()
        d_vars = [var for var in self.t_vars if 'dis' in var.name]
        g_vars = [var for var in self.t_vars if 'ge' in var.name]

        self.d_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        self.d_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                 ).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                                 ).minimize(self.g_loss, var_list=g_vars)
        self.generate_data = self.generate(self.concat_features, reuse=True)

    def train(self):
        self._build()
        tf.global_variables_initializer().run()
        start_time = time.time()
        self.d_loss_sum, self.g_loss_sum = [], []
        saver = tf.train.Saver(max_to_keep=1)

        for epoch in tqdm(range(self.epoch)):
            d_bat, g_bat = [], []
            for i in range(self.batch_nums):
                x_train = self.get_batch(i, self.batch_nums, self.train_x)
                y_train = self.get_batch(i, self.batch_nums, self.train_y)
                d_per = []
                for _ in range(5):
                    _, D_loss, _ = self.sess.run([self.d_optim, self.d_loss, self.d_clip],
                                                 feed_dict={self.x_real: y_train,
                                                            self.x: x_train})
                    d_per.append(D_loss)

                _, G_loss = self.sess.run([self.g_optim, self.g_loss],
                                          feed_dict={self.x_real: y_train,
                                                     self.x: x_train})
                d_bat.append(np.mean(np.asarray(d_per)))
                g_bat.append(G_loss)
            d_epoch, g_epoch = np.mean(np.asarray(d_bat)), np.mean(np.asarray(g_bat))
            self.d_loss_sum.append(d_epoch)
            self.g_loss_sum.append(g_epoch)
            if epoch % 20 == 0:
                use_time = (time.time() - start_time) / 60
                print('Epoch:[{}], G_Loss:{:.4f}, D_Loss:{:.8f}, usu_time:{:.2f}min'
                      .format(epoch, g_epoch, d_epoch, use_time))

        if self.save_model:
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
            saver.save(self.sess, self.save_model_path + '/GA_GAN')  # Save model

        rec = self.sess.run(self.generate_data, feed_dict={self.x: self.test_x, })
        rec = np.transpose(rec, (0, 2, 1)).reshape(-1, self.node_num)

        rec = self.scaler.inverse_transform(rec)
        true = np.transpose(self.test_y[:rec.shape[0]], (0, 2, 1)).reshape(-1, self.node_num)
        true = self.scaler.inverse_transform(true)

        return rec, true
