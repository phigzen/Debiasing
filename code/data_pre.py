# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:52:00 2020
准备数据
@author: Phigzen
"""

import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import some_configure as cfg
from evalue import _create_answer_file_for_evaluation

NAN_FILL = 0  # 未知数据，新数据，nan填充


def gender2num(gender):
    if gender == 'F':
        return 1
    elif gender == 'M':
        return 2
    else:
        return 0


def get_item_feat(path):
    with open(path, 'r', encoding="UTF-8") as f:
        data = f.readlines()
    result = []
    col = ['item_id', 'txt_vec', 'img_vec']
    for i in range(len(data)):
        result.append(eval(data[i]))
    return pd.DataFrame(result, columns=col)


def gen_click_seq(click_data, click_user):
    click_seq = {}
    for ius in click_user:
        idata = click_data[click_data.user_id == ius].copy().sort_values(
            by='time')
        # 本次操作与下一次操作的间隔
        idata['stay'] = idata.time.shift(-1) - idata.time
        # 下一次操作后停留时间，即下下一次与下一次操作的间隔
        idata['next_click'] = idata.item_id.shift(-1)
        idata['next_stay'] = idata.time.shift(-2) - idata.time.shift(-1)
        click_seq[int(ius)] = {
            'item': idata.item_id.tolist(),
            'time': idata.time.tolist(),
            'stay': idata.stay.tolist(),
            'next_click': idata.next_click.tolist(),
            'next_stay': idata.next_stay.tolist(),
            'if_pred': idata.if_pred.tolist(),
        }
    return click_seq


class SeqData(object):
    """每个用户的点击item历史序列"""
    def __init__(self, user_feat=None, item_feat=None, click_data=None):
        if user_feat is None:
            user_feat_path = '../data/underexpose_train/underexpose_user_feat.csv'
            user_feat = pd.read_csv(user_feat_path,
                                    header=None,
                                    names=[
                                        'user_id', 'user_age_level',
                                        'user_gender', 'user_city_level'
                                    ])
            user_feat = user_feat.fillna(NAN_FILL)
        if item_feat is None:
            item_feat_path = '../data/underexpose_train/underexpose_item_feat.csv'
            item_feat = get_item_feat(item_feat_path)
        if click_data is None:
            click_path = '../data/underexpose_train/underexpose_train_click-0.csv'
            click_data = pd.read_csv(click_path,
                                     header=None,
                                     names=['user_id', 'item_id', 'time'])
        click_user = click_data['user_id'].unique()
        click_item = click_data['item_id'].unique()
        user_uni = set(user_feat.user_id)
        item_uni = set(item_feat.item_id)
        self.click_user_num = len(click_user)
        self.click_item_num = len(click_item)
        new_user = sum(~np.isin(click_user, list(user_uni)))
        new_item = sum(~np.isin(click_item, list(item_uni)))
        print(
            f'用户数据有{len(user_uni)}个用户，item数据有{len(item_uni)}个，\n点击数据共有{self.click_user_num}个用户，{self.click_item_num}个item，\n其中{new_user}个新用户，{new_item}个新item'
        )
        self.user_feat = user_feat
        self.item_feat = item_feat
        self.click_data = click_data
        self.user_id_size = 50000
        self.user_age_level_size = len(self.user_feat.user_age_level.unique())
        self.user_gender_size = len(self.user_feat.user_gender.unique())
        self.user_city_level_size = len(
            self.user_feat.user_city_level.unique())
        self.item_id_size = 150000

        self.click_seq = gen_click_seq(click_data, click_user)


def read_files(now_phase):
    item_feat_path = os.path.join(cfg.train_dir, 'underexpose_item_feat.csv')
    user_feat_path = os.path.join(cfg.train_dir, 'underexpose_user_feat.csv')

    item_feat = get_item_feat(item_feat_path)
    user_feat = pd.read_csv(
        user_feat_path,
        header=None,
        names=['user_id', 'user_age_level', 'user_gender', 'user_city_level'])

    qclick_data = []
    tclick_data = []
    train_click_data = []
    for i in range(0, now_phase + 1):
        qtest_path = os.path.join(cfg.test_dir, f'underexpose_test_qtime-{i}.csv')
        test_path = os.path.join(cfg.test_dir, f'underexpose_test_click-{i}.csv')
        train_click_path = os.path.join(cfg.train_dir,
                                        f'underexpose_train_click-{i}.csv')

        iqclick_data = pd.read_csv(qtest_path,
                                   header=None,
                                   names=['user_id', 'time'])
        # 待递交的item_id填充为-1
        iqclick_data['item_id'] = -1
        iqclick_data = iqclick_data[['user_id', 'item_id', 'time']]
        qclick_data.append(iqclick_data)

        test_click_data = pd.read_csv(test_path,
                                      header=None,
                                      names=['user_id', 'item_id', 'time'])
        tclick_data.append(test_click_data)

        train_click_data0 = pd.read_csv(train_click_path,
                                        header=None,
                                        names=['user_id', 'item_id', 'time'])
        train_click_data.append(train_click_data0)
    qclick_data = pd.concat(qclick_data, ignore_index=True)
    tclick_data = pd.concat(tclick_data, ignore_index=True)
    train_click_data = pd.concat(train_click_data, ignore_index=True)
    return item_feat, user_feat, train_click_data, tclick_data, qclick_data


def find2(x):
    x = x.sort_values('time')
    if len(x) < 2:
        print(x, '该用户只有一次点击记录')  # 用户只有一次点击数据
    else:
        x.loc[x.iloc[-2].name, 'if_pred'] = 2
    return x


def gen_click(now_phase, train_click_data, tclick_data, user_feat, item_feat):
    train_click_data['if_pred'] = 0
    tclick_data['if_pred'] = 1  #c测试集设为1

    #测试集中，如果是倒数第二次的点击，if_pred=2，表示最后一次预测结果，作为模拟提交使用。倒数第一次点击设为-1
    tclick_data_1 = tclick_data.groupby('user_id').apply(lambda x: find2(x))
    tclick_data_1 = tclick_data_1.reset_index(drop=True)
    # 训练融合预测
    train_test_1 = train_click_data.append(tclick_data_1)
    click_seq0_1 = SeqData(user_feat, item_feat, train_click_data)
    if not os.path.exists('./pkl_data'):
        os.makedirs('./pkl_data')
    pickle.dump(click_seq0_1,
                open(f'./pkl_data/click_seq0_{now_phase}.pkl', 'wb'))

    click_seq_test0_1 = SeqData(user_feat, item_feat, train_test_1)
    pickle.dump(click_seq_test0_1,
                open(f'./pkl_data/click_seq_test0_{now_phase}.pkl', 'wb'))
    return click_seq0_1, click_seq_test0_1


def data_pre_dnngru(click_seq0_1):
    """点击序列，及点击对应的用户信息，item信息，点击停留时间"""
    user_feat = click_seq0_1.user_feat.copy()
    user_feat['user_gender'] = user_feat['user_gender'].apply(
        lambda x: gender2num(x))
    user_feat = user_feat.fillna(0).astype(int)
    item_feat = click_seq0_1.item_feat.copy()
    click_seq = click_seq0_1.click_seq.copy()
    user_data = []  #用户的四个字段
    click_info = []  # 当前点击停留的时间，当前点击属于训练还是预测
    click_id = []  # 当前点击的item_id
    click_txt = []  # 当前点击的txt
    click_img = []  # 当前点击的img
    y_click_id = []  # 下一次点击的item_id
    user_feat_user_id = user_feat.user_id.values
    item_id_valus = item_feat.item_id.values
    for iuser in tqdm(click_seq):
        user_click = click_seq[iuser]
        if iuser in user_feat_user_id:
            _, age, gender, city = user_feat[user_feat.user_id ==
                                             iuser].values[0]
        else:
            age, gender, city = 0, 0, 0

        click_list = user_click['item']
        len_click_list = len(click_list)
        for ilen in range(len_click_list):
            if ilen == len_click_list - 1:  # 当前点击是用户点击序列的最后一次点击，则此次点击数据全部设为0
                click_id.append(0)
                y_click_id.append(0)
                click_txt.append([0] * 128)
                click_img.append([0] * 128)
                click_info.append([0, -1])  # if_pred设为-1
                user_data.append([0, 0, 0, 0])
            else:
                itm = click_list[ilen]
                click_id.append(click_list[ilen])
                y_click_id.append(click_list[ilen + 1])
                if itm in item_id_valus:
                    click_txt.append(item_feat.loc[item_feat.item_id ==
                                                   itm, 'txt_vec'].iat[0])
                    click_img.append(item_feat.loc[item_feat.item_id ==
                                                   itm, 'img_vec'].iat[0])
                else:
                    click_txt.append([0] * 128)
                    click_img.append([0] * 128)
                click_info.append(
                    [user_click['stay'][ilen], user_click['if_pred'][ilen]])
                user_data.append([iuser, age, gender, city])
    return user_data, click_info, click_id, click_txt, click_img, y_click_id


def gen_answer_file():
    for ird in range(cfg.now_phase + 1):
        tclick_data = []
        for i in range(0, ird + 1):
            test_path = os.path.join(cfg.test_dir,
                                     f'underexpose_test_click-{i}.csv')
            test_click_data = pd.read_csv(test_path,
                                          header=None,
                                          names=['user_id', 'item_id', 'time'])
            tclick_data.append(test_click_data)

        tclick_data = pd.concat(tclick_data, ignore_index=True)
        # 按照id号筛选，官方每个阶段T的id是有规律的
        tclick_data = tclick_data[tclick_data.user_id % 11 == ird]
        last_click = tclick_data.groupby('user_id').apply(
            lambda x: x.loc[x.time == max(x['time'])])
        if not os.path.exists('./test_last'):
            os.makedirs('./test_last')
        np.savetxt(f'./test_last/underexpose_test_qtime_with_answer-{ird}.csv',
                   last_click.values.astype(int),
                   fmt='%d',
                   delimiter=',')
    if not os.path.exists('./debias_track_answer'):
        os.makedirs('./debias_track_answer')
    _create_answer_file_for_evaluation(
        f'./debias_track_answer/debias_track_answer-{cfg.now_phase}.csv')
