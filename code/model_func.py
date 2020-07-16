import math
import os
import pickle
import random
import time

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, Linear
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.optimizer import AdagradOptimizer

import some_configure as cfg
from data_pre import data_pre_dnngru
from modelGRU import Model_2_dnngru


def data_loader_dnngru(user_data, click_info, click_id, click_txt, click_img,
                       y_click_id, batch_size, gru_steps):
    # 4+2+1+128+128+1
    all_con = []
    for i in [
            user_data, click_info, click_id, click_txt, click_img, y_click_id
    ]:
        dataarray = np.array(i)
        if len(dataarray.shape) == 1:
            dataarray = dataarray.reshape(-1, 1)
        all_con.append(dataarray)
    all_con = np.concatenate(all_con, axis=1)

    batch_len = len(all_con) // batch_size
    one_size = batch_size * batch_len
    st = random.randint(0, len(user_data) - one_size)  #从数据中随机选择起始位置
    data = all_con[st:st + one_size].reshape((batch_size, batch_len, -1))
    epoch_size = (batch_len - 1) // gru_steps
    for i in range(epoch_size):
        yield np.copy(data[:, i * gru_steps:(i + 1) * gru_steps])


def data_loader_dnngru_pred(user_data, click_info, click_id, click_txt,
                            click_img, y_click_id, batch_size, gru_steps):
    """与训练集不同，测试集每一个数据都参与计算，为了正好凑够batch，最后一批补0"""
    # 4+2+1+128+128+1
    all_con = []
    for i in [
            user_data, click_info, click_id, click_txt, click_img, y_click_id
    ]:
        dataarray = np.array(i)
        if len(dataarray.shape) == 1:
            dataarray = dataarray.reshape(-1, 1)
        all_con.append(dataarray)
    all_con = np.concatenate(all_con, axis=1)
    # 仅保留if_pred为1的数据，第6列。if_pred为0: 训练集，1: 预测集， -1: 补零数据， 2:测试集倒数第1,2次预测，
    all_con = all_con[all_con[:, 5] >= 1]

    # 计算补0长度
    batch_len = math.ceil(len(all_con) / batch_size)
    batch_len += (gru_steps - batch_len % gru_steps)
    add0 = batch_size * batch_len - len(all_con)
    addarray = np.zeros([add0, all_con.shape[1]])
    addarray[:, 5] = -1  # 补0的数据，if_pred为-1
    # 补0
    data = np.vstack([all_con, addarray]).reshape((batch_size, batch_len, -1))

    epoch_size = batch_len // gru_steps
    for i in range(epoch_size):
        yield np.copy(data[:, i * gru_steps:(i + 1) * gru_steps])


def find_max50_inx(pred):
    # 最大50个位置
    result = []
    indices = np.argpartition(pred, -50, axis=1)[:, -50:]
    for i in range(len(indices)):
        result.append(indices[i, np.argsort(-pred[i, indices[i]])])
    result = np.concatenate(result).reshape(len(indices), -1)
    return result


def train_model_2_dnn(
        click_seq0_1,
        embedding_weight,
        save_model_path=f'./t2_dnngru/load_emb_save_model_{cfg.now_phase}'):
    model = Model_2_dnngru(click_seq0_1,
                           embedding_weight,
                           gru_steps=cfg.gru_steps,
                           gru_num_layers=1)
    if not os.path.exists('./t2_dnngru'):  # 创建储存中间数据，模型的文件夹
        os.makedirs('./t2_dnngru')
    data_path = f'./t2_dnngru/list6_0-{cfg.now_phase}.pkl'
    if not os.path.exists(data_path):
        # 处理数据
        user_data, click_info, click_id, click_txt, click_img, y_click_id = data_pre_dnngru(
            click_seq0_1)
        pickle.dump((user_data, click_info, click_id, click_txt, click_img,
                     y_click_id), open(data_path, 'wb'))
    else:
        user_data, click_info, click_id, click_txt, click_img, y_click_id = pickle.load(
            open(data_path, 'rb'))
    batch_len = len(user_data) // cfg.batch_size
    total_batch_size = (batch_len - 1) // cfg.gru_steps
    print("total_batch_size:", total_batch_size)
    #     opt = fluid.optimizer.Adam(learning_rate=0.05, parameter_list=model.parameters())
    bd = []
    lr_arr = [cfg.base_learning_rate]
    for i in range(1, cfg.max_epoch):
        bd.append(total_batch_size * i)
        new_lr = cfg.base_learning_rate * (cfg.lr_decay**max(
            i + 1 - cfg.epoch_start_decay, 0.0))
        lr_arr.append(new_lr)

    # 定义梯度的clip即取值范围
    grad_clip = fluid.clip.GradientClipByGlobalNorm(cfg.max_grad_norm)
    # 优化器选择adam，会降低训练准确率，选sgd会过拟合
    sgd = AdagradOptimizer(parameter_list=model.parameters(),
                           learning_rate=fluid.layers.piecewise_decay(
                               boundaries=bd, values=lr_arr),
                           grad_clip=grad_clip)

    model.train()
    for epoch in range(cfg.max_epoch):
        start_time = time.time()
        train_loader = data_loader_dnngru(user_data, click_info, click_id,
                                          click_txt, click_img, y_click_id,
                                          cfg.batch_size, cfg.gru_steps)

        init_hidden_data = np.zeros(
            (model.gru_num_layers, cfg.batch_size, model.gru_hidden_size),
            dtype='float32')
        init_hidden = to_variable(init_hidden_data)

        for batch_id, data in enumerate(train_loader):
            (user_data_pp, click_info_pp, click_id_pp, click_txt_pp,
             click_img_pp,
             y_click_id_pp) = (data[..., :4], data[..., 4:6], data[..., 6:7],
                               data[..., 7:128 + 7],
                               data[..., 128 + 7:256 + 7],
                               data[..., 256 + 7:256 + 8])
            user_data_pp = user_data_pp.astype(int)
            (user_id_pp, user_age_level_pp, user_gender_pp,
             user_city_level_pp) = (user_data_pp[:, :, 0],
                                    user_data_pp[:, :, 1],
                                    user_data_pp[:, :, 2],
                                    user_data_pp[:, :, 3])
            user_id_pp = to_variable(user_id_pp)
            user_age_level_pp = to_variable(user_age_level_pp)
            user_gender_pp = to_variable(user_gender_pp)
            user_city_level_pp = to_variable(user_city_level_pp)

            stay_data_pp = to_variable(
                click_info_pp[..., 0:1].astype('float32'))

            click_id_pp = to_variable(click_id_pp[..., 0].astype(int))
            click_txt_pp = to_variable(click_txt_pp.astype('float32'))
            click_img_pp = to_variable(click_img_pp.astype('float32'))

            y_click_id_pp = to_variable(y_click_id_pp.astype(int))

            pred_out, last_hidden = model([
                user_id_pp, user_age_level_pp, user_gender_pp,
                user_city_level_pp
            ], stay_data_pp, [click_id_pp, click_txt_pp, click_img_pp],
                                          init_hidden)
            init_hidden = last_hidden.detach()
            # 交叉熵
            loss = fluid.layers.softmax_with_cross_entropy(logits=pred_out,
                                                           label=y_click_id_pp,
                                                           soft_label=False,
                                                           axis=2)
            # 计算recall@50 指标
            pre_2d = fluid.layers.reshape(pred_out, shape=[-1, cfg.vocab_size])
            label_2d = fluid.layers.reshape(y_click_id_pp, shape=[-1, 1])
            acc = fluid.layers.accuracy(input=pre_2d, label=label_2d, k=50)
            acc_ = acc.numpy()[0]
            # 综合所有batch和序列长度的loss， 与5.2不同
            loss = fluid.layers.reduce_mean(loss)

            loss.backward()
            sgd.minimize(loss)
            model.clear_gradients()
            out_loss = loss.numpy()

            # 每隔一段时间可以打印信息
            if batch_id > 0 and batch_id % 100 == 1:
                print("-- Epoch:[%d]; Batch:[%d]; loss: %.5f, acc: %.5f" %
                      (epoch, batch_id, out_loss, acc_))

        print("one ecpoh finished", epoch)
        print("time cost ", time.time() - start_time)
        print("loss: %.5f, acc: %.5f" % (out_loss, acc_))

    fluid.save_dygraph(model.state_dict(), save_model_path)
    print("Saved model to: %s.\n" % save_model_path)


def eval_model_2_dnngru(
        click_seq_test0_1,
        embedding_weight,
        model_dict_path=f'./t2_dnngru/load_emb_save_model_{cfg.now_phase}'):
    """在测试集上预测"""
    result = []
    y_real = []
    if_pred = []
    pred_user_id = []

    model = Model_2_dnngru(click_seq_test0_1,
                           embedding_weight,
                           gru_steps=cfg.gru_steps,
                           gru_num_layers=1)
    model_dict, _ = fluid.load_dygraph(model_dict_path)
    model.set_dict(model_dict)
    model.eval()

    start_time = time.time()
    # 读取预测数据
    data_path = f'./t2_dnngru/test_list6_0-{cfg.now_phase}.pkl'
    if not os.path.exists(data_path):
        # 测试集数据
        user_data, click_info, click_id, click_txt, click_img, y_click_id = data_pre_dnngru(
            click_seq_test0_1)
        # 保存
        data_path = f'./t2_dnngru/test_list6_0-{cfg.now_phase}.pkl'
        pickle.dump((user_data, click_info, click_id, click_txt, click_img,
                     y_click_id), open(data_path, 'wb'))
    else:
        user_data, click_info, click_id, click_txt, click_img, y_click_id = pickle.load(
            open(data_path, 'rb'))

    print(np.unique(np.array(user_data)[:, 0]).shape)
    pred_loader = data_loader_dnngru_pred(user_data, click_info, click_id,
                                          click_txt, click_img, y_click_id,
                                          cfg.batch_size, cfg.gru_steps)

    init_hidden_data = np.zeros(
        (model.gru_num_layers, cfg.batch_size, model.gru_hidden_size),
        dtype='float32')
    init_hidden = to_variable(init_hidden_data)

    for batch_id, data in enumerate(pred_loader):
        (user_data_pp, click_info_pp, click_id_pp, click_txt_pp, click_img_pp,
         y_click_id_pp) = (data[..., :4], data[..., 4:6], data[..., 6:7],
                           data[..., 7:128 + 7], data[..., 128 + 7:256 + 7],
                           data[..., 256 + 7:256 + 8])
        user_data_pp = user_data_pp.astype(int)
        (user_id_pp, user_age_level_pp, user_gender_pp,
         user_city_level_pp) = (user_data_pp[:, :, 0], user_data_pp[:, :, 1],
                                user_data_pp[:, :, 2], user_data_pp[:, :, 3])
        user_id_pp = to_variable(user_id_pp)
        user_age_level_pp = to_variable(user_age_level_pp)
        user_gender_pp = to_variable(user_gender_pp)
        user_city_level_pp = to_variable(user_city_level_pp)

        stay_data_pp = to_variable(click_info_pp[..., 0:1].astype('float32'))

        click_id_pp = to_variable(click_id_pp[..., 0].astype(int))
        click_txt_pp = to_variable(click_txt_pp.astype('float32'))
        click_img_pp = to_variable(click_img_pp.astype('float32'))

        y_click_id_pp = to_variable(y_click_id_pp.astype(int))
        pred_out, last_hidden = model([
            user_id_pp, user_age_level_pp, user_gender_pp, user_city_level_pp
        ], stay_data_pp, [click_id_pp, click_txt_pp, click_img_pp],
                                      init_hidden)
        init_hidden = last_hidden.detach()
        # 交叉熵
        loss = fluid.layers.softmax_with_cross_entropy(logits=pred_out,
                                                       label=y_click_id_pp,
                                                       soft_label=False,
                                                       axis=2)
        # 计算recall@500 指标
        pre_2d = fluid.layers.reshape(pred_out, shape=[-1, cfg.vocab_size])
        label_2d = fluid.layers.reshape(y_click_id_pp, shape=[-1, 1])

        result.append(find_max50_inx(pre_2d.numpy()))
        y_real.append(label_2d.numpy())
        if_pred.append(click_info_pp[..., 1])
        pred_user_id.append(user_id_pp.numpy())

        acc = fluid.layers.accuracy(input=pre_2d, label=label_2d, k=50)
        acc_ = acc.numpy()[0]
        # 综合所有batch和序列长度的loss， 与5.2不同
        loss = fluid.layers.reduce_mean(loss)

        out_loss = loss.numpy()

        # 每隔一段时间可以打印信息
        if batch_id > 0 and batch_id % 100 == 1:
            print("-- Batch:[%d]; loss: %.5f, acc: %.5f" %
                  (batch_id, out_loss, acc_))

    print("finished")
    print("-- Batch:[%d]; loss: %.5f, acc: %.5f" % (batch_id, out_loss, acc_))
    print("time cost ", time.time() - start_time)

    pred_item_id = np.array(result).reshape(-1, 50)
    real_item_id = np.array(y_real).reshape(-1)
    if_pred = np.array(if_pred).reshape(-1)
    pred_user_id = np.array(pred_user_id)
    user_id_seq = np.array(pred_user_id).reshape(-1)

    return real_item_id, pred_item_id, if_pred, user_id_seq


def hitrate50(real_item_id,
              pred_item_id,
              user_id_seq,
              submit_fname,
              if_pred=None):
    """计算测试集的预测准确率，50个中包括正确答案的概率"""
    bingo = []  # 测试集所有点击预测的击中率，最后一次点击没有label不算在内
    last_click_bingo = []  # 测试集倒数第二次点击，预测倒数第一次
    final_result = []
    ii = 0
    user_id_uni = np.unique(user_id_seq, return_counts=True)
    onece_user = user_id_uni[0][np.where(user_id_uni[1] == 1)]
    for i in range(len(real_item_id)):
        if if_pred[i] >= 1 and real_item_id[i] != 0:
            if real_item_id[i] in pred_item_id[i]:
                bingo.append(1)
            else:
                bingo.append(0)
            if if_pred[i] == 2 or user_id_seq[
                    i] in onece_user:  # 最后一次预测，if_pred为2，有时预测集只有一个，此时if_pred为1，用户数量为1
                final_result.append([user_id_seq[i]] +
                                    pred_item_id[i].tolist())
                ii += 1
                if real_item_id[i] in pred_item_id[i]:
                    last_click_bingo.append(1)
                else:
                    last_click_bingo.append(0)
    print(f"测试集hitrate50: {sum(bingo)/len(bingo):.4f}")
    print(f"最后一次预测: {sum(last_click_bingo)/len(last_click_bingo):.4f}")

    final_result = np.array(final_result)
    np.savetxt(submit_fname, final_result, fmt='%d', delimiter=',')
    print(f'最后一次预测，存在{submit_fname}')
    return final_result