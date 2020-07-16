# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:59:37 2020

@author: Phigzen
"""

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.dygraph import Linear
import paddle.fluid.dygraph as dygraph


class SimpleGRURNN(fluid.Layer):
    def __init__(
            self,
            hidden_size,  # 隐层维度
            num_steps,  # 更新一次梯度时选取序列的长度，输入实例的个数为batch_size * num_steps
            num_layers=2,  # gru的层数
            init_scale=0.1,  # 初始化参数
            dropout=None):  # dropout的值
        super(SimpleGRURNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._num_steps = num_steps

        self.weight_1_arr = []
        self.weight_2_arr = []
        self.weight_3_arr = []
        self.bias_1_arr = []
        self.bias_2_arr = []
        self.mask_array = []
        # 创建参数
        for i in range(self._num_layers):
            # 根据gru 的层数循环创建，默认是一层。
            # 根据gru公式 构建x和h的w，为2*2 ，适用于公式1和公式3
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 2],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w1_%d' % i, weight_1))
            # 适用于公式2的x对应的参数
            weight_2 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size, self._hidden_size],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_2_arr.append(self.add_parameter('w2_%d' % i, weight_2))
            # 适用于公式2的r和h点积的参数
            weight_3 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size, self._hidden_size],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_3_arr.append(self.add_parameter('w3_%d' % i, weight_3))
            # 适用于公式1和公式3的偏置参数
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_1_arr.append(self.add_parameter('b1_%d' % i, bias_1))
            #适用于公式2的偏置参数
            bias_2 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 1],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_2_arr.append(self.add_parameter('b2_%d' % i, bias_2))

    def forward(self, input_embedding, init_hidden=None):
        hidden_array = []
        # 传入上一个一个隐层的值作为输入
        for i in range(self._num_layers):
            hidden_array.append(init_hidden[i])

        res = []
        # 遍历序列
        for index in range(self._num_steps):
            # 选取序列中当前点的输入embedding
            step_input = input_embedding[:, index, :]
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                weight_1 = self.weight_1_arr[k]
                weight_2 = self.weight_2_arr[k]
                weight_3 = self.weight_3_arr[k]
                bias_1 = self.bias_1_arr[k]
                bias_2 = self.bias_2_arr[k]
                # 公式1和公式3
                nn = fluid.layers.concat([step_input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)
                gate_input = fluid.layers.elementwise_add(gate_input, bias_1)
                u, r = fluid.layers.split(gate_input,
                                          num_or_sections=2,
                                          dim=-1)
                # 公式2
                hidden_c = fluid.layers.tanh(
                    fluid.layers.elementwise_add(
                        fluid.layers.matmul(x=step_input, y=weight_2) +
                        fluid.layers.matmul(
                            x=(fluid.layers.sigmoid(r) * pre_hidden),
                            y=weight_3), bias_2))
                # 公式4
                hidden_state = fluid.layers.sigmoid(u) * pre_hidden + (
                    1.0 - fluid.layers.sigmoid(u)) * hidden_c
                hidden_array[k] = hidden_state
                step_input = hidden_state

                if self._dropout is not None and self._dropout > 0.0:
                    step_input = fluid.layers.dropout(
                        step_input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(step_input)
        # 最后输出
        real_res = fluid.layers.concat(res, 1)
        real_res = fluid.layers.reshape(
            real_res, [-1, self._num_steps, self._hidden_size])
        # 隐层输出
        last_hidden = fluid.layers.concat(hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        return real_res, last_hidden


class Model_2_dnngru(dygraph.layers.Layer):
    """先使用dnn处理得到包含物品信息，用户信息，点击信息的向量，再将向量输入到GRU"""
    def __init__(self,
                 Data,
                 embedding_weight,
                 gru_steps=10,
                 gru_num_layers=1,
                 init_scale=0.1):
        super(Model_2_dnngru, self).__init__()
        self.init_scale = init_scale

        self.Data = Data
        USR_ID_SIZE = self.Data.user_id_size
        self.usr_id_emb = Embedding([USR_ID_SIZE, 32])
        self.usr_fc = Linear(32, 32)

        USR_GENDER_SIZE = self.Data.user_gender_size + 1
        self.usr_gender_emb = Embedding([USR_GENDER_SIZE, 4])
        self.usr_gender_fc = Linear(4, 4)

        USR_AGE_LEV_SIZE = self.Data.user_age_level_size + 1
        self.usr_age_emb = Embedding([USR_AGE_LEV_SIZE, 16])
        self.usr_age_fc = Linear(16, 16)

        USR_CITY_LEV_SIZE = self.Data.user_city_level_size + 1
        self.usr_city_emb = Embedding([USR_CITY_LEV_SIZE, 16])
        self.usr_city_fc = Linear(16, 16)

        ITM_ID_SIZE = self.Data.item_id_size
        item_emb_weight = fluid.ParamAttr(
            learning_rate=0.5,
            initializer=fluid.initializer.NumpyArrayInitializer(
                embedding_weight),
            trainable=False)
        self.itm_id_emb = Embedding([ITM_ID_SIZE, 200],
                                    param_attr=item_emb_weight)

        self.click_fc = Linear(1, 16)
        # (32+4+16+16)+(200+128+128)+16=540
        #         self.all_combined = Linear(540, 512, act='tanh')

        # GRU，前向传播时，实际预测时取最后一步的结果，构建序列时0填充在实际元素之前
        self.gru_steps = gru_steps
        self.gru_num_layers = gru_num_layers
        self.gru_hidden_size = 540
        self.simple_gru_rnn = SimpleGRURNN(hidden_size=self.gru_hidden_size,
                                           num_steps=self.gru_steps,
                                           num_layers=self.gru_num_layers,
                                           init_scale=0.1,
                                           dropout=None)

        # 最后映射到每一个维度概率的参数
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.gru_hidden_size, ITM_ID_SIZE],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[ITM_ID_SIZE],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

        #         self.gru_fc = Linear(self.gru_hidden_size, 512, act='sigmoid')
        self.gru_fc_out = Linear(self.gru_hidden_size, ITM_ID_SIZE)

    def forward(self, usr_var, stay, hist_itm_var, init_hidden):
        # 输入有用户信息，历史上点击过的item信息，最近一次点击及其停留时间
        user_id, user_age_level, user_gender, user_city_level = usr_var
        item_id_list, txt_vec_list, img_vec_list = hist_itm_var
        # self.itm_id_emb(item_id_list) None*steps*200
        # txt_vec_list, None*steps*128
        # img_vec_list, None*steps*128
        # item_input, None*steps*456
        combi = fluid.layers.concat([
            self.usr_fc(self.usr_id_emb(user_id)),
            self.usr_gender_fc(self.usr_gender_emb(user_gender)),
            self.usr_age_fc(self.usr_age_emb(user_age_level)),
            self.usr_city_fc(self.usr_city_emb(user_city_level)),
            self.itm_id_emb(item_id_list), txt_vec_list, img_vec_list,
            self.click_fc(stay)
        ],
                                    axis=2)
        shape1 = combi.shape[1]
        combi = fluid.layers.slice(combi, [1], [shape1 - self.gru_steps],
                                   [shape1])

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.gru_num_layers, -1, self.gru_hidden_size])
        #         x_emb = self.itm_id_emb(item_id_list)
        gru_result, last_hidden = self.simple_gru_rnn(
            combi, init_h)  # None*steps*hidden
        #         projection = fluid.layers.matmul(gru_result, self.softmax_weight)
        #         projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = self.gru_fc_out(gru_result)
        return projection, last_hidden
