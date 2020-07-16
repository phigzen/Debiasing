import random
import pickle
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding


def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[query_token]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:] # 最大k个的位置
    indices = indices[np.argsort(-flat[indices])] # 取出值，从大到小排序
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(i)))

class SkipGram(fluid.dygraph.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        #vocab_size定义了这个skipgram这个模型的词表大小
        #embedding_size定义了词向量的维度是多少
        #init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        #使用paddle.fluid.dygraph提供的Embedding函数，构造一个词向量参数
        #这个参数的大小为：[self.vocab_size, self.embedding_size]
        #数据类型为：float32
        #这个参数的名称为：embedding_para
        #这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))

        #使用paddle.fluid.dygraph提供的Embedding函数，构造另外一个词向量参数
        #这个参数的大小为：[self.vocab_size, self.embedding_size]
        #数据类型为：float32
        #这个参数的名称为：embedding_para_out
        #这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        #跟上面不同的是，这个参数的名称跟上面不同，因此，
        #embedding_para_out和embedding_para虽然有相同的shape，但是权重不共享
        self.embedding_out = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))

    #定义网络的前向计算逻辑
    #center_words是一个tensor（mini-batch），表示中心词
    #target_words是一个tensor（mini-batch），表示目标词
    #label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    #用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, center_words, target_words, label):
        #首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        #这里center_words和eval_words_emb查询的是一个相同的参数
        #而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        #center_words_emb = [batch_size, embedding_size]
        #target_words_emb = [batch_size, embedding_size]
        #我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = fluid.layers.elementwise_mul(center_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim = -1)
        word_sim = fluid.layers.reshape(word_sim, shape=[-1])
        pred = fluid.layers.sigmoid(word_sim)

        #通过估计的输出概率定义损失函数，注意我们使用的是sigmoid_cross_entropy_with_logits函数
        #将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        #返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss

def gen_negative_samp(iseq, all_item, negative_len=5):
    # item的前1个，后2个为正样本。可以考虑增加停留时间的影响
    data_set = []
    vocab_size = len(all_item)
    len_iseq = len(iseq)
    for center_idx in range(len_iseq):
        center = iseq[center_idx]
        positive_range = (max(0,center_idx-1), min(center_idx+2,len_iseq-1))
        positive_candidates = [iseq[idx] for idx in range(positive_range[0], positive_range[1]+1) if idx != center_idx]
        # 所有其他的item都是负样本
        for positive in positive_candidates:
            data_set.append((center, positive, 1))
            ct = 0
            while ct < negative_len:
                negative= all_item[random.randint(0, vocab_size-1)]
                if negative not in positive_candidates:
                    data_set.append((center, negative, 0))
                    ct += 1
    return data_set

def build_batch(dataset, batch_size, epoch_num):
    #center_word_batch缓存batch_size个中心词
    center_word_batch = []
    #target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    #label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        #每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for center_word, target_word, label in dataset:
            #遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            #当样本积攒到一个batch_size后，我们把数据都返回回来
            #在这里我们使用numpy的array函数把list封装成tensor
            #并使用python的迭代器机制，将数据yield出来
            #使用迭代器的好处是可以节省内存
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"), \
                    np.array(target_word_batch).astype("int64"), \
                    np.array(label_batch).astype("float32")
                center_word_batch = []
                target_word_batch = []
                label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"), \
            np.array(target_word_batch).astype("int64"), \
            np.array(label_batch).astype("float32")

def gen_item_embedding(click_seq0_1):
    all_item = click_seq0_1.item_feat.item_id.tolist()
    dataset_res = []
    for kkey in click_seq0_1.click_seq.keys():
        iseq = click_seq0_1.click_seq[kkey]['item']
        # item的前1个，后2个为正样本。可以考虑增加停留时间的影响
        dataset_tmp = gen_negative_samp(iseq,all_item)
        dataset_res += dataset_tmp
    
    batch_size = 512
    epoch_num = 3
    embedding_size = 200
    step = 0
    learning_rate = 0.001
    vocab_size = click_seq0_1.item_id_size

    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        #通过我们定义的SkipGram类，来构造一个Skip-gram模型网络
        skip_gram_model = SkipGram(vocab_size, embedding_size)
        #构造训练这个网络的优化器
        adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list = skip_gram_model.parameters())

        #使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
        for center_words, target_words, label in build_batch(
            dataset_res, batch_size, epoch_num):
            #使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
            center_words_var = fluid.dygraph.to_variable(center_words)
            target_words_var = fluid.dygraph.to_variable(target_words)
            label_var = fluid.dygraph.to_variable(label)

            #将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
            pred, loss = skip_gram_model(
                center_words_var, target_words_var, label_var)

            #通过backward函数，让程序自动完成反向计算
            loss.backward()
            #通过minimize函数，让程序根据loss，完成一步对参数的优化更新
            adam.minimize(loss)
            #使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
            skip_gram_model.clear_gradients()

            #每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
            step += 1
            if step % 500 == 0:
                print("step %d, loss %.3f" % (step, loss.numpy()[0]))
            
        model_dict = skip_gram_model.state_dict()
        fluid.save_dygraph(model_dict, './item_embed/skip_model') # 会自动创建文件夹

    embedding_weight = skip_gram_model.embedding.weight.numpy()
    pickle.dump(embedding_weight,open('./item_embed/embedding_array','wb'))
    return embedding_weight