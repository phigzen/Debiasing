import os
import pickle

import paddle.fluid as fluid

import some_configure as cfg
from data_pre import gen_click, read_files, gen_answer_file
from evalue import evaluate
from model_func import train_model_2_dnn, eval_model_2_dnngru, hitrate50
from modelGRU import Model_2_dnngru
from skipgram import gen_item_embedding


print('当前比赛phase：', cfg.now_phase)
# 数据准备
item_feat, user_feat, train_click_data, tclick_data, qclick_data = read_files(
    cfg.now_phase)
train_seq_path = f'./pkl_data/click_seq0_{cfg.now_phase}.pkl'
test_seq_path = f'./pkl_data/click_seq_test0_{cfg.now_phase}.pkl'
if not os.path.exists(train_seq_path):
    click_seq0_1, click_seq_test0_1 = gen_click(cfg.now_phase,
                                                train_click_data, tclick_data,
                                                user_feat, item_feat)
else:
    click_seq0_1 = pickle.load(open(train_seq_path, 'rb'))
    click_seq_test0_1 = pickle.load(open(test_seq_path, 'rb'))

# 训练好的item_embedding
item_embedding_path = './item_embed/embedding_array'
if os.path.exists(item_embedding_path):
    embedding_weight = pickle.load(open(item_embedding_path, 'rb'))
else:
    embedding_weight = gen_item_embedding(click_seq0_1)

if cfg.if_train:
    print('开始训练')
    place = fluid.CUDAPlace(0)
    # place = fluid.CPUPlace()
    # place = core.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        train_model_2_dnn(click_seq0_1, embedding_weight)
    print('训练结束')

print('开始预测')
place = fluid.CUDAPlace(0)
with fluid.dygraph.guard(place):
    real_item_id, pred_item_id, if_pred, user_id_seq = eval_model_2_dnngru(
        click_seq_test0_1, embedding_weight)
print('预测结束')

submit_fname = f'./t2_dnngru/underexpose_submit-{cfg.now_phase}.csv'
final_result = hitrate50(real_item_id, pred_item_id, user_id_seq, submit_fname,
                         if_pred)

answer_fname = f'./debias_track_answer/debias_track_answer-{cfg.now_phase}.csv'
if not os.path.exists(answer_fname):
    gen_answer_file()
# ndcg_50_full,ndcg_50_half,hitrate_50_full,hitrate_50_half
eva_res = evaluate(1, submit_fname, answer_fname, 1591315200 + 1)
eva_res6 = evaluate(1, submit_fname, answer_fname, 1589558399 + 1)