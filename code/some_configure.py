now_phase = 9
train_dir = '../data/underexpose_train'
test_dir = '../data/underexpose_test'

# 模型参数
vocab_size = 150000
num_layers = 1
batch_size = 256
hidden_size = 200
gru_steps = 5
init_scale = 0.1
max_grad_norm = 5.0
epoch_start_decay = 10
max_epoch = 10
dropout = 0.0
lr_decay = 0.5
base_learning_rate = 0.05

if_train = True