from sklearn.datasets import load_boston
from tqdm import tqdm
from sklearn.utils import shuffle, resample
import numpy as np
from xhp_flow.nn.node import Placeholder,Linear,Sigmoid,ReLu,Leakrelu,Elu,Tanh,LSTM
from xhp_flow.optimize.optimize import toplogical_sort,run_steps,forward,save_model,load_model,Auto_update_lr,Visual_gradient,Grad_Clipping_Disappearance,SUW,\
SGD,\
Momentum,\
Adagrad,\
RMSProp,\
AdaDelta,\
Adam,\
AdaMax,\
Nadam,\
NadaMax

from xhp_flow.loss.loss import MSE,EntropyCrossLossWithSoftmax
import matplotlib.pyplot as plt
import torch
from glob import glob
from data_prepare_for_many import *
torch.manual_seed(1)
MAX_LENGTH = 100
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
length = 10
predict_length = 1
batch_size = 512
file_path_train = np.array(glob('data/train/*'))
# file_path_test = np.array(glob('data/test/*'))
# file_path_valid = np.array(glob('data/valid/*'))
# Training_generator1, Test, Valid, WholeSet= get_dataloader(batch_size,length,predict_length)
Training_generator, WholeSet_train = get_dataloader(batch_size, length, predict_length, file_path_train, 'train')
x1, y = next(iter(Training_generator))
input_x, y = x1.numpy(), y.numpy()


class LSTMtest():
    def __init__(self, input_size=8, hidden_size=256, output_size=1):
        self.x, self.y = Placeholder(name='x', is_trainable=False), Placeholder(name='y', is_trainable=False)
        self.wf, self.bf = Placeholder(name='wf'), Placeholder(name='bf')
        self.wi, self.bi = Placeholder(name='wi'), Placeholder(name='bi')
        self.wc, self.bc = Placeholder(name='wc'), Placeholder(name='bc')
        self.wo, self.bo = Placeholder(name='wo'), Placeholder(name='bo')

        self.w0, self.b0 = Placeholder(name='w0'), Placeholder(name='b0')
        self.w1, self.b1 = Placeholder(name='w1'), Placeholder(name='b1')
        self.w2, self.b2 = Placeholder(name='w2'), Placeholder(name='b2')

        self.linear0 = Linear(self.x, self.w0, self.b0, name='linear0')
        self.lstm = LSTM(self.linear0, self.wf, self.wi, self.wc, self.wo, self.bf, self.bi, self.bc, self.bo,
                         input_size, hidden_size, name='LSTM')
        self.linear1 = Linear(self.lstm, self.w1, self.b1, name='linear1')
        self.output = ReLu(self.linear1, name='Elu')
        self.y_pre = Linear(self.output, self.w2, self.b2, name='output_pre')
        self.MSE_loss = MSE(self.y_pre, self.y, name='MSE')

        # 初始化数据结构
        self.feed_dict = {
            self.x: input_x,
            self.y: y,
            self.w0: np.random.rand(input_x.shape[2], input_size),
            self.b0: np.zeros(input_size),
            self.wf: np.random.rand(input_size + hidden_size, hidden_size),
            self.bf: np.zeros(hidden_size),
            self.wi: np.random.rand(input_size + hidden_size, hidden_size),
            self.bi: np.zeros(hidden_size),
            self.wc: np.random.rand(input_size + hidden_size, hidden_size),
            self.bc: np.zeros(hidden_size),
            self.wo: np.random.rand(input_size + hidden_size, hidden_size),
            self.bo: np.zeros(hidden_size),
            self.w1: np.random.rand(hidden_size, hidden_size),
            self.b1: np.zeros(hidden_size),
            self.w2: np.random.rand(hidden_size, output_size),
            self.b2: np.zeros(output_size),
        }


# In[ ]:
lstm = LSTMtest(16, 16, 1)
def train(model, train_data, epoch=6, learning_rate=1e-3):
    # 开始训练
    losses = []
    loss_min = np.inf
    graph_sort_lstm = toplogical_sort(model.feed_dict)  # 拓扑排序
    optim = Adam(graph_sort_lstm)
    update_lr = Auto_update_lr(lr=learning_rate, alpha=0.1, patiences=20, print_=True)
    for e in range(epoch):
        for X, Y in train_data:
            X, Y = X.numpy(), Y.numpy()
            model.x.value = X
            model.y.value = Y
            run_steps(graph_sort_lstm)
            # if model.y_pre.value is not None:
            # print(model.y_pre.value.shape,Y.shape)
            learning_rate = update_lr.lr
            optim.update(learning_rate=learning_rate)
            Visual_gradient(model)
            Grad_Clipping_Disappearance(model, 5)
            loss = model.MSE_loss.value
            losses.append(loss)
        update_lr.updata(np.mean(np.mean(losses)))
        print("epoch:{}/{},loss:{:.6f}".format(e,epoch,np.mean(losses)))
        if np.mean(losses) < loss_min:
            print('loss is {:.6f}, is decreasing!! save moddel'.format(np.mean(losses)))
            save_model("model/lstm.xhp", model)
            loss_min = np.mean(losses)
    print('min loss:',loss_min)
    plt.plot(losses)
    plt.savefig("image/lstm_loss.png")
    plt.show()


train(lstm, Training_generator, 1000,0.01)