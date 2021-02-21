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

# 加载数据
dataset = load_boston()
"""
print(dataset['feature_names'])
print(dataset['data'].shape)
print(dataset['target'].shape)
"""
x_ = dataset['data']
y_ = dataset['target']
mean = np.mean(x_, axis=0)
std = np.std(x_, axis=0)
mean_y = np.mean(y_, axis=0)
std_y = np.std(y_, axis=0)
# Normalize data
x_ = (x_ - mean) / std
y_ = (y_ - mean_y) / std_y


# 定义网络
class MLP():
    def __init__(self, x_, y_):
        self.x, self.y = Placeholder(name='x', is_trainable=False), Placeholder(name='y', is_trainable=False)
        self.w1, self.b1 = Placeholder(name='w1'), Placeholder(name='b1')
        self.w2, self.b2 = Placeholder(name='w2'), Placeholder(name='b2')
        self.w3, self.b3 = Placeholder(name='w3'), Placeholder(name='b3')

        self.output1 = Linear(self.x, self.w1, self.b1, name='linear1')
        self.output2 = ReLu(self.output1, name='Relu')
        self.output3 = Linear(self.output2, self.w2, self.b2, name='linear2')
        self.output4 = ReLu(self.output3, name='Relu')
        self.y_pre = Linear(self.output4, self.w3, self.b3, name='linear3')
        self.MSE_loss = MSE(self.y_pre, self.y, name='MSE')

        hidden = 10
        hidden1 = 16
        output = 1
        # 初始化数据结构
        self.feed_dict = {
            self.x: x_,
            self.y: y_,
            self.w1: np.random.rand(x_.shape[1], hidden),
            self.b1: np.zeros(hidden),
            self.w2: np.random.rand(hidden, hidden1),
            self.b2: np.zeros(hidden1),
            self.w3: np.random.rand(hidden1, output),
            self.b3: np.zeros(output)}


batch_size = 64
mlp = MLP(x_, y_)

m = x_.shape[0]
steps_per_epoch = m // batch_size


def train(model, epoch=1000, learning_rate=1e-3, steps_per_epoch=steps_per_epoch):
    # 开始训练
    losses = []
    loss_min = np.inf
    graph_sort = toplogical_sort(model.feed_dict)  # 拓扑排序
    optim = Nadam(graph_sort)
    update_lr = Auto_update_lr(lr=learning_rate, alpha=0.1, patiences=500, print_=True)
    for e in range(epoch):
        loss = 0
        for b in range(steps_per_epoch):
            X_batch, y_batch = resample(x_, y_, n_samples=batch_size)
            # print(x_.shape,X_batch.shape)
            mlp.x.value = X_batch  # 在这更新值
            mlp.y.value = y_batch[:, None]
            # print(X_batch.shape)
            run_steps(graph_sort, monitor=False)

            optim.update(learning_rate=learning_rate)
            Visual_gradient(model)
            Grad_Clipping_Disappearance(model, 5)

            loss += mlp.MSE_loss.value
        update_lr.updata(loss/steps_per_epoch)
        print("epoch:{}/{},loss:{:.6f}".format(e,epoch,loss / steps_per_epoch))
        losses.append(loss / steps_per_epoch)
        if loss / steps_per_epoch < loss_min:
            print('loss is {:.6f}, is decreasing!! save moddel'.format(loss / steps_per_epoch))
            save_model("model/mlp.xhp", model)
            loss_min = loss / steps_per_epoch
    print('The min loss:',loss_min)
    # print("loss:{}".format(np.mean(losses)))
    plt.plot(losses)
    plt.savefig("image/many_vectoy.png")
    plt.show()


train(mlp)

load_model("model/mlp.xhp",mlp)

def predict(x_rm, graph,model):
    model.x.value = x_rm
    run_steps(graph, monitor=False, train=False,valid=False)

    return model.y_pre.value*std_y + mean_y
graph_sort = toplogical_sort(mlp.feed_dict)
print("预测值：",predict(x_[17:50],graph_sort,mlp),"真实值：",y_[17:50]*std_y + mean_y)

