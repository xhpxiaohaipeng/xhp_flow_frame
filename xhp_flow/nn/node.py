import numpy as np
import random


class Node:
    """
    我们把这个Node类作为这个神经网络的基础模块
    """

    def __init__(self, inputs=[], name=None, is_trainable=False):
        """

        :param inputs:输入节点
        :param name:节点名字
        :param is_trainable: 这个节点是否可训练
        """
        """
        这个节点的输入，输入的是Node组成的列表
        """
        self.inputs = inputs
        """
        这个节点的输出节点
        """
        self.outputs = []
        self.name = name
        self.is_trainable = is_trainable
        for n in self.inputs:
            """
            这个节点正好对应了这个输人的输出节点，从而建立了连接关系
            """
            n.outputs.append(self)

        """
        每个节点必定对应有一个值
        """
        self.value = None

        """
        每个节点对下个节点的梯度
        """
        self.gradients = {}

    def forward(self):
        """
        先预留一个方法接口不实现，在其子类中实现,
        且要求其子类一定要实现，不实现的时话会报错。
        """
        raise NotImplemented

    def backward(self):
        raise NotImplemented

    def __repr__(self):
        return "{}".format(self.name)

class Placeholder(Node):
    """
    作为x,weights和bias这类需要赋初始值和更新值的类
    """

    def __init__(self, name='Placeholder', is_trainable=True):

        Node.__init__(self, name=name, is_trainable=is_trainable)

    def forward(self, value=None):

        if value is not None: self.value = value

    def backward(self):

        if len(self.value.shape) == 3:
            self.value = np.mean(self.value,axis=1,keepdims=False)
        self.gradients[self] = np.zeros_like(self.value).reshape((self.value.shape[0], -1))
        for n in self.outputs:
            self.gradients[self] = np.add(self.gradients[self],n.gradients[self].reshape((n.gradients[self].shape[0], -1)))  # 没有输入。

from xhp_flow.loss.loss import EntropyCrossLossWithSoftmax

class Linear(Node):

    def __init__(self, x=None, weight=None, bias=None, name='Linear', is_trainable=False):

        Node.__init__(self, [x, weight, bias], name=name, is_trainable=is_trainable)

    def forward(self):

        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]

        self.value = np.dot(x.value, k.value) + b.value.squeeze()

        if self.value.ndim == 3:
            for n in self.outputs:
                if isinstance(n,EntropyCrossLossWithSoftmax):
                    self.value = self.value[:,-1]


    def backward(self):

        k, x, b = self.inputs[1], self.inputs[0], self.inputs[2]

        self.gradients[k] = np.zeros_like(k.value)
        self.gradients[b] = np.zeros_like(b.value).reshape((len(np.zeros_like(b.value))))
        self.gradients[x] = np.zeros_like(x.value)

        for n in self.outputs:
            """
            输出节点对这个节点的偏导，self：指的是本身这个节点
            """
            gradients_from_loss_to_self = n.gradients[self]
            if len(x.value.shape) == 2:
                self.gradients[k] += np.dot(gradients_from_loss_to_self.T, x.value).T
                self.gradients[b] += np.mean(gradients_from_loss_to_self, axis=0, keepdims=False).reshape((len(np.zeros_like(b.value))))
                self.gradients[x] += np.dot(gradients_from_loss_to_self, k.value.T)
            elif len(x.value.shape)==3:
                x.value = np.mean(x.value,axis=1,keepdims=False)
                self.gradients[x] = np.mean(self.gradients[x],axis=1,keepdims=False)

                self.gradients[k] += np.dot(gradients_from_loss_to_self.T, x.value).T
                self.gradients[b] += np.mean(gradients_from_loss_to_self, axis=0, keepdims=False).reshape((len(np.zeros_like(b.value))))
                self.gradients[x] += np.dot(gradients_from_loss_to_self, k.value.T)



class Sigmoid(Node):

    def __init__(self, x, name='Sigmoid', is_trainable=False):

        Node.__init__(self, [x], name=name, is_trainable=is_trainable)
        self.x = self.inputs[0]

    def _Sigmoid(self, x):

        return 1. / (1 + np.exp(-1 * x))

    def forward(self):

        self.value = self._Sigmoid(self.x.value)

    def partial(self):

        return self._Sigmoid(self.x.value) * (1 - self._Sigmoid(self.x.value))

    def backward(self):

        self.gradients[self.x] = np.zeros_like(self.value)
        for n in self.outputs:
            if len(self.gradients[self.x].shape) == 3:
                self.gradients[self.x] = np.mean(self.gradients[self.x], axis=1, keepdims=False)
            if len(self.x.value.shape) == 3:
                self.x.value = np.mean(self.x.value, axis=1, keepdims=False)
                """
                输出节点对这个节点的偏导，self：指的是本身这个节点
                """
            gradients_from_loss_to_self = n.gradients[self]
            self.gradients[self.x] += gradients_from_loss_to_self * self.partial()


class ReLu(Node):

    def __init__(self, x, name='Relu', is_trainable=False):

        Node.__init__(self, [x], name=name, is_trainable=is_trainable)
        self.x = self.inputs[0]

    def forward(self):

        self.value = self.x.value * (self.x.value > 0)

    def backward(self):

        self.gradients[self.x] = np.zeros_like(self.value)
        for n in self.outputs:
            """
            输出节点对这个节点的偏导，self：指的是本身这个节点
            """
            gradients_from_loss_to_self = n.gradients[self]
            if len(self.gradients[self.x].shape) == 3:
                self.gradients[self.x] = np.mean(self.gradients[self.x], axis=1, keepdims=False)
            if len(self.x.value.shape) == 3:
                self.x.value = np.mean(self.x.value, axis=1, keepdims=False)
            self.gradients[self.x] += gradients_from_loss_to_self * (self.x.value > 0)




class Leakrelu(Node):

    def __init__(self,x,alpha=0.01,name='Leakrelu',is_trainable=False):
        Node.__init__(self,[x],name=name,is_trainable=is_trainable)
        self.alpha = alpha
        self.x = self.inputs[0]

        assert 0 <= alpha <= 1,'alpha should be biger than 0 and smaller than 1,[0,1]'

    """
    使用实数替代bool矩阵内的bool值
    """
    def replace_bool_value(self,x,new_True_value, new_False_value):
        y = np.zeros(x.shape)
        if len(x.shape) == 3:
            for thrid_dimension in range(len(x)):
                for second_dimension in range(len(x[thrid_dimension])):
                    for value in range(len(x[thrid_dimension][second_dimension])):
                        if x[thrid_dimension][second_dimension][value] == True:
                            y[thrid_dimension][second_dimension][value] = new_True_value
                        if x[thrid_dimension][second_dimension][value] == False:
                            y[thrid_dimension][second_dimension][value] = new_False_value

        if len(x.shape) == 2:
            for second_dimension in range(len(x)):
                for value in range(len(x[second_dimension])):
                    if x[second_dimension][value] == True:
                        y[second_dimension][value] = new_True_value
                    if x[second_dimension][value] == False:
                        y[second_dimension][value] = new_False_value

        return y

    def forward(self):
        bool_value = self.x.value > 0
        bool_value_replaced = self.replace_bool_value(bool_value,1,self.alpha)
        self.value = self.x.value * bool_value_replaced

    def backward(self):

        self.gradients[self.x] = np.zeros_like(self.value)
        for n in self.outputs:
            """
            输出节点对这个节点的偏导，self：指的是本身这个节点
            """
            gradients_from_loss_to_self = n.gradients[self]
            if len(self.gradients[self.x].shape) == 3:
                self.gradients[self.x] = np.mean(self.gradients[self.x],axis=1,keepdims=False)
            if len(self.x.value.shape) == 3:
                self.x.value = np.mean(self.x.value,axis=1,keepdims=False)

            bool_value = self.x.value > 0
            bool_value_replaced = self.replace_bool_value(bool_value, 1, self.alpha)
            self.gradients[self.x] += gradients_from_loss_to_self*bool_value_replaced

class Elu(Node):

    def __init__(self,x,alpha = 0.1,name='Elu',is_trainable = False):
        Node.__init__(self,[x],name=name,is_trainable=is_trainable)

        self.x = self.inputs[0]
        self.alpha = alpha

        assert 0 <= alpha <= 1,'alpha should be biger than 0 and smaller than 1,[0,1]'

    """
    计算函数值，使用替代的方式分别计算大于0的值和小于0的值。
    """
    def calculate_value(self,x, alpha):

        if len(x.shape) == 3:
            for thrid_dimension in range(len(x)):
                for second_dimension in range(len(x[thrid_dimension])):
                    for value in range(len(x[thrid_dimension][second_dimension])):
                        if x[thrid_dimension][second_dimension][value] < 0:
                            x[thrid_dimension][second_dimension][value] = alpha * \
                            (np.exp(x[thrid_dimension][second_dimension][value]) - 1)

        if len(x.shape) == 2:
            for second_dimension in range(len(x)):
                for value in range(len(x[second_dimension])):
                    if x[second_dimension][value] < 0:
                        x[second_dimension][value] = alpha * (np.exp(x[second_dimension][value]) - 1)
        return x

    def forward(self):

        self.value = self.calculate_value(self.x.value,self.alpha)

    """
    使用替代的方式分别计算大于0和小于0时的导数值
    """
    def calculate_diff_value(self,x,alpha):

        y = np.zeros(x.shape)
        if len(x.shape) == 3:
            for thrid_dimension in range(len(x)):
                for second_dimension in range(len(x[thrid_dimension])):
                    for value in range(len(x[thrid_dimension][second_dimension])):
                        if x[thrid_dimension][second_dimension][value] == True:
                            y[thrid_dimension][second_dimension][value] = 1
                        if x[thrid_dimension][second_dimension][value] == False:
                            y[thrid_dimension][second_dimension][value] = alpha*\
                              np.exp(x[thrid_dimension][second_dimension][value])

        if len(x.shape) == 2:
            for second_dimension in range(len(x)):
                for value in range(len(x[second_dimension])):
                    if x[second_dimension][value] == True:
                        y[second_dimension][value] = 1
                    if x[second_dimension][value] == False:
                        y[second_dimension][value] = alpha*np.exp(x[second_dimension][value])

        return y

    def backward(self):

        self.gradients[self.x] = np.zeros_like(self.value)
        for n in self.outputs:
            """
            输出节点对这个节点的偏导，self：指的是本身这个节点
            """
            gradients_from_loss_to_self = n.gradients[self]
            if len(self.gradients[self.x].shape) == 3:
                self.gradients[self.x] = np.mean(self.gradients[self.x],axis=1,keepdims=False)
            if len(self.x.value.shape) == 3:
                self.x.value = np.mean(self.x.value,axis=1,keepdims=False)

            bool_value = self.x.value > 0
            diff_value = self.calculate_diff_value(bool_value,self.alpha)
            self.gradients[self.x] += gradients_from_loss_to_self*diff_value

class Tanh(Node):

    def __init__(self,x,name='Tanh',is_trainable=False):
        Node.__init__(self,[x],name=name,is_trainable=is_trainable)
        self.x = self.inputs[0]

    def _Tanh(self,x):

        return (np.exp(-x) - np.exp(x)) / (np.exp(-x) + np.exp(x))

    def forward(self):

        self.value = self._Tanh(self.x.value)

    def backward(self):

        self.gradients[self.x] = np.zeros_like(self.value)
        for n in self.outputs:
            """
            输出节点对这个节点的偏导，self：指的是本身这个节点
            """
            gradients_from_loss_to_self = n.gradients[self]
            if len(self.gradients[self.x].shape) == 3:
                self.gradients[self.x] = np.mean(self.gradients[self.x],axis=1,keepdims=False)
            if len(self.x.value.shape) == 3:
                self.x.value = np.mean(self.x.value,axis=1,keepdims=False)
            self.gradients[self.x] += gradients_from_loss_to_self * (1.-self._Tanh(self.x.value)**2)

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LSTMcell():

    def __init__(self, x, wf, wi, wc, wo, bf, bi, bc, bo, input_size, hidden_size, s_prev=None, h_prev=None):
        super(LSTMcell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        """
        判断输入变量的特征大小是否正确
        """
        assert x.shape[2] == input_size ,'input expect size:{},but get size:{}!!'.format(input_size,x.shape[2])
        """
        初始化计算变量
        """
        self.f = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.i = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.c = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.o = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.s = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.h = np.zeros((x.shape[0], x.shape[1], hidden_size))
        self.xc= np.zeros((x.shape[0], x.shape[1], hidden_size+input_size))

        self.wf = wf
        self.wi = wi
        self.wc = wc
        self.wo = wo
        """
        统一将偏置变量设为一维变量
        """
        self.bf = bf.squeeze()
        self.bi = bi.squeeze()
        self.bc = bc.squeeze()
        self.bo = bo.squeeze()


        self.h_prev = h_prev
        self.s_prev = s_prev

        self.gradients = {}
        self.x = x

    def sigmoid(self,x):

        return 1. / (1 + np.exp(-x))

    def forward(self):

        """
        如果输入的第一个LSTM细胞，初始化隐藏状态向量和细胞状态向量
        """
        if self.s_prev is None:
            self.s_prev = np.zeros((self.x.shape[0], self.x.shape[1], self.hidden_size))
        if self.h_prev is None:
            self.h_prev = np.zeros((self.x.shape[0], self.x.shape[1], self.hidden_size))

        """
        LSTM细胞前向计算
        """
        self.xc = np.concatenate((self.x,self.h_prev),axis=2)
        self.f = self.sigmoid(np.matmul(self.xc,self.wf) + self.bf)
        self.i = self.sigmoid(np.dot(self.xc,self.wi) + self.bi)
        self.c = np.tanh(np.dot(self.xc,self.wc) + self.bc)
        self.s = self.c*self.i + self.s_prev*self.f
        self.o = self.sigmoid(np.dot(self.xc,self.wo) + self.bo)
        self.h = np.tanh(self.s)*self.o



    def diff_sigmoid(self, x):

        return (1. - x) * x

    def diff_tanh(self, x):

        return 1. - x ** 2

    def backward(self):

        """
        LSTM细胞反向梯度计算,基于乘法运算求导和链式法则求导
        """
        """
        公共的梯度
        """
        ds = self.diff_tanh(np.tanh(self.s))
        """
        各梯度
        """
        df = self.s_prev * self.diff_sigmoid(self.f) * self.o * ds
        di = self.c * self.diff_sigmoid(self.i) * self.o * ds
        dc = self.i * self.diff_tanh(self.c) * self.o * ds
        do = np.tanh(self.c) * self.diff_sigmoid(self.o)

        dxc = self.o * self.diff_tanh(self.c)*(self.s_prev * self.diff_sigmoid(self.f) * self.wf + \
                                      self.i*self.diff_tanh(self.c)*self.wc + self.c * \
                                      self.diff_sigmoid(self.i)*self.wi ) + np.tanh(self.s) * \
                                      self.diff_sigmoid(self.o)*self.wo

        ds_prev = self.o * ds * self.f
        """
        取一个batch_size梯度的平均值作为最后的梯度值
        """
        self.xc = np.concatenate((self.x, self.h_prev), axis=2)
        self.xc = self.xc.transpose(0, 2, 1)
        self.gradients['wf'] = np.mean(np.multiply(self.xc,df),axis=0,keepdims=False)
        self.gradients['wi'] = np.mean(np.multiply(self.xc,di),axis=0,keepdims=False)
        self.gradients['wc'] = np.mean(np.multiply(self.xc,dc),axis=0,keepdims=False)
        self.gradients['wo'] = np.mean(np.multiply(self.xc,do),axis=0,keepdims=False)

        self.gradients['bf'] = np.mean(df,axis=0,keepdims=False)
        self.gradients['bi'] = np.mean(di,axis=0,keepdims=False)
        self.gradients['bc'] = np.mean(dc,axis=0,keepdims=False)
        self.gradients['bo'] = np.mean(do,axis=0,keepdims=False)

        self.gradients['xc'] = np.mean(dxc,axis=0,keepdims=False)
        self.gradients['x'] = self.gradients['xc'][:self.x.shape[2]]
        self.gradients['h_prev'] = self.gradients['xc'][self.x.shape[2]:]
        self.gradients['s_prev'] = ds_prev

class LSTM(Node):

    def __init__(self, input_x, wf, wi, wc, wo, bf, bi, bc, bo, input_size, hidden_size,
                 h_prev = None,s_prev = None,name='LSTM', is_trainable=False):
        Node.__init__(self, [input_x, wf, wi, wc, wo, bf, bi, bc, bo], \
                      name=name, is_trainable=is_trainable)

        """
        初始化变量值
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_x = input_x

        self.wf = wf
        self.wi = wi
        self.wc = wc
        self.wo = wo

        self.bf = bf
        self.bi = bi
        self.bc = bc
        self.bo = bo
        """
        用来传递初始细胞状态和隐藏状态
        """
        self.h_ = h_prev
        self.s_ = s_prev
    def forward(self):

        assert self.input_x.value.ndim == 3, 'expect 3 dim input,but get {} dim input!!'.format(self.input_x.ndim)

        """
        定义存储LSTM细胞的列表容器，不能在init里定义，
        否则所有输入的LSTM细胞都会被保存在列表里，
        不会随着输入的更新而清空并更新列表
        """
        self.lstm_node_list = []
        """
        初始细胞状态和隐藏状态不能在init里定义，
        否则上一个输入的最后一个LSTM细胞的细胞
        状态和隐藏状态输出会被记住，并用于下一个
        输入的初始细胞状态和隐藏状态输入，这会造成无法训练。
        """
        self.s_prev = self.s_
        self.h_prev = self.h_
        """
        按照输入变量依次填入LSTM细胞
        """
        for i in range(self.input_x.value.shape[1]):
            """
            把前一个LSTM细胞输出的隐藏状态和细胞状态
            传递给下一个LSTM细胞
            """
            if len(self.lstm_node_list) > 0:
                self.s_prev = self.lstm_node_list[i-1].s
                self.h_prev = self.lstm_node_list[i-1].h
            """
            按照输入数据的顺序依次填入LSTM细胞
            """
            self.lstm_node_list.append(LSTMcell(self.input_x.value[:,i,:][:,None,:], self.wf.value, self.wi.value, self.wc.value, self.wo.value, \
                         self.bf.value, self.bi.value, self.bc.value, self.bo.value, self.input_size, self.hidden_size, self.s_prev, self.h_prev))
            """
            LSTM细胞进行前向计算
            """
            self.lstm_node_list[i].forward()
            """
            合并LSTM细胞的输出结果作为LSTM的输出
            """
            if i == 0:
                self.value = self.lstm_node_list[i].h
            else:
                self.value = np.concatenate((self.value, self.lstm_node_list[i].h), axis=1)

    def backward(self):
        """
        初始化各个梯度值为0
        """
        self.gradients[self.wf] = np.zeros_like(self.wf.value)
        self.gradients[self.wi] = np.zeros_like(self.wi.value)
        self.gradients[self.wc] = np.zeros_like(self.wc.value)
        self.gradients[self.wo] = np.zeros_like(self.wo.value)

        self.gradients[self.bf] = np.zeros_like(self.bf.value).squeeze()
        self.gradients[self.bi] = np.zeros_like(self.bi.value).squeeze()
        self.gradients[self.bc] = np.zeros_like(self.bc.value).squeeze()
        self.gradients[self.bo] = np.zeros_like(self.bo.value).squeeze()
        """
        实际上与LSTM网络连接的MLP，相当于只与最后一个LSTM细胞相连，
        因为最后的梯度更新都会流向最后一个LSTM细胞， 相当于梯度更新
        只与最后一个LSTM细胞有关
        """
        self.gradients[self.input_x] = np.zeros_like(self.input_x.value[:,0,:])

        """
        按照倒序进行梯度计算
        将节点反转过来求梯度
        """
        for backward_node_index in range(len(self.lstm_node_list[::-1])):
            self.lstm_node_list[backward_node_index].backward()
            """
            最后一个LSTM细胞的梯度不涉及到基于时间序列的链式法则求解梯度
            """
            if backward_node_index == 0:

                gradients_wf = self.lstm_node_list[backward_node_index].gradients['wf']
                gradients_wi = self.lstm_node_list[backward_node_index].gradients['wi']
                gradients_wc = self.lstm_node_list[backward_node_index].gradients['wc']
                gradients_wo = self.lstm_node_list[backward_node_index].gradients['wo']

                gradients_bf = self.lstm_node_list[backward_node_index].gradients['bf']
                gradients_bi = self.lstm_node_list[backward_node_index].gradients['bi']
                gradients_bc = self.lstm_node_list[backward_node_index].gradients['bc']
                gradients_bo = self.lstm_node_list[backward_node_index].gradients['bo']

                gradients_h = self.lstm_node_list[backward_node_index].gradients['h_prev']
                gradients_x = self.lstm_node_list[backward_node_index].gradients['x']

            else:
                """
                基于时间的梯度计算法则计算梯度（BPTT,其实就是链式法则）
                """
                h_grdient_index = 1
                while h_grdient_index != backward_node_index:
                    """
                    #0,1,2,...i-1  各LSTM细胞之间的h梯度相乘，按照先后顺序有不同数量的h梯度因数
                    """
                    gradients_h *= self.lstm_node_list[h_grdient_index].gradients['h_prev']
                    h_grdient_index += 1
                """ 
                梯度相加原则
                #0,1,2,3,....i 所有节点的梯度相加
                """
                gradients_wf += np.dot(self.lstm_node_list[backward_node_index].gradients['wf'], gradients_h)
                gradients_wi += np.dot(self.lstm_node_list[backward_node_index].gradients['wi'], gradients_h)
                gradients_wc += np.dot(self.lstm_node_list[backward_node_index].gradients['wc'], gradients_h)
                gradients_wo += np.dot(self.lstm_node_list[backward_node_index].gradients['wo'], gradients_h)

                gradients_bf += np.dot(self.lstm_node_list[backward_node_index].gradients['bf'], gradients_h)
                gradients_bi += np.dot(self.lstm_node_list[backward_node_index].gradients['bi'], gradients_h)
                gradients_bc += np.dot(self.lstm_node_list[backward_node_index].gradients['bc'], gradients_h)
                gradients_bo += np.dot(self.lstm_node_list[backward_node_index].gradients['bo'], gradients_h)

                gradients_x += np.dot(self.lstm_node_list[backward_node_index].gradients['x'], gradients_h)

        gradients_bf = gradients_bf.squeeze()
        gradients_bi = gradients_bi.squeeze()
        gradients_bc = gradients_bc.squeeze()
        gradients_bo = gradients_bo.squeeze()

        for n in self.outputs:

            gradients_from_loss_to_self = n.gradients[self]
            gradients_from_loss_to_self = np.mean(gradients_from_loss_to_self, axis=0, keepdims=False)
            """
            对于输入x的梯度计算需保留所有LSTM细胞的梯度计算，为LSTM的输入节点的梯度计算做准备。
            """
            self.gradients[self.input_x] += np.dot(gradients_from_loss_to_self,gradients_x.T)

            """
            #取一个batch的平均值和所有node的平均值
            """
            gradients_from_loss_to_self = np.mean(gradients_from_loss_to_self, axis=0, keepdims=True)
            self.gradients[self.wf] += gradients_from_loss_to_self*gradients_wf
            self.gradients[self.wi] += gradients_from_loss_to_self*gradients_wi
            self.gradients[self.wc] += gradients_from_loss_to_self*gradients_wc
            self.gradients[self.wo] += gradients_from_loss_to_self*gradients_wo
            self.gradients[self.bf] += gradients_from_loss_to_self*gradients_bf
            self.gradients[self.bi] += gradients_from_loss_to_self*gradients_bi
            self.gradients[self.bc] += gradients_from_loss_to_self*gradients_bc
            self.gradients[self.bo] += gradients_from_loss_to_self*gradients_bo

