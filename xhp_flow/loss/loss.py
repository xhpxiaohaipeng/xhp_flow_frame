import numpy as np

from xhp_flow.nn.node import Node


class MSE(Node):

    def __init__(self, y_pre, y, name='MSE', is_trainable=False):

        Node.__init__(self, [y_pre, y], name=name, is_trainable=is_trainable)
        self.y_pre, self.y = self.inputs[0], self.inputs[1]


    def forward(self):
        y = self.y.value
        y_pre = self.y_pre.value

        assert y.shape == y_pre.shape,'actual y shape:{},get y shape:{}!!'.format(y.shape,y_pre.shape)


        self.batch_size = self.inputs[0].value.shape[0]
        self.diff = y - y_pre

        if len(y.shape) == 2:
            self.value = np.mean(self.diff ** 2)
        elif len(y_pre.shape) == 3:
            self.value = np.mean(self.diff**2,axis=0,keepdims=False)
            self.value = np.mean(self.value,axis=0,keepdims=True)[0][0]
            self.diff = np.mean(self.diff,axis=1,keepdims=False)

    def backward(self):

        self.gradients[self.y] = (2 / self.batch_size) * self.diff
        self.gradients[self.y_pre] = (-2 / self.batch_size) * self.diff


class EntropyCrossLossWithSoftmax(Node):
    def __init__(self, y_pre, y_label,alpha=0.01, name='EntropyLossWithSoftmax', is_trainable=False):
        Node.__init__(self, [y_pre, y_label], name=name, is_trainable=is_trainable)
        self.y_pre = self.inputs[0]
        self.y_label = self.inputs[1]
        """
        prevent the np.exp(x) to nan
        """
        self.softmax_alpha = alpha

        assert alpha <= 1,'alpha should not be biger than 1'

    def softmax(self, L):

        EXEP = np.exp(L)
        SUM = np.sum(EXEP, axis=1)
        for i in range(len(L)):
            for k in range(len(L[i])):
                L[i][k] = EXEP[i][k] / SUM[i]
        return L

    def cross_entropy_error(self,y_, y):

        if y_.ndim == 1:
            y = y.reshape(1, y.size)
            y_ = y_.reshape(1, y_.size)

        if y.size == y_.size:
            y = y.argmax(axis=1)

        self.result = y_.argmax(axis=1)
        self.y_bool = self.result == y
        self.accuracy = np.mean(self.y_bool)

        batch_size = y.shape[0]
        """
        prevent taking the log of 0
        """
        eps = np.finfo(float).eps
        """
        #y_[np.arange(batch_size), y]代表选择每一行
        代表的所有类别里面的正确类别此时的概率,然后所有
        概率的对数值的相反数相加就是交叉熵损失
        """
        return -np.mean(np.log(y_[np.arange(batch_size), y] + eps))

    def forward(self):

        self.y = self.y_label.value
        """
        self.softmax_alpha 防止np.exp(x)的值倾向于无穷大
        """
        self.y_ = self.softmax(self.softmax_alpha*self.y_pre.value)
        self.value = self.cross_entropy_error(self.y_, self.y)

    def backward(self):

        batch_size = self.y.shape[0]
        if self.y.size == self.y_.size:
            dx = (self.y_ - self.y) / batch_size
        else:
            """
            sottmax(x)函数对x的导数为sottmax(x) - softmax(x)*sottmax(x)
            """
            dx = self.y_.copy()
            dx[np.arange(batch_size), self.y] -= 1

        self.gradients[self.y_pre] = self.softmax_alpha*dx
        self.gradients[self.y_label] = -self.softmax_alpha*dx

        if self.gradients[self.y_pre].ndim == 3:
            self.gradients[self.y_pre] = np.mean(self.gradients[self.y_pre],axis=1,keepdims=False)
            self.gradients[self.y_label] = np.mean(self.gradients[self.y_label],axis=1,keepdims=False)