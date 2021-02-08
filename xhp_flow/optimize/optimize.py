import random
from collections import defaultdict
from xhp_flow.nn.node import  Placeholder
import os,zipfile
from glob import glob
import numpy as np
import shutil

"""
使用拓扑排序找到网络节点的前向计算顺序（反向传播反过来就行）
"""
def toplogical(graph):

    sorted_graph_nodes = []

    while graph:
        all_nodes_have_inputs = set()
        all_nodes_have_outputs = set()

        for have_output_node, have_inputs in graph.items():
            """
            包括只有输出的节点 和既有输入又有输出的点
            """
            all_nodes_have_outputs.add(have_output_node)
            """
            有输入的点：包括既有输入和输出的点 和只有输入的点（末尾终点）
            """
            all_nodes_have_inputs |= set(have_inputs)
            """
            减去之后留下只有输出的节点
            """
        need_removed_nodes = all_nodes_have_outputs - all_nodes_have_inputs

        if need_removed_nodes:
            """
            随机删去一个节点
            """
            node = random.choice(list(need_removed_nodes))
            visited_next = [node]
            """
            当最后删到只留一个有输出的节点
            的时候，那么需要把这个节点对应的输出节点也加上，否则会漏掉这个点
            """
            if len(graph) == 1: visited_next += graph[node]

            graph.pop(node)
            sorted_graph_nodes += visited_next

            for _, links in graph.items():
                """
                如果删除的节点在别的节点的连接关系内，那么把他从连接关系里删除
                """
                if node in links: links.remove(node)
        else:
            break

    return sorted_graph_nodes
"""
得到的网络连接关系示例如下：
# 网络连接关系：
defaultdict(list,
            {Node:b1: [Node:linear1],
             Node:b2: [Node:linear2],
             Node:linear1: [Node:sigmoid],
             Node:linear2: [Node:MSE],
             Node:sigmoid: [Node:linear2],
             Node:w1: [Node:linear1],
             Node:w2: [Node:linear2],
             Node:x: [Node:linear1],
             Node:y: [Node:MSE]})
#最后得到的排序结果：
[Node:b2, Node:y, Node:w2, Node:x, Node:w1, Node:b1, Node:linear1, Node:sigmoid, Node:linear2, Node:MSE]
"""
"""
根据feed_dict和网络节点的初始化结果,建立网络的连接关系
"""
from collections import defaultdict


def convert_feed_dict_graph(feed_dict):
    computing_graph = defaultdict(list)

    nodes = [n for n in feed_dict]

    while nodes:
        """
        移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
        """
        n = nodes.pop(0)

        if isinstance(n, Placeholder):
            n.value = feed_dict[n]
        if n in computing_graph: continue

        for m in n.outputs:
            """
            建立好网络连接关系
            """
            computing_graph[n].append(m)
            nodes.append(m)

    return computing_graph


"""
根据网络的连接关系，进行拓扑排序。
"""


def toplogical_sort(feed_dict):
    graph = convert_feed_dict_graph(feed_dict)

    return toplogical(graph)


import copy


def forward(graph, monitor=False, valid=True):
    for node in graph if valid else graph[:-1]:
        if monitor: print('forward:{}'.format(node))
        node.forward()


def backward(graph, monitor=False):
    for node in graph[::-1]:
        if monitor: print('backward:{}'.format(node))
        node.backward()


"""
进行前向和反向传播计算
"""
def run_steps(graph_topological_sort_order, train=True, valid=True, monitor=False, ):
    if train:
        forward(graph_topological_sort_order, monitor)
        backward(graph_topological_sort_order, monitor)
    else:
        forward(graph_topological_sort_order, monitor, valid)



class optimize():
    def __init__(self,graph, learning_rate=1e-2):
        self.graph = graph
        self.learning_rate = learning_rate
        for node in self.graph:
            if node.is_trainable:
                node.value = node.value.reshape((node.value.shape[0], -1))
                node.gradients[node] = node.gradients[node].reshape((node.gradients[node].shape[0], -1))
                node.value += -1 * node.gradients[node] * self.learning_rate


class Auto_update_lr():
    def __init__(self,lr,alpha = 0.1,patiences = 200,print_ = False):


        self.patiences = patiences
        self.print = print_
        self.alpha = alpha
        self.lr = lr
        self.loss_min = np.inf

    def updata(self,loss):

        self.loss = loss

        if self.loss < self.loss_min:
            self.loss_min = self.loss
            self.patience = 0

        if self.loss > self.loss_min:
            self.patience += 1
        else:
            self.patience = 0

        """
        只要连续paiences次损失不下降就更新梯度
        """
        if self.patience > self.patiences:
            self.lr *= self.alpha
            self.patience = 0

            if self.print:
                print('lr:', self.lr)





import os,zipfile
from glob import glob
"""
压缩文件成zip文件
"""
def compress(zip_file, input_dir):
    f_zip = zipfile.ZipFile(zip_file, 'w')
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            """
            获取文件相对路径，在压缩包内建立相同的目录结构
            """
            abs_path = os.path.join(os.path.join(root, f))
            rel_path = os.path.relpath(abs_path, os.path.dirname(input_dir))

            f_zip.write(abs_path, rel_path, zipfile.ZIP_STORED)

"""
解压zip文件
"""
def extract(zip_file,pwd=None):
    if pwd:
        pwd = pwd.encode()
    f_zip = zipfile.ZipFile(zip_file, 'r')
    """
    解压所有文件到指定目录
    """
    f_zip.extractall(zip_file.split(".")[0],pwd=pwd)


"""
保存模型参数
"""
import shutil
def save_model(save_path,model):

    if len(save_path.split('/')) > 1:
        save_path_txt =  save_path.split('/')[-1]
        save_path_ = '.'+ save_path_txt.split('.')[0]
    else:
        save_path_ = '.'+ save_path.split('.')[0]

    save_path = save_path.split('.')[0]
    """
    如果文件夹不存在，创建一个新的
    """
    if not os.path.exists(save_path_):
        os.mkdir(save_path_)
    for name, node in vars(model).items():
        if isinstance(node, Placeholder):
            if node.is_trainable:
                np.savetxt("{}/{}.txt".format(save_path_,name),node.value)
    compress(os.getcwd() + '/{}.xhp'.format(save_path), save_path_, )
    """
    删除文件
    """
    shutil.rmtree(save_path_)


"""
加载模型参数
"""
def load_model(load_path,model):
    """"
    vars(model).items()，分别返回类的名字和值
    例如：
    class MLP():
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
    model = MLP()
    for name, node in vars(model).items():
        print(name,node)
    返回：
    a 1
    b 2
    c 3
    """
    extract(load_path)
    load_path = load_path.split(".")[0]
    if len(load_path.split('/')) > 1:
        load_path_txt = load_path.split('/')[-1]
    else:
        load_path_txt = load_path
    model_parameter_path = np.array(glob(load_path + '/.'+load_path_txt + "/*"))
    model_parameter = set()
    save_model_parameter = set()
    for name, node in vars(model).items():
        if isinstance(node, Placeholder):
            if node.is_trainable:
                model_parameter.add(name)
                for path in model_parameter_path:
                    save_model_parameter.add(path.split(".")[1].split("/")[1])
                    if path.split(".")[1].split("/")[1] == name:
                        assert model.feed_dict[node].size == np.loadtxt(path).size,\
                        'The trainable parameter shape of mdoel is not match,{} and {} is not same!'.\
                        format(model.feed_dict[node].shape,np.loadtxt(path).shape)
                        node.value = np.loadtxt(path)
                        model.feed_dict[node] = node.value
    assert model_parameter == save_model_parameter,\
    "\nThe parameters of the model " \
    "being loaded do not match the parameters in the current model!! {} is not common.\n" \
    "Please check wheter the trainabel parameters of the model are correct!"\
    .format(list(model_parameter-save_model_parameter)+list(save_model_parameter-model_parameter))

    shutil.rmtree(load_path)

def Visual_gradient(model,visual_gradient = True,visual_shape = False):

    for name,node in vars(model).items():
        if isinstance(node, Placeholder):
            if node.is_trainable:
                for key,values in node.gradients.items():
                    if visual_shape:
                        file_shape = open('grdient_shape.txt', 'w+')
                        print("node:", key, " ", "value:", values.shape,file=file_shape)
                    if visual_gradient:
                        file_value = open('gradient.txt', 'w+')
                        print("node:", key, " ", "value:", values,file=file_value)


def Grad_Clipping_Disappearance(model,clip_min = 1e-10,clip_max=5):
    eps = np.finfo(float).eps
    for name, node, in vars(model).items():
        if isinstance(node, Placeholder):
            if node.is_trainable:
                norm = 0
                for key, values in node.gradients.items():
                    norm += np.sum(values**2)
                norm = np.sqrt(norm)
                if norm > clip_max:
                    for key, values in node.gradients.items():
                        values *= (clip_max/norm)
                if norm < clip_min:
                    for key, values in node.gradients.items():
                        values *= (clip_min/(norm+eps))

