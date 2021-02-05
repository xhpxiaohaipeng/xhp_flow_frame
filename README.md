##ABOUT
This is a mini deep learning framework created by xiaohaipeng @tianjin and changsha.
The framework is built by numpy library,and initially realized the construction of MLP and LSTM networks,which can be used to solve regression and classification problems


##Features
+ Define Dynamic Computing Graph
+ Define Neural Network Operator
+ Define Placeholder,Linear,Sigmoid,Relu,Leakrelu,Elu,Tanh,LSTM,MSE,EntropyCrossLossWithSoftmax.
+ Auto-Diff Computing
+ Auto-Feedforward and Backward
+ Save and Load model
+ Examples for using MLP,LSTM in Regression and classification problem
+ A algorithm is designed to prevent gradient explosion and gradient disappearance

#How to USE
1.You can install xhp_flow by pip3 install xhp_flow and run the python file in examples folder.
2.Also you can run python flie directly.

#Remain
+ LSTM Classification effect is not good enough,I find the the cause is gradient disappearance(the gradient of LSTM is close to zero),so I write a function Visual_gradient() to see the gradient.So far,I have not found a good solution,I just relieved it a litle bit,but the classification effect is still not good.
+ CNN
+ SGD,Adam....


# Connection
wechat:18651271973
Mail:307153814@qq.com



