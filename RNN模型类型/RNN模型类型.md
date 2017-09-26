<h1 align = "center">RNN模型的类型</h1>

## 前言
**
深度学习中CNN多用于空间域相关信息的特征提取，而时间序列的特征提取如自然语言，语音序列，视频动作序列等，需要RNN模型的支持，不同的RNN模型可以实现不同的功能，这里浅谈一些RNN模型。
**
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`


----


# RNN模型的基本结构

　　很多分类问题，如CIFAR-10，ImageNet等的确是空间域上的分类，而不涉及到时间序列的问题，但是显然也有很多重要问题涉及到了时间序列的识别，需要考虑序列上下文才能作出合适的分类。对于这类问题，传统的CNN已经不能满足需求了，因此引入了**RNN**(Recurrent Neural Networks)循环神经网络作为提取时间序列特征的利器。

　　RNN的基本结构很简单，就是一个RNN基本单元以及其反馈，如：
<div align=center>![BasicRNN]</div>

　　有点像是IIR滤波器，因为计算机无法实现无限递归，因此展开得到如下的弱化结构：
![ExpandRNN]
其中所有的参数$U,W,V$都是共享参数，只有基本单元在不断的重复。
$$
h_{i+1} = f(Ux_{i+1}+Wh_i+b),其中h_i为第i个cell的状态量
$$
$$
y_{i} = softmax(Vh_i+c)，其中y_i为第i个cell的输出
$$
这种是典型的RNN结构，其输出长度和输入长度等长，适用于输入一段文字，预测下一段文字的应用场景。

# N输入 VS 1输出
　　在对动作序列进行识别分类的时候，需要输入多个序列而输出一个类别，因此有了以下这种RNN结构：
![ExpandRNN_N_1]
　　其实比起基本的RNN就只是将最后一个输出提取出来而已，其余的输出可以忽略。


# 1输入 VS N输出
　　适用于图片生成文字的场景，需要输入一张图片而输出多个文字片段，其基本结构如：
![ExpandRNN_1_N_type1]
　　也有改进为：
![ExpandRNN_1_N_type2]


# N输入 VS M输出
　　这种类型也称为编码器-解码器类型(Encoder-Decoder模型， Seq2Seq模型)，其多用于机器翻译，其思路就是构建编码器和解码器，编码器的基本结构如下：
![ExpandRNN_N_M_encoder]
　　编码器的作用是，**生成一个上下文编码向量$C$，这个向量可以表示输入的上下文，并且交由解码器解码出N的输出**。其中$C$可以有很多选择，如：
* $C = h_n$
* $C = q(h_n)$
* $C = q(h_1, h_2,\cdots,h_n)$
其中$q(x)$为关联函数，负责关联不同隐状态量之间的上下文。

　　解码的时候可以直接用**1输入VSN输出**的结构解码。这样就得到了**N输入 VS M输出**。




[BasicRNN]: ./imgs/BasicRNN.png
[ExpandRNN]: ./imgs/ExpandRNN.png
[ExpandRNN_N_1]: ./imgs/ExpandRNN_N_1.png
[ExpandRNN_1_N_type1]: ./imgs/ExpandRNN_1_N_type1.png
[ExpandRNN_1_N_type2]: ./imgs/ExpandRNN_1_N_type2.png
[ExpandRNN_N_M_encoder]: ./imgs/ExpandRNN_N_M_encoder.png