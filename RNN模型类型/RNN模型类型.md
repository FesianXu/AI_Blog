<h1 align = "center">RNN模型的类型</h1>

## 前言
**
深度学习中CNN多用于空间域相关信息的特征提取，而时间序列的特征提取如自然语言，语音序列，视频动作序列等，需要RNN模型的支持，这里浅谈一些RNN模型。
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
![BasicRNN]





[BasicRNN]: ./imgs/BasicRNN.png