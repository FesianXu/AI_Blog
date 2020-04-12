<div align='center'>
    基础卷积神经网络架构总结
</div>

<div align='right'>
    2020/3/11 FesianXu
</div>

# 前言

本文总结了基础但是最为常用的10个卷积神经网络架构，并且尝试对其进行发展的辨析等，本文参考了若干网上的资料，并且加以总结而成。 **如有谬误或者补充，请各位联系指出，转载请注明出处。谢谢。**

$\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu



-------



卷积神经网络(Convolutional Neural Network, CNN)是深度学习中最为流行和常见的，用于提取图像特征的利器，正是由于以CNN和RNN为代表的基础神经网络的提出（当然算力也是功不可没），引起了这新一轮的人工智能热潮。目前为止，常用的基础卷积神经网络特征提取网络有以下几个，其中以3，4，5，6及其它们的变种最为常用。

1. LeNet-5,   **Paper**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/index.html#lecun-98) (Proceedings of the IEEE (1998))

2. AlexNet,  **Paper**: [ImageNet Classification with Deep Convolutional Neural](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) (NeurIPS 2012)

3. VGG-16,  **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) 

4. Inception-v1,  **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) （CVPR 2015）

5. Inception-v3,  **Paper**: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) （CVPR 2016）

6. ResNet-50, **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)

   

