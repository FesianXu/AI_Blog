<h1 align = "center">《weekly paper》DenseNet的理解</h1>

## 前言
**DenseNet是CVPR2017的best paper，其可以看成是ResNet和HighwayNet一类型的网络，都是通过引进了shortcut以减少梯度消失，增强不同层之间的信息融合。**

**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

*******************************************************

原论文标题为《Densely Connected Convolutional Networks》，其arxiv连接：[https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)


# 1. 为什么要用shortcut连接
在实践中，单纯的前馈深层网络特别容易出现梯度消失的问题，并且因为网络太深，参数太多，训练和测试耗时极大，这个时候就出现了残差网络ResNet和Highway Network，以ResNet为例，其在上下两层之间构建了一个shortcut，使得下一层可以共享前一层的信息，其基本结构如下图所示：

![resnet][resnet]

假设一个组合函数（里面可以包括BN，激活函数，卷积操作等）$H_{l}(\cdot)$，并且假设第$l$层的输出为$x_l$，那么传统的卷积网络可以表示为(1-1)：
$$
x_l= H_{l}(x_{l-1})
\tag{1-1}
$$
而在resnet中，因为引入了残差操作，因此公式化为(1-2)：
$$
x_{l} = H_{l}(x_{l-1})+x_{l-1}
\tag{1-2}
$$

然而作者认为，残差网络resnet在合并不同层的信息的时候，用的是向量加法操作（summation），这有可能会阻碍（impede）网络中的信息流动。在[《Deep networks with stochastic depth》](https://arxiv.org/pdf/1603.09382.pdf)的工作中，提出了一个观点，就是resnet中多达上百层的卷积层对整体的效果贡献是很少的，有许多层其实是冗余的，可以在训练的过程中随机丢掉（drop），实际上整个网络并不需要那么多参数就可以表现得很好。（类似于dropout的做法，只不过是dropout是对一个层里面的神经元输出进行丢弃，而stochastic depth是对整个层进行随机丢弃，都可以看成是一种正则手段。）

作者基于这种观察，将浅层的输出合并起来，作为后面若干层的输入，使得每一层都可以利用到前几个层提取出来的特征，在传统的CNN中，因为是串行的连接，$L$层网络就有$L$个连接，而在DenseNet中，因为采用了密集连结，所以$L$层网络有$L(L-1)/2$个连接，整个Dense连接见下图所示：

![densenet][densenet]

其中的这几层是经过了密集连结的一个单元，作者称之为“Dense Block”，我们可以发现，其实这个block结构还是很简单的，就是在block中位于$l$层的输入同时合并（concatenate）了上$l-1$层的所有输出，因此其输入feature map有$k_0+k*(l-1)$个，其中$k_0$是原始图片的通道数，$k$称为增长率，我们待会再谈。同样的，用公式去表达这个过程，就如式(1-3)所示：
$$
x_{l} = H_l([x_0, x_1, x_2, \cdots,x_{l-1}])
\tag{1-3}
$$

# 2. 关于DenseNet的更多细节

## 2.1 composite function和transition layer
在本文中，一个组合单元$H_l(\cdot)$包括有BN层，ReLU函数和卷积层，依次叠加。其中这里卷积层采用了3 $\times$ 3大小的卷积核。在CNN中，为了减少计算量，是需要进行下采样的，经常由池化操作进行。但是为了在dense block中可以合并输入，其输出输入的张量尺寸不能改变，因此在dense block中不能使用池化操作，而是应该在连接两个dense block之间应用池化，这个中间层，作者称之为transition layer。在transition layer中，依次叠加了一个BN层，1$\times$1大小的卷积和一个2$\times$2的均值池化。其中的1$\times$1卷积是为了进一步压缩张量的，这个我们后面谈。

## 2.2 growth rate
如果每一个函数$H_l(x)$都输出$k$个特征图，那么在第$l$层就会有$k_0+k \times (l-1)$个输入特征图，其中的$k_0$是初始图片的通道数。对比DenseNet和其他传统网络，前者特点就是可以做的很狭窄，因此这里的$k$不必取得很大（比如传统的基本上都是64，32，48等等），在这里$k=12$，这样就大大减少了参数量。这个超参数$k$就称之为growth rate。

为什么不需要那么多的特征图呢？作者认为特征图可以看成是网络的一种全局的状态，而在densenet中，因为某一层和前面几层都有连接，因此一旦全局状态一旦写入，就可以在后面的层里面直接挪用了，而不用像传统的网络一样在层与层之间复制这些状态。

## 2.3 bottleneck layer
这个层的引入原因很直接就是为了减少输入量。尽管每一个层只是产生了$k$个输出，但是经过若干层的叠加后，也会扩充的很大，会导致后面层的计算复杂度和参数问题，因此可以用$1 \times 1$的卷积操作进行降维，在本文中，具体操作是：**在原先的3 $\times$ 3的卷积层前面加入一个1 $\times$ 1的卷积**。具体的设置如：BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) 。作者统一让每个1x1卷积输出$4k$个特征图。（也就是说不管输入的特征图有多少通道，都会缩为4k个。）**将具有bottleneck layer的densenet称之为DenseNet-B。**

(为什么1x1能够降维？这个降维主要是体现在通道方向上的降维，可以将原先N个通道降到M个通道。)

## 2.4 compression
为了进一步增加模型的紧致性，作者在transition layer也进行了缩减特征图的操作，同样地，也是通过$1 \times 1$卷积实现，如果一个block输出了m个特征图，那么在接下来的transition layer中就缩减为$\lfloor \theta m \rfloor$个特征图，其中的$\theta$为缩减系数，有$\theta \in [0, 1]$。将同时具有bottleneck layer和transition layer（$\theta < 1$）的densenet称为DenseNet-BC。

# 3. 实验部分
在ImageNet上，整个网络的配置如：
![configuration][configuration]

实验结果如：
![result][result]

其中蓝色的表示最好结果，可以看到DenseNet的确是比其他网络有着更为优良的表现，同时其参数数量更少。

在同样的参数数量和运算量下的验证准确率对比：

![result_1][result_1]
可以发现，在同样的参数数量和运算量情况下，DenseNet无论是那种网络配置，都比ResNet的结果要好。

然后在测试集上进行对比，可以发现结论和验证集上的一致。
![result_2][result_2]


**总而言之，实验结果证明了DenseNet的有效性，并且在同样参数数量和运算量的程度下，DenseNet可以表现得比同类型的ResNet更好，并且作者在实验部分强调了他的DenseNet在ImageNet上的结果是没有经过调优的，因此还可以继续提高结果。**

# 4. 分析与讨论
为什么这个模型能够work呢？作者给了几个解释：
1. 采用了紧致性的模型，有着更少的参数，减少了过拟合。
2. 隐式深度监督（implict Deep 	supervision），因为存在有shortcut的存在，作为优化目标的loss function不必要从最后一层提取信息，可以直接取得前面层次的信息进行学习，类似与将分类器网络连接在所有隐藏层里面进行学习的deeply-supervised nets（DSN）网络。
3. 类似于ResNet，DenseNet这种shortcut的方式有利于梯度的传导，减少了梯度消失。

其实，这种密集连结的策略也类似于随机丢弃层的Stochastic depth网络，因为在Stochastic depth网络中，池化层是不会被丢弃的，而中间的卷积层可能会被丢弃，也就是在训练的时候，其实可以看出某两个非直接连接的层有可能因为丢弃的原因而连接起来，因此两者有点类似的地方。

按照设计，DenseNet应该可以通过前面层的信息中直接获取特征图从而得到提升小姑的，虽说实验效果的确是比resnet的要好，但是前面层的信息的短接（shortcut）真的对分类结果有贡献吗？作者又补充了一个实验：
![reuse][reuse]
作者训练了一个网络，然后将一个block中的源层(source layer)和目标层(target layer)之间的权值取均值，然后绘制成了一个热值图，如上。其中每一个图都是一个block的层层连接权值均值图。颜色越深表示目标层从源层里面得到了更多的贡献，我们可以发现，目标层除了最近邻的层之外，的确是在前面的若干层中获得了不少信息的。



# 5. 其他
因为现在的框架对于densenet没有直接的支持，因此实现的时候往往是采用concatenate实现的，将之前层的输出与当前层的输出拼接在一起，然后传给下一层。对于大多数框架（如 Torch 和 TensorFlow），每次拼接操作都会开辟新的内存来保存拼接后的特征。这样就导致一个 L 层的网络，要消耗相当于 L(L+1)/2 层网络的内存（第 l 层的输出在内存里被存了 (L-l+1) 份）。

这里有一些框架的解决方案，可供参考，具体内容见引用1：

1. Torch implementation: https://github.com/liuzhuang13/DenseNet/tree/master/models
2. PyTorch implementation: https://github.com/gpleiss/efficient_densenet_pytorch
3. MxNet implementation: https://github.com/taineleau/efficient_densenet_mxnet
4. Caffe implementation: https://github.com/Tongcheng/DN_CaffeScript



# Referrence
1. [《CVPR 2017最佳论文作者解读：DenseNet 的“what”、“why”和“how”｜CVPR 2017》](https://www.leiphone.com/news/201708/0MNOwwfvWiAu43WO.html)
2. [《DenseNet算法详解》](https://blog.csdn.net/u014380165/article/details/75142664)




[reuse]: ./imgs/reuse.png
[result_2]: ./imgs/result_2.png
[result_1]: ./imgs/result_1.png
[result]: ./imgs/result.png
[configuration]: ./imgs/configuration.png
[densenet]: ./imgs/densenet.png
[resnet]: ./imgs/resnet.png