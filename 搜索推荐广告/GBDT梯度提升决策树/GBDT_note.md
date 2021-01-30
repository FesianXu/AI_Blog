<div align='center'>
    GBDT-梯度提升决策树的一些思考
</div>

<div align='right'>
    FesianXu 20210129 @ Baidu intern
</div>

# 前言

最近笔者工作中用到了GBRank模型[1]，其中用到了GBDT梯度提升决策树，原论文的原文并不是很容易看懂，在本文纪录下GBDT的一些原理和个人理解，作为笔记。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：



![qrcode][qrcode]

------

梯度提升决策树（Gradient Boost Decision Tree，GBDT）是一种通过**梯度提升策略**去提高**决策树**性能表现的一种树模型设计思想，本文集中讨论的是GBDT在回归中的应用。对于一个数据集$\mathcal{D}=\{\mathbf{x}_i,y_i\}_{i=1,\cdots,N}$而言，$\mathbf{x}_i \in \mathbb{R}^{d}, y_i \in \mathbb{R}$为第$i$个样本的特征和输出目标值，我们期望学习到一个模型$f(\cdot)$，有$y_i = f(\mathbf{x}_i)$。在GBDT中，我们用加性模型（additive model）去对这个模型建模，也即是有：
$$
f(\mathbf{x}_i) = f_0(\mathbf{x}_i)+\sum_{k=1}^{K}\alpha_k f_k(\mathbf{x}_i)
\tag{1}
$$
其中的$f_{k}(\cdot), k=0,\cdots,K$为基础子模型（base model），在GBDT中使用CART决策树作为这个基础子模型。假设得到了$f_{k-1}(\mathbf{x}_i)$的情况下，我们期望通过CART树模型，学习到一个$h_k(\mathbf{x}_i)$，使得有式子(2)表示的loss最小，那么就可以更新模型到$f_{k}(\cdot)$了：
$$
\sum_{i=1}^N \mathcal{L}(y_i, f_k(\mathbf{x}_i)) = \sum_{i=1}^{N} \mathcal{L}(y_i, f_{k-1}(\mathbf{x}_i)+h_{k}(\mathbf{x}_i))
\tag{2}
$$
那么我们要如何去更新模型，以及取得最初始的$f_0(\cdot)$呢？我们先看一个例子，如Fig 1所示，蓝色点表示若干样本点，现在希望GBDT模型可以拟合这些样本点，从直观上，我们的初始值考虑需要和所有样本的loss之和最小，也就是：
$$
f_0(\mathbf{x}) = \arg\min_{c} \sum_{i=1}^N \mathcal{L}(y_i, c)
\tag{3}
$$
当损失函数为MSE函数时，也即是$\mathcal{L}(x,y) = \dfrac{1}{2}(x-y)^2$时，我们通过求导的方法可以求得式子(3)的最优解为:
$$
c = \dfrac{1}{N}\sum_{i=1}^N y_i
\tag{4}
$$
也即是所有待回归值的均值，如Fig 1中的绿色虚线所示。这一点也很容易理解，在不知道其他任何信息量时，初始化选择数据集待回归值的中心时可能导致偏差可能性最小的。

![tree_f0][tree_f0]

<div align='center'>
    <b>
        Fig 1. 若干样本点（蓝色），现在希望学习到的GBDT模型可以拟合这些样本点。
    </b>
</div>

接下来我们得考虑如何去更新模型，在得到了$f_{k-1}(\mathbf{x})$之后，拟合并不是完美的，和最终目标$y_i$仍然有偏差，这个偏差就是$\mathcal{L}(y_i, f_{k-1}(\mathbf{x}))$，为了让模型更新方向操作缩小这个偏差的方向行进，我们可以借助**负梯度方向**更新的方法。首先我们得求出此时偏差的负梯度：
$$
r_{ki} = -\dfrac{\partial\mathcal{L}(y_i, f_{k-1}(\mathbf{x}_i))}{\partial f_{k-1}(\mathbf{x}_i)}
\tag{5}
$$
然而，式子(5)有个小问题，一个函数如何对一个函数求导数呢，我们之前接触的都是一个函数对其中一个变量求导数。其实这个倒也不难，因为对于函数的每个输入而言，都会输出一个值（对于单值函数而已），因此我们可以把函数看成是一个无限维的向量，向量的每一个维度都是对应输入的函数输出，如$f(x) = [v_0,\cdots,v_n] \in \mathbb{R}^{\infty}$。在样本有限的情况下，比如只有$N$个，那么我们可以用$N$维向量去近似表示这个函数。回归到我们的原问题，我们的$f_{k-1}(\mathbf{x})$其实可以表示为$N$个样本的向量输出表示，为$f_{k-1}(\mathbf{x}) = [v_0,\cdots,v_N] \in \mathbb{R}^{N}$，那么对函数求导的问题就变成对向量求导了。如果损失函数是MSE函数，那么我们可以求得其梯度为：
$$
\dfrac{\partial\mathcal{L}(y_i, f_{k-1}(\mathbf{x}_i))}{\partial f_{k-1}(\mathbf{x}_i)} = f_{k-1}(\mathbf{x}_i)-y_i
\tag{6}
$$
但是！我们注意到这个梯度计算得并不准确，因为我们只能用到能够观察到的$N$个训练样本去估计梯度，这个梯度对于未知样本而言是不置信的。如Fig 2所示，黄色点是通过有限的$N$个样本估计出来的$N$个梯度，而虚线才是实际梯度函数，因此我们需要用某种方式从有限个样本中估计出梯度函数，在GBDT中采用了CART树模型去进行这个操作。也即是说，通过树模型$h_{k}(\mathbf{x})$去拟合$\mathcal{G} = \{\mathbf{x}_i, r_{ki}\}, i=1,\cdots,N$。在损失函数为MSE函数的情况下，我们发现其实就是在拟合前$k-1$模型所造成的误差，如式子(6)所示。

![gradient_estimation][gradient_estimation]

<div align='center'>
    <b>
        Fig 2. 黄色点是观察有限样本估计的梯度，而虚线才是实际的梯度函数曲线。
    </b>
</div>

注意到得到的CART树模型$h_{k}(\mathbf{x})$，对于每个输入的样本$\mathbf{x}_i$其最终都会归结为某端的某个叶子节点，也即是预测梯度值，我们用$R_{kj}$表示第$k$个回归树的第$j$个叶子节点，其中$j=1,\cdots,J$，$J$为叶子节点的数量。这样不够准确，应该很可能不同的样本贡献同一个预估的梯度值，如Fig 3所示，对于属于某个集合$\mathcal{S}$的样本$\mathbf{x}_i$来说，可能CART树$h_k(\cdot)$会将其归为同一个叶子节点$R_{kj}|_{j=j_{0}}$，此时树模型其实起到的作用跟类似与聚类，如Fig 1所示就可以发现有明显的三个聚类簇。对于不同的聚类簇，其需要“弥补”以减少与待回归值之前差别的值$c_{kj}$是不同的，我们同样可以求最小值的方法求得，如：
$$
c_{kj} = \arg\min_{c} \sum_{\mathbf{x}_i \in R_{kj}} \mathcal{L}(y_i, f_{k-1}(\mathbf{x}_i)+c)
\tag{7}
$$
这个式子一眼并不容易看懂，式子(7)其实是对于被CART回归树$h_k(\cdot)$归为同一叶子节点的样本进行处理，对于同一类的样本$\mathbf{x}_i$而言（也即是$\mathbf{x}_i \in R_{kj}$），我们通过Linear Search的方法求得了对于该类的$c$，使得$f_{k-1}(\mathbf{x})+c$之后使得损失最小，这就是式子(7)的含义。这个过程同样可以通过求导的方式进行最优化求解，假如损失函数为MSE函数，那么我们可以求得：
$$
c_{kj} = \dfrac{1}{|{i:\mathbf{x}_i \in R_{kj}}|} \sum_{\mathbf{x}_i \in R_{kj}} (y_i-f_{k-1}(\mathbf{x}_i))
\tag{8}
$$
其中的$|{i:\mathbf{x}_i \in R_{kj}}|$是求得该类样本的数量，以便于做归一化。

![tree_terminal_node][tree_terminal_node]

<div align='center'>
    <b>
        Fig 3. 可能多个不同的输入样本在CART树模型中会被归为同一个叶子节点。
    </b>
</div>

此时我们便有：
$$
h_k(\mathbf{x}) = \sum_{j=1}^{J} c_{kj} I(\mathbf{x} \in R_{kj})
\tag{9}
$$
$I(\cdot)$表示其中条件满足时，返回1，否则返回0。因此本轮的最终更新的回归函数为：
$$
f_k(\mathbf{x}) = f_{k-1}(\mathbf{x})+h_k(\mathbf{x})
\tag{10}
$$
对于$k=1,\cdots,K$我们不停地迭代计算，可以求得最终的回归函数为：
$$
f(\mathbf{x})=f_{K}(\mathbf{x}) = f_0(\mathbf{x})+\sum_{k=1}^{K}\sum_{j=1}^{J} c_{kj} I(\mathbf{x} \in R_{kj})
\tag{11}
$$
这也就是加性模型(1)的另一种表现形式而已。

因此，如果当损失函数为MSE函数时，可以表现为是用多棵CART树模型对残差的不断拟合，但是如果当损失函数不是MSE时，则不能这样解释。



# Reference

[1]. Zheng, Z., Chen, K., Sun, G., & Zha, H. (2007, July). A regression framework for learning ranking functions using relative relevance judgments. In *Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval* (pp. 287-294).





[qrcode]: ./imgs/qrcode.jpg
[tree_f0]: ./imgs/tree_f0.png
[gradient_estimation]: ./imgs/gradient_estimation.png
[tree_terminal_node]: ./imgs/tree_terminal_node.png

