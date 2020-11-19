<div align='center'>
    在Graph上定义卷积操作，图卷积网络
</div>

<div align='right'>
     FesianXu 5/19, 2019 at UESTC
</div>


# 前言

我们曾在[1]中探讨了欧几里德结构数据（如图像，音视频，文本等）和非欧几里德结构数据（如Graph和Manifold等）之间的不同点，在本文中，我们探讨如何在非欧几里德结构数据，特别是Graph数据上定义出卷积操作，以便于实现深度神经学习。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----



# 非欧几里德数据 嵌入 欧几里德空间
我们提到过欧几里德数据可以很容易嵌入到欧几里德空间中，无论是样本空间还是经过特征提取后的特征空间，这个特性可以方便后续的分类器设计。然后，遗憾的是，非欧几里德数据因为天然地具有不固定的领域元素数量或者连接数量等，不能直观地嵌入欧几里德空间，并且也很难在Spatial上定义出卷积操作出来，而这个卷积操作，在欧几里德数据上是很直观可以定义出来的，如：
$$
(f \star \gamma)(x) = \int_{\Omega} f(x-x^{\prime})\gamma(x^{\prime}) \mathrm{d}x^{\prime}
\tag{1.1}
$$
因此我们后续要想办法在非欧几里德数据，比如常见的Graph数据上定义出卷积操作。在进一步探讨之前，我们不妨先探讨下什么叫做 **嵌入到欧几里德空间**以及为什么要这样做。
一般来说，欧几里德数据因为其排列整齐，天然可以嵌入到欧几里德空间内，并且进行欧几里德空间下定义的算子的度量，比如欧式距离等，然后可以进一步的进行样本之间的距离计算以及分类聚类等更为高级的操作。然而，非欧数据不能直接嵌入到其中，需要用特定的方法才能嵌入到欧几里德空间中，在Geometric Deep Learning中，这个特定方法就是指的是深度学习方法，整个框图如：

![feat_proj][feat_proj]
![graph_proj][graph_proj]

<div align='center'>
    <b>
        Fig 1. 将非欧几里德结构数据通过深度学习方法映射到欧几里德结构的特征空间中。
    </b>
</div>
有了这个操作，即便是对于元素排列不整齐的Graph或者Manifold，也可在欧几里德空间进行样本之间的距离度量了，而且，这个过程还往往伴随着降维，减少计算量，方便可视化等优点。这个将会方便后续的分类器，聚类器等设计。

---

# Graph Deep Learning
因为Graph数据是最为常见的非欧几里德数据，我们这里以图深度学习为例子。图深度学习的任务目标可以分为几种：

1.  将有着不同拓扑结构，不同特征的图分类为几类。在这种情况是对整个Graph进行分类，每个Graph有一个标签。
2. 对一个Graph的所有节点node进行分类。这种情况相当于是文献引用网络中对文献类型分类，社交网络对用户属性分类，每个节点有一个标签。
3. 生成一个新的Graph。这个相当于是药物分子的生成等。

![medicines_gnn][medicines_gnn]

<div align='center'>
    <b>
        Fig 2. 使用GNN去探索新的药物分子。
    </b>
</div>

其中，最为常见的是第一类型，我们对此进行详细的任务定义如：
>  我们用$\mathcal{G}=\{A, F\}$表示一个图，其中$A \in \{0,1\}^{n \times n}$是对于图的邻接矩阵[2]，而$F \in \mathbb{R}^{n \times d}$是节点的特征矩阵，其中$n$表示有n个节点，d表示每个节点有d个特征。给定一个有标签的graph样本集: $\mathcal{D} = \{(G_1, y_1), \cdots, (G_n, y_n)\}$，其中$y_i \in \mathcal{Y}$是标签有限集合，并且对应于$G_i \in \mathcal{G}$，那么我们的学习目标就是学习到一个映射$f$使得：
>  $$
>  f: \mathcal{G} \rightarrow \mathcal{Y}
>  $$

## 在频域定义卷积
我们之前谈到在spatial域上难以直接定义graph的卷积操作，那么我们自然就想到如果能在频域定义出来卷积，那也是不错的，因此我们接下来想要探讨怎么在频域定义卷积。在此之前，我们需要了解下**热传播模型**的一点东西，因为图中节点的信息传递，一般来说是通过邻居节点进行传递的，这一点和物体将热量从高到低的传递非常相似，可以对此建立和热传递相似的模型。

在[3]的这篇回答中，作者对热传播和图节点信息传递进行了非常精彩的阐述，并且引出了 **拉普拉斯矩阵(Laplacian Matrix)** 对于节点之间关系的描述作用，值得各位读者参考。

**总的来说，就是对拉普拉斯矩阵进行特征值分解，其每个特征向量可以看成是频域中正交的正交基底，其对应的特征值可以看成是频率**，对拉普拉斯矩阵进行特征值分解的公式如下：
$$
\Delta \mathbf{\Phi}_{k} = \mathbf{\Phi}_{k} \mathbf{\Lambda}_{k}
\tag{2.1}
$$
其中$\Delta$是拉普拉斯矩阵，而$\mathbf{\Phi}_{k}$是前k个拉普拉斯特征向量组成的矩阵，而$\mathbf{\Lambda}_k$是由对应特征值组成的对角矩阵。我们接下来先暂时忘记这个(2.1)公式，我们要对整个图的拓扑结构进行表示，那么通过邻接矩阵A就可以很容易的表示出来，其中，无向图的邻接矩阵是对称矩阵，而有向图的邻接矩阵则不一定，让我们举一个例子方便接下来的讨论，如Fig 4所示，这是个无向图的例子。其中我们之前谈到的拉普拉斯矩阵可以用公式(2.2)
$$
L = D-A
\tag{2.2}
$$
确定，其中$D$为度数矩阵，$A$为邻接矩阵，整个过程如Fig 3.所示。

![graph][graph]

<div align='center'>
    <b>
        Fig 3. 一个图结构数据的例子，其中有度数矩阵，邻接矩阵和拉普拉斯矩阵。
    </b>
</div>

总的来说，用拉普拉斯矩阵可以在某种程度上表示一个Graph的拓扑结构，这点和邻接矩阵相似。

注意到我们有对拉普拉斯矩阵$L$的特征值分解：
$$
LU = U\Lambda
\tag{2.3}
$$
其中$U = [u_0, \cdots,u_{n-1}] \in \mathbb{R}^{n \times n}$为正交矩阵，其每一列都是特征向量，而$\Lambda = \mathrm{diag}([\lambda_0, \cdots, \lambda_{n-1}]) \in \mathbb{R}^{n \times n	}$是一个对角矩阵，其每个对角元素都是对应特征向量的特征值。对式子(2.3)进行变换，注意到正交矩阵的转置等于其逆，我们有:
$$
\begin{aligned}
L &= U \Lambda U^{-1} \\
&= U \Lambda U^T
\end{aligned}
\tag{2.4}
$$
因此，对于一个给定信号$x \in \mathbb{R}^n$来说，其傅立叶变换可以定义为 $\hat{x} = U^{T}x \in \mathbb{R}^n$，其反傅立叶变换为 $x = U\hat{x}$，我们用符号 $*_{\mathcal{G}}$ 表示图傅立叶变换。那么对于信号$x$和卷积核$y$，我们有:
$$
x *_{\mathcal{G}} y = U((U^{T}x) \odot (U^{T}y))
\tag{2.5}
$$
其中$\odot$表示逐个元素的相乘。

那么对于一个卷积核$g_{\theta}(\cdot)$，我们有:
$$
\begin{aligned}
y &= g_{\theta}(L)(x) = g_{\theta}(U\Lambda U^{T})x = U g_{\theta}(\Lambda)U^{T}x \\
&\mathrm{where} \ \ g_{\theta}(\Lambda) = \mathrm{diag}(\theta) = \begin{bmatrix} 
\theta_1 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \theta_{n-1}
\end{bmatrix}
\end{aligned}
\tag{2.6}
$$
其中我们需要学习的参数就在$\mathrm{diag}(\theta)$中，一共有$n$个。

类似于一般欧几里德结构数据的傅立叶变换，在图傅立叶变换中，其每个基底（也就是特征向量）也可以可视化出来，如Fig 4所示:

![fourier_base_visualization][fourier_base_visualization]

<div align='center'>
    <b>
        Fig 4. 图傅立叶变换的基底可视化，每个红色区域的都是卷积核中心，可以类比为热传递中的热量中心。
    </b>
</div>

![manifold_base][manifold_base]

<div align='center'>
    <b>
        Fig 5. manifold傅立叶变换的基底可视化，每个红色区域的都是卷积核中心，可以类比为热传递中的热量中心，对于Manifold来说也有类似的性质。
    </b>
</div>

# ChebNet， 切比雪夫网络
上面介绍的网络有两个很大的问题：
1. 它们不是在空间上具有**局部性**的，比如二维图像的卷积网络就具有局部性，也称之为局部连接，某个输出神经元只是和局部的输入神经元直接相连接。这个操作有利于减少计算量，参数量和提高泛化能力。
2. 计算复杂度为$\mathcal{O}(n)$，和节点的数量成比例，不利于应用在大规模Graph上。

为了解决第一个问题，我们把$g_{\theta}(\Lambda)$写成:
$$
g_{\theta}(\Lambda) = \sum_{k=0}^K \theta_k \Lambda^{k}, \theta \in \mathbb{R}^K是一个多项式系数
\tag{3.1}
$$
我们在图论中知道有个结论：
> 若节点 $i$和节点 $j$的最短路径$d_{\mathcal{G}}(i,j) > K$，那么有 $(L^{K})_{i,j} = 0$，其中 $L$ 为拉普拉斯矩阵。

因此有:
$$
(g_{\theta}(L)\delta_i)_j = (g_{\theta}(L))_{i,j} = \sum_{k} = \theta_{k}(L^{k})_{i,j}
\tag{3.2}
$$
不难看出当$k > K$，其中$K$为预设的核大小 时， $(L^k)_{i,j} = 0$，因此式子(3.2)其实只有前$K$项不为0，因此是具有K-局部性的，这样我们就定义出了局部性，那么计算复杂度变成了$\mathcal{O}(K)$。

注意到，在式子$y=Ug_{\theta}(\Lambda)U^{T}x$中，因为$U \in \mathbb{R}^{n \times n}$，因此存在$U, U^T$的矩阵相乘，其计算复杂度为$\mathcal{O}(n^2)$，而且每次计算都要计算这个乘积，这个显然不是我们想看到的。

一种解决方法就是把$g_{\theta}(L)$参数化为一个可以从$L$递归地计算出来的多项式，用人话说就是可以$k$时刻的$g_{\theta}(L)$可以由$k-1$时刻的$g_{\theta}(L)$简单地通过多项式组合计算出来。在文章[5]中，使用了**切比雪夫多项式展开**作为这个近似的估计。该式子可表示为:
$$
T_k(x) = 2xT_{k-1}(x)-T_{k-2}(x), \ \ \ \mathrm{where} \ \ T_0 = 1, T_1 = x
\tag{3.3}
$$
因此我们最后有:
$$
g_{\theta}(\Lambda) = \sum_{k=0}^K \theta_kT_{k}(\tilde{\Lambda})
\tag{3.4}
$$
式子(3.4)仍然是和$\Lambda$有关的值，我们希望直接和$L$相关，以便于计算，因此继续推导，有：
$$
\begin{aligned}
g_{\theta}(\Lambda) &= \sum_{k=0}^K \theta_k \Lambda^k \\
U g_{\theta}(\Lambda) U^T &= \sum_{k=0}^K \theta_k U \Lambda^k U^T
\end{aligned}
\tag{3.5}
$$
注意到$U$是幂等矩阵，也就是有$U^k = U$，因此继续推导(3.5)有：
$$
\begin{aligned}
& U \Lambda^k U^T = (U \Lambda U^T)^k = L^k \\
&\Rightarrow U g_{\theta}(\Lambda) U^T = \sum_{k=0}^K \theta_k L^k 
\end{aligned}
\tag{3.6}
$$
同样的，采用切比雪夫多项式展开，有：
$$
g_{\theta}(\Lambda) \approx \sum_{k=0}^K \theta_k T_{k}(\tilde{L})
\tag{3.7}
$$

因此，最后对于第$j$个输出的特征图而言，我们有：
$$
y_{s,j} = \sum_{i=1}^{F_{in}} g_{\theta_{i,j}}(L)x_{s,i} \in \mathbb{R}^{n}
\tag{3.8}
$$
其中$x_{s,i}$是输入的特征图，$s$表示第$s$个样本。因此我们一共有$F_{in} \times F_{out}$个向量，每个向量中有$K$个参数，因为$\theta_{i,j} \in \mathbb{R}^{K}$，最后一共有$F_{in} \times F_{out} \times K$个可训练参数。其中的$g_{\theta_{i,j}}(L)$是采用了切比雪夫多项式展开，因此可以递归运算，以减少运算复杂度。


----

# ChebNet一阶近似
根据我们之前的讨论，K阶的ChebNet可以表示为：
$$
y = U g_{\theta}(\Lambda) U^T x = \sum_{k=0}^{K} \theta_k U \Lambda^k U^T x
\tag{4.1}
$$
我们从VGG网络的设计中知道，3*3的卷积核在足够多的的层数的叠加后，有足够的层次信息可以提供足够大的感知野[6]。因此，也许对于ChebNet的一阶近似就足够了，因此我们将$K=1$，有：
$$
\begin{aligned}
y_{K=1} &= \theta_0 x + \theta_1 Lx = \theta(D^{-\frac{1}{2}} A D^{-\frac{1}{2}})x
\end{aligned}
\tag{4.2}
$$
在式子(4.2)中，我们假设了$\theta_0 = -\theta_1$以进一步减少参数量，而且对拉普拉斯矩阵进行了归一化，这个归一化操作我们接下来会继续讨论，这里暂且给出归一化的公式为:
$$
L = I_n - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
\tag{4.3}
$$
其中，$D$是度数矩阵，$D_{i,i} = \sum_{j} A_{i,j}$。

另外，为了增加节点自身的连接（这个我们以后继续讲解），我们通常会对邻接矩阵$A$进行变化，有:
$$
\begin{aligned}
\tilde{A} &= A+I_n \\
\tilde{D}_{i,i} &= \sum_{j} \tilde{A}_{i,j} 
\end{aligned}
\tag{4.4}
$$
因此最终有ChebNet的一阶近似结果为:
$$
g_{\theta}  \ *_{\mathcal{G}} x = \theta(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}})x
\tag{4.5}
$$
对(4.5)进行矩阵表达并且加入激活函数$\sigma(\cdot)$，有 **Graph Convolution Network(GCN)** 的表达[7]，如：
$$
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
\tag{4.6}
$$
其中的$H^{(l)}$是$l$层的特征图，$W^{(l)}$是$l$层的可学习参数。其中激活函数一般选择ReLU，即是$\sigma(x) = \max(0,x)$。

![graph_network][graph_network]

本篇文章就此结束，我们在下一篇文章将会继续介绍GCN在空间域上的理解，即是基于消息传递（Message Passing）中的解释，并且会举出一些例子来方便理解。



# 该系列其他文章

1. [《学习geometric deep learning笔记系列》第一篇，Non-Euclidean Structure Data之我见](https://fesian.blog.csdn.net/article/details/88373506)
2. [《Geometric Deep Learning学习笔记》第三篇，GCN的空间域理解，Message Passing以及其含义](https://blog.csdn.net/LoseInVain/article/details/90348807)

# Reference
[1]. https://blog.csdn.net/LoseInVain/article/details/88373506
[2]. https://en.wikipedia.org/wiki/Adjacency_matrix
[3]. https://www.zhihu.com/question/54504471/answer/630639025
[4]. https://blog.csdn.net/wang15061955806/article/details/50902351
[5]. Defferrard M, Bresson X, Vandergheynst P. Convolutional neural networks on graphs with fast localized spectral filtering[C]//Advances in neural information processing systems. 2016: 3844-3852.
[6]. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
[7]. Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.






[qrcode]: ./imgs/qrcode.jpg
[feat_proj]: ./imgs/feat_proj.png

[graph_proj]: ./imgs/graph_proj.png
[medicines_gnn]: ./imgs/medicines_gnn.png
[graph]: ./imgs/graph.png
[fourier_base_visualization]: ./imgs/fourier_base_visualization.png
[manifold_base]: ./imgs/manifold_base.png
[graph_network]: ./imgs/graph_network.png



