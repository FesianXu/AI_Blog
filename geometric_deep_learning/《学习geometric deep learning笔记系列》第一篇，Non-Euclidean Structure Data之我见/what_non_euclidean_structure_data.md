<div align='center'>
    《学习geometric deep learning笔记系列》第一篇，Non-Euclidean Structure Data之我见
</div>

<div align='right'>
    FesianXu at UESTC
</div>

# 前言

本文是笔者在学习`Geometric deep learning`的过程中的一些笔记和想法，较为零散，主要纪录了非欧几里德结构数据和欧几里德结构数据之间的区别，后续会引出图卷积网络模型。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----



总的来说，数据类型可以分为两大类，分别是：**欧几里德结构数据(Euclidean Structure Data)** 以及 **非欧几里德结构数据(Non-Euclidean Structure Data)**，接下来谈自己对这两类数据的认识。

# 欧几里德结构样本

在我们日常生活中，最常见到的媒体介质莫过于是`图片(image)`和`视频(video)`以及`语音(voice)`了，这些数据有一个特点就是：“排列整齐”。什么叫做排列整齐呢？举例子来说，图片可以用矩阵来表达其像素，就如同下图所示[2]：

![formatted_data][formatted_data]

<div align='center'>
    <b>
        Fig 1. 欧几里德结构数据示例。
    </b>
</div>

对于某个节点，我们很容易可以找出其邻居节点，就在旁边嘛，不偏不倚。而且，图片数据天然的，节点和邻居节点有着统计上的相关性，因此能够找出邻居节点意味着可以很容易地定义出卷积这个操作出来，而我们在深度学习的过程中知道，卷积这个操作是提取局部特征以及层次全局特征的利器，因此图片可以很容易定义出卷积操作出来，并且在深度网络中进行进一步操作。

而且，因为这类型的数据排列整齐，不同样本之间可以容易的定义出“距离”这个概念出来。我们且思考，假设现在有两个图片样本，尽管其图片大小可能不一致，但是总是可以通过空间下采样的方式将其统一到同一个尺寸的，然后直接逐个像素点进行相减后取得平方和，求得两个样本之间的欧几里德距离是完全可以进行的。如下式所见：
$$
d(\mathbf{s_i}, \mathbf{s_j}) = \dfrac{1}{2}||\mathbf{s_i}-\mathbf{s_j}||^2
\tag{1}
$$
因此，不妨把图片样本的不同像素点看成是高维欧几里德空间中的某个维度，因此一张$m \times n$的图片可以看成是$m \times n$维的欧几里德样本空间中的一个点，而不同样本之间的距离就体现在了样本点之间的距离了。

**这就是称之为欧几里德结构数据的原因了。** 同样的，视频可以在时间轴上进行采样做到统一的目的，而音频也是一样的。因此它们都是符合欧几里德距离定义的类型的样本。

# 非欧几里德结构样本
非欧几里德结构的样本总得来说有两大类型[1]，分别是图(Graph)数据[3]和流形数据[4]，如Fig 2和Fig 3所示:

![graph_data][graph_data]

<div align='center'>
    <b>
        Fig 2. 图结构数据是典型的非欧几里德结构数据。
    </b>
</div>

![manifold][manifold]

<div align='center'>
    <b>
        Fig 3. 流形数据也是典型的非欧几里德结构数据。
    </b>
</div>

这两类数据有个特点就是，排列不整齐，比较的随意。具体体现在：**对于数据中的某个点，难以定义出其邻居节点出来，或者是不同节点的邻居节点的数量是不同的**[5]，这个其实是一个特别麻烦的问题，因为这样就意味着难以在这类型的数据上定义出和图像等数据上相同的卷积操作出来，而且因为每个样本的节点排列可能都不同，比如在生物医学中的分子筛选中，显然这个是一个Graph数据的应用，但是我们都明白，不同的分子结构的原子连接数量，方式可能都是不同的，因此难以定义出其欧几里德距离出来，这个是和我们的欧几里德结构数据明显不同的。**因此这类型的数据不能看成是在欧几里德样本空间中的一个样本点了**，而是要想办法将其嵌入(embed)到合适的欧几里德空间后再进行度量。而我们现在流行的Graph Neural Network便可以进行这类型的操作。这就是我们的后话了。

-----

另外，欧几里德结构数据所谓的“排列整齐”也可以视为是一种特殊的非欧几里德结构数据，比如说是一种特殊的Graph数据，如下图所示[5]：

![image_data][image_data]

<div align='center'>
    <b>
        Fig 4. 即便是欧几里德结构数据，也可以视为是特殊形式的非欧几里德结构数据。
    </b>
</div>

因此，用Graph Neural Network的方法同样可以应用在欧几里德结构数据上，比如文献[6]中report的结果来看，的确这样是可行的。事实上，**只要是赋范空间中的数据，都可以建立数据节点与数据节点之间的某种关联，都可以尝试用非欧几里德结构数据的深度方法进行实验。[7]** 

那么什么叫做赋范空间中的数据呢？赋范空间，指的就是定义了范数的向量空间，我认为，指的是数据中的每个样本的单元的特征维度都是一致的，比如，一张图片的像素一般都是RGB三个维度的，不同像素之间可以进行求范数的操作，再比如，一个Graph上的某个节点和另外一个节点的维度都是相同的，因此也可以定义出范数出来。不过这个是我一家之言，如有其他见解，请在评论区指出。

![result][result]

<div align='center'>
    <b>
        Fig 5. 在传统的图片上利用图神经网络进行分类，可以达到接近传统CNN方法的效果。意味着欧几里德结构数据也可以通过某种形式，用非欧几里德结构数据模型建模。
    </b>
</div>



# 该系列的后续

1. [《Geometric Deep Learning学习笔记》第二篇， 在Graph上定义卷积操作，图卷积网络](https://blog.csdn.net/LoseInVain/article/details/90171863)

2. [《Geometric Deep Learning学习笔记》第三篇，GCN的空间域理解，Message Passing以及其含义](https://blog.csdn.net/LoseInVain/article/details/90348807)

   

-----

# Reference
[1]. Bronstein M M, Bruna J, LeCun Y, et al. Geometric deep learning: going beyond euclidean data[J]. IEEE Signal Processing Magazine, 2017, 34(4): 18-42.
[2]. https://www.zhihu.com/question/54504471
[3]. https://en.wikipedia.org/wiki/Graph
[4]. https://en.wikipedia.org/wiki/Manifold
[5]. Niepert M, Ahmed M, Kutzkov K. Learning convolutional neural networks for graphs[C]//International conference on machine learning. 2016: 2014-2023.
[6]. Defferrard M, Bresson X, Vandergheynst P. Convolutional neural networks on graphs with fast localized spectral filtering[C]//Advances in neural information processing systems. 2016: 3844-3852.
[7]. https://www.zhihu.com/question/54504471



[qrcode]: ./imgs/qrcode.jpg
[formatted_data]: ./imgs/formatted_data.png
[graph_data]: ./imgs/graph_data.png
[manifold]: ./imgs/manifold.png
[image_data]: ./imgs/image_data.png
[result]: ./imgs/result.png

