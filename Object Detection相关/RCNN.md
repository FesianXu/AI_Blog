<div align='center'>
    《土豆学OD》 之 RCNN初探
</div>



# 前言

本土豆最近在做Human-Object Interaction（HOI）任务的研究，其中有用到物体识别的模块，因此也打算趁此机会把Object Detection（OD）的拿来系统学习下，并且在此纪录下笔记。土豆我深知OD已经在网络上有着很多中文博客资料了，但是个人觉得很多都不够详细，不够入门级，因此我尽量在此博客里面做到提供更多的细节等，希望尽量做到初学者友好。不过毕竟土豆在OD还是初学者，如果文章有纰漏的地方，请联系指出，谢谢。



----

# 从物体识别说起

据我所知，Region-CNN也就是简称为RCNN的这篇文章[1]是最早的一批尝试将深度学习应用在物体识别这个任务上的，我觉得有必要先对之前的主流方向思路先描述下。

一般来说，物体识别这个任务需要在提供了一张图片的情况下，将图片中某些特定物体（由训练集里面的标签类别指定，如苹果，猫，狗等）的位置标记出来，这个标记需要提供该物体在图中的坐标和大小$(x,y,w,h)$，这个标记我们一般称之为**bounding box** (bbox)，如下图所示。其次需要对这个物体的类别进行分类，比如将此分类为狗，猫等，有些情况下还会给定这个分类的置信度等。其评价标准一般是mAP，mean Average Precision。

![odtask][odtask]

<div align='center'>
    <b>
        Fig 1.1 物体识别中的bbox和类别分类以及其置信度等。
</div>

一般解决这个问题的思路很直接暴力，就是用设置多个不同尺寸的窗口，然后从图像的开始出开始遍历整个图像，得到不同尺寸的窗口的遍历集合之后，对每个窗口的图像片段进行物体分类判断，最后再考虑将其相同类别的窗口进行合并等，最后得到bbox和每个bbox的类别等。这个滑动窗口的方法叫做暴力搜索(Exhaustive Search)，显然的，这种方法会导致运算量特别的大，因为由这种方法会产生大量的滑动窗口需要后续进行分类，而且，我们知道这些窗口很多都是背景或者是重复的，重叠的区块，因此其实对于后续来说是多余的。

那么为了解决这个问题，就提出了所谓的**Selective Search** [2]这种方法，这个方法尝试去从每个图像中单纯根据图像的纹理，光照，形状，颜色等底层的图像信息去提取中少量的窗口，这里将这个窗口称之为侯选框 **Proposal**。一般来说会对每个图片提取出约2000个侯选框，这个数量将远远小于暴力搜索的方法。

而后续，为了对每个侯选框进行分类，在深度学习之前，会考虑采用传统的人工设计的特征，如HOG，SIFT，DPM等，也会考虑用复合的特征，如UVA detection system[3]。那么在深度学习之后呢，就会考虑采用卷积神经网络去弄啦，这个也就是我们之后文章的主题了。

----

# Region CNN

RCNN的思路虽然很简单，看起来不值得独立写成一篇博客，但是其实其中有很多细节值得注意，其在arxiv的原始论文也达到了21页之长[4]，因此还是有必要仔细学习下细节的。主要需要注意的分为几个部分：

1. 如何提取侯选框
2. 这里使用的CNN如何设计
3. 输入的图片该怎么进行尺寸统一化
4. 类别分类器的设计
5. 正负样本挑选
6. 训练细节等

不过在陷入细节之前，我们要明确，整个RCNN的流程如：

```mermaid
graph LR
	a(输入图片) ==> a1(侯选框提取)
	a1 ==> a2(CNN特征提取)
	a2 ==> a3(类别分类-使用SVM)
	a3 ==> a4(相同类别的侯选框进行聚合-使用NMS)
	
```

![rcnn][rcnn]

<div align='center'>
    <b>
        Fig 2.1 RCNN的流程框图。
</div>



## 如何提取侯选框

首先使用selective search先从每张图片中提取出2000个侯选框，注意这里的侯选框是所谓类别无关的，毕竟只是根据纹理等底层图像特征得到的。我们要注意到的是，这里的每个侯选框的尺寸可能都不一样，如Fig 2.2所示。因此在输入后续的CNN时，需要进行尺寸上的统一化，一般会统一到227 x 227这个尺寸，由此又有几种不同的统一化策略。

![selective_search][selective_search]

<div align='center'>
    <b>
        Fig 2.2 Selective Search提取出来的是类别无关的侯选框，而且每个侯选框的尺寸可能不一。
</div>

统一化这个尺寸除了单纯的线性缩放之外，还有考虑到上下文和原图比例的两种策略，见：

![norm][norm]

<div align='center'>
    <b>
        Fig 2.3 不同的图像统一化策略。A列为原始的侯选框的图像，B列为考虑到了侯选框周围p个像素的结果（这里p=16），也就是考虑了侯选框周围的上下文信息了，然后再进行线性的缩放等；C列是考虑到了原图侯选框的尺寸长宽比例，因此在周围进行了填充的操作，然后在缩放成指定尺寸，在这种情况下，我们发现C列的有效图像其比例和原始侯选框是一样的；而D列是简单的直接线性缩放。
</div>



至此，我们得到了侯选框和经过统一化的侯选框图像，接下来可以喂进CNN了。



## CNN的预训练和调整

这里的CNN特征提取网络比较了两种常用的网络，一种是T-net，也就是OxfordNet  [5]，另一种是O-net，也就是TorontoNet  [6]。从结果来看，基于O-net的结果好了8个mAP，但是其速度慢了七倍，因此作者最后的实验还是基于T-net进行的。T-net的网络结构如Fig 2.4所示，这里的输入本应该是224 x 224的，不知道为什么在论文里统一化成了 227 x 227，不过这个对我们的研究影响并不大因此忽略不考虑。

![alexnet][alexnet]

<div align='center'>
    <b>
        Fig 2.4 T-net的网络结构示意图。
</div>

RCNN利用T-net从每个proposal中提取一个4096维的特征，我们会发现这里的4096维比起以前传统方法的，比如UVA detection system [3]，维度少了两个数量级，分别是4k和360k，因此大幅度提高了运算速度。

显然，这个T-net首先需要预训练，一般是在ImageNet数据集（ILSVRC2012 classification ）上利用图片级的标注（比如图片分类任务，不能利用其bbox的标签数据）先进行预训练。为了适应OD任务中$N$个物体的分类，需要将T-net的最后一层的1000类分类的层更换成$N+1$层，其中加上一表示背景。

我们注意到最后一层的池化层，也即是$Pool_{5}$层，其输出特征图尺寸为$6 \times 6 \times 256=9216$维（按照论文中报告的，和Fig 2.4所示不同）。



## 类别分类器为什么不使用softmax

那么在上一步得到了proposal的特征之后，下一步就是拿到分类器中给每个proposal进行分类了。这里的分类器采用了线性SVM分类器，也即是给每个类别都指定了一个二类线性SVM分类器，用于判定是否是本类，见Fig 2.1。那么这里为什么不用softmax分类器而是采用传统的SVM呢？作者在实验中发现如果更改为softmax，那么在VOC 2007数据集上的表现将从54.2%下降到50.9%，作者认为是因为这里的正负样本采样，并没有特别强调精细的位置关系，因此softmax是在随机采样的负样本中进行训练的，而不是如同在SVM策略中，可以采用所谓的难负样本（hard negatives）进行训练。







## 其他训练细节等





# bbox回归

用了以上的策略，我们的RCNN的表现已经能够超越以前的工作了，但是还不够。我们知道根据selective search选出来的侯选框直接决定了后续的物体的bbox的标定和类别的决定，但是selective search是一个静态过程，是不能学习的，也是不能调整的，这就直接决定了其bbox的效果固定了。这个很糟糕，应该是可以提高的。因此作者借鉴DPM模型[7]中的策略，对bbox进行回归，期待可以微调bbox的位置，取得更好效果。如Fig 3.1所示，我们将$N$个训练样本表示为$\{(P^i, G^i)\}_{i=1,\cdots,N}$，其中$P^{i}=(P^i_x, P^i_y, P^i_w, P^i_h)$表示第$i$个样本的proposal中心的坐标$(P_x, P_y)$和proposal的长宽。对应的，$G^{i}=(G^i_x, G^i_y, G^i_w, G^i_h)$ 表示的是真实的标签的bbox的坐标和长宽，因此在这个回归任务中，希望学习到函数$f(\cdot)$，使得$f(P) \rightarrow G$。

![bbox_reg][bbox_reg]

<div align='center'>
    <b>
        Fig 3.1 bbox的回归和更新。
</div>

我们将这个$f(\cdot)$拆解成四个函数，分别是$d_x(P), d_y(P), d_w(P), d_h(P)$。最后我们有对bbox回归后的估计结果为:
$$
\begin{align}
\hat{G}_x &= P_wd_x(P)+P_x \\
\hat{G}_y &= P_hd_y(P)+P_y \\
\hat{G}_w &= P_w \exp(d_w(P)) \\
\hat{G}_h &= P_h \exp(d_h(P))
\end{align}
\tag{3.1}
$$
为了简单起见，这里的$d_{\star}(P)$采用的是线性回归，用池化层$Pool_{5}$的输出作为$P$的特征，也即是表示为$\phi_5(P)$，因此对于$\star = x,y,w,h$分别来说，都有$d_{\star}(P) = \hat{\mathbf{w}_{\star}}^T \phi_5 (P)$。

最后用loss去描述，为:
$$
\mathbf{w}_{\star} = \arg\min_{\hat{\mathbf{w}_{\star}}} \sum_{i}^{N}(t^{i}_{\star}-\hat{\mathbf{w}_{\star}}^T \phi_5(P^i))^2+\lambda||\hat{\mathbf{w}_{\star}}||^2
\tag{3.2}
$$





# Reference

[1]. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 580-587).

[2]. Uijlings J R R , K. E. A. van de Sande…. Selective Search for Object Recognition[J]. International Journal of Computer Vision, 2013, 104(2):154-171.

[3]. J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013.  

[4]. https://arxiv.org/abs/1311.2524

[5]. A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012 

[6]. K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint, arXiv:1409.1556, 2014. 

[7]. P. Felzenszwalb, R. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained part
based models. TPAMI, 2010.  





[odtask]: ./imgs/odtask.jpg

[rcnn]: ./imgs/rcnn.jpg
[selective_search]: ./imgs/selective_search.jpg
[norm]: ./imgs/norm.jpg
[alexnet]: ./imgs/alexnet.jpg

[bbox_reg]: ./imgs/bbox_reg.png





