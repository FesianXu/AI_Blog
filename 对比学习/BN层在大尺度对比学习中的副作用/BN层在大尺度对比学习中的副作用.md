<div align='center'>
    Batch Norm层在大尺度对比学习中的过拟合现象及其解决
</div>

<div align='right'>
    FesianXu 20210830 at Baidu Search Team
</div>

# 前言

在之前的博文[1,2]中已经说明了在对比学习中提高batch size的巨大作用，然而在大尺度对比学习的训练过程中，被广泛实践证明有效的Batch Norm层则很容易出现过拟合的现象。笔者在本文对该现象进行笔记，并且纪录其解决方案。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]



----

在大尺度的对比学习训练过程中，Batch Norm层所造成的过拟合现象被众多公开论文所报道过，比如MoCo[3],  SimCLR[4]和其他一些工作[5]。之前笔者在MoCo的笔记中也简单谈到过这个问题[1]，然而当时尚未深究，现在在工作中实际遇到了这个问题，就权当笔记将其纪录。

在大尺度的对比学习中，一种常见的实践是：设置一个较大的batch size，比如4096，显然一张GPU卡很难塞下，特别是在多模态模型中，因此通过数据并行将大batch size均分到不同卡上，比如16张卡。在双塔模型中，我们需要对两个塔输出的特征进行计算得到打分矩阵，如Fig 1所示。然而分布在不同卡上的双塔特征$\mathbf{f}_i \in \mathbb{R}^{256 \times D}$的批次大小为256，如果对每个卡上的双塔特征进行打分，那么得到的打分矩阵$\mathbf{S} \in \mathbb{R}^{256 \times 256}$。可以看到此时多GPU并没有真正意义上增大batch size，只是通过数据并行的方式提高了训练速度而已，我们期望中的打分矩阵应该是$\mathbf{S} \in \mathbb{R}^{4096 \times 4096}$，在这种大尺度的打分矩阵上能更高效地获得足够好的负样本进行对比学习。

![batchneg][batchneg]

<div align='center'>
    <b>
        Fig 1. Batch Negative的方式从一个batch中构造负样本。
    </b>
</div>

一种实现这种方式的实践是通过各个深度学习框架提供的`all_gather`机制，比如`pytorch`中的`torch.distributed.all_gather()` [6]或者`paddle`中的`paddle.distributed.all_gather()` [7]。这些函数可以从所有的GPU中汇聚某个矩阵，比如特征矩阵$\mathbf{f}_i \in \mathbb{R}^{256 \times D}$，将$i=0,\cdots,15$所有的GPU中的特征矩阵进行汇聚，可以得到$\mathbf{f}_{gather} \in \mathbb{R}^{4096 \times D}$的汇聚特征矩阵。我们可以对这个$\mathbf{f}_{gather}$进行打分，然后进行对比学习的训练。

然而，这只是大尺度对比学习的第一步，我们发现在汇聚特征之前，特征的计算都是在各自的GPU中进行计算的，假如模型中具有Batch Norm层，那么其统计参数$\mu, \sigma^2$都是在各自的GPU中计算的（假设是异步Batch Norm机制），和Batch Norm有关的知识可见之前博文[8]。而因为BN层的统计参数和`all_gather`机制，会导致在大尺度对比学习训练过程中的严重过拟合现象。且让笔者慢慢道来。







# Reference

[1]. https://fesian.blog.csdn.net/article/details/119515146

[2]. https://fesian.blog.csdn.net/article/details/119516894

[3]. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

[4]. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations. In *International conference on machine learning* (pp. 1597-1607). PMLR.

[5]. Hénaff, O. J., Razavi, A., Doersch, C., Eslami, S., and Oord, A.v. d. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019.  

[6]. https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather

[7]. https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/all_gather_cn.html

[8]. https://blog.csdn.net/LoseInVain/article/details/86476010









[qrcode]: ./imgs/qrcode.jpg
[batchneg]: ./imgs/batchneg.png