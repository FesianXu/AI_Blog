<div align='center'>
    《土豆学OD》 之 fast RCNN初探
</div>

[TOC]

----

# 前言

在上一篇[1]文章中我们谈到了RCNN，提到了基于proposal和深度学习中的卷积神经网络进行物体识别的思路，在本文中，我们继续改进RCNN，对fast RCNN进行介绍，其具有更好的性能和更快的计算速度。本文如有纰漏，请联系指出，谢谢。

*联系方式：*
**e-mail**: `FesianXu@gmail.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

----

# RCNN的缺陷

RCNN[2]中的流程如Fig 1.1所示，我们首先需要用Selective Search提取若干个proposal侯选框，然后将每个侯选框都喂给CNN网络进行特征提取，最后每一个侯选框都经过多分类SVM进行类别分类，最后进行bbox的回归和非极大抑制得到最终结果。

![rcnn][rcnn]

<div align='center'>
    <b>
     	Fig 1.1 RCNN的pipeline流程图。
    </b>
</div>

然而，这个过程存在着很多问题，可以改进以得到更好的效果和更快的计算，列举如下：

1. RCNN的训练过程不是端到端的，而是分为三个部分分别训练，使得每个部分的最优不一定能达到整个系统的最优。
2. 每个proposal都得经过CNN提取特征，效率极其低下。

对这些问题进行改进，我们就有了fast RCNN[3].



# fast RCNN

fast RCNN [3] 的pipeline很简单，如Fig 2.1所示。

![fast_rcnn][fast_rcnn]

<div align='center'>
    <b>
     	Fig 2.1 fast RCNN的pipeline示意图。
    </b>
</div>

其基本思想就是将整幅图像通过CNN提取整个图片的特征图，然后将原图上通过selective search得到的proposal映射到特征图上，得到在特征图上的proposal后，因为每个proposal的尺寸不一，其在特征图上的尺寸也是不一样的，因此需要用`ROI pooling`层进行处理，最后末端是两个任务分支，一个是对proposal的类别分类，另一个则是对bbox的回归。

通过这种方法，解决了RCNN中被诟病的多阶段训练和proposal特征提取低效的问题，接下来我们分别介绍fast RCNN的细节。













-----

# Reference

[1]. https://blog.csdn.net/LoseInVain/article/details/98054030

[2]. Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

[3]. Girshick R. Fast r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1440-1448.





[rcnn]: ./imgs/rcnn.jpg
[fast_rcnn]: ./imgs/fast_rcnn.jpg



