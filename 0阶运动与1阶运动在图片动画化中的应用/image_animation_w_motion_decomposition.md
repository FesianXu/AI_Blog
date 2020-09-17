<div align='center'>
    运动的零阶分解与一阶分解以及在图片动画化中的应用（The 0th-order and first-order decomposition of motion and the application in image animation）
</div>

<div align='right'>
    FesianXu 2020/09/16 at UESTC
</div>

# 前言

最近基于AI的换脸应用非常的火爆，同时也引起了新一轮的网络伦理大讨论。如果光从技术的角度看，对于视频中的人体动作信息，通常可以通过泰勒展开分解成零阶运动信息与一阶运动信息，如文献[1,2]中提到的，动作的分解可以为图片动画化提供很好的光流信息，而图片动画化是提供换脸技术的一个方法。笔者在本文将会根据[1,2]文献中的内容，对其进行笔记和个人理解的探讨。 **如有谬误请联系指出，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

github: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

----

$\S$   **本文使用术语纪要**：

<1>. 指引视频(Guided Video)，驱动视频(Driving Video)：指的是给定的用于提供动作信息的视频，该视频负责驱动，引导图片的动态信息，这两个术语在本文中将会视场合混用。

<2>. 静态图：需要被驱动的图片，其主体类别通常需要和指引视频中的类别一致，主体身份可以不同。

<3>. 泰勒展开(Taylor Expansion)：将复杂的非线性函数通过展开的方式变换成若干项的线性组合。

<4>. 变形(deformation)：指的是通过某些控制点去操控一个图片的某些部位，使得图片像素发生移动或者插值，从而形成一定程度空间上变化。

<5>. 主体（entity）：指的是图片或者视频中的活动主体，该主体不一定是人体，也可能是其他任意的物体。这里要明确的是本文提到的 主体类别（entity category） 和 主体身份（entity identity），主体身份不同于类别，比如都是人脸，一个张三的人脸，而另一个是李四的人脸。

# 从图片动画化说起

我们知道最近的基于AI的换脸应用非常火爆，也引起了一轮轮关于AI使用的伦理大讨论，这从侧面反映了AI技术应用在我们日常生活的渗透。如Fig 1.1所示，给定一个指引视频，让一张静态图片跟随着该视频表演其中的表情（或者动作），这种技术的主要框架在于需要分离指引视频中的动作信息（motion）和外观信息（appearance），将提取出的动作信息以某种形式添加到静态图上，让静态图达到一定程度的变形（deformation），以达到图片动态化表演特定动作的目的。

![vox-teaser][vox-teaser]

<div align='center'>
    <b>
        Fig 1.1 换脸的例子。给定一个静态的图片和一段相对应主体类别的（比如同样是人脸）的指引视频，我们可以让图片跟着视频一起动起来，达到以假乱真的目的。原图出自[10]。
    </b>
</div>

![fashion-teaser][fashion-teaser]

<div align='center'>
    <b>
        Fig 1.2 不仅仅是人脸替换，图片动画化可以应用在主体类别一致的其他主体中。原图出自[10]。
    </b>
</div>



这类型的工作可以称之为**图片动画化** （image animation），指的是给定一张具有某个主体的静态图（Source Image）（主体不一定是人体，如Fig 1.2所示，不过我们这里主要以人体做例子），再给定一个由该主体表演某个动作的视频，一般称之为驱动视频（Driving Video），让静态图跟随着驱动视频的动作“活动”起来。注意到静态图和驱动视频中的主体是同一类型的主体，但是身份可能是不同的，比如都是人脸，但是不是同一个人的人脸。如Fig 1.3所示，给定了一个驱动视频，其主体是一个人脸的表情序列，给定了一个静态图，主体是一个不同身份的人，然后任务期望提取出序列中的动作信息，期望以某种方法添加到静态图上，使得静态图可以通过像素变形的方式，形成具有指定动作，但是主体身份和静态图一致的新的动作序列。

![image_animation][image_animation]

<div align='center'>
    <b>
        Fig 1.3 图片动画化的例子。给定一个驱动视频，和若干不同的静态图，从驱动视频中提取出人脸的运动信息，将运动信息以某种方式“添加”到静态图上，以达到让静态图跟随着驱动视频活动起来的目的。
    </b>
</div>

当然，该任务不一定被局限在人脸上，如Fig 1.2所示，事实上，只要输入驱动视频和静态图的主体类别一致，就可以通过某些自监督的方法进行动作信息提取，并且接下来进行动作信息迁移到目标静态图上的操作。

我们现在已经对图片动画化有了基本的认识，那么从技术上看，这整个任务的难点在于哪儿呢？主要在于以下几点：

1. 如何表征运动信息？
2. 如何提取驱动视频中的运动信息？
3. 如何将提取到的动作信息添加到静态图中，让静态图变形？

通常来说，表征一个主体的运动信息可以通过密集光流图的方式表达，光流（optical flow）[5] 表示的是某个局部运动的速度和方向，简单地可以理解为在时间很短的两个连续帧的某个局部，相对应像素的变化情况。如Fig 1.4所示，如果计算(a)(b)两帧关于蓝色框内的光流，我们可以得到如同(c)所示的光流图，表征了这个“拔箭”动作的局部运动速度和方向，因此是一个向量场，我们通常可以用$\mathbf{F} \in \mathbb{R}^{H \times W \times 2}$表示，其中的$H \times W$表示的是局部区域的空间尺寸，维度2表示的是二维空间$(\Delta x, \Delta y)$偏移。如果该局部区域的每一个像素都计算光流图，那么得到的光流图就称之为 **密集光流图**（Dense Optical Flow Map），如Fig 1.4 (c)所示。密集光流图中的每一个像素对应的向量方向，提供了从一个动作转移到下一个动作所必要的信息，是图片动画化过程中的必需信息。

![optical_flow][optical_flow]

<div align='center'>
    <b>
        Fig 1.4 （a）（b）两张图是连续的两帧，而（c）是对蓝色框区域进行计算得到的光流图，表征了运动的速度和方向。原图出自[4]。
    </b>
</div>

如果能够给出某个运动的密集光流图，那么就可以根据每个像素对应在光流图中的向量方向与大小对像素进行位移插值后，实现图像的变形的过程。然而，在图片动画化过程中，我们的输入通常如Fig 1.5所示，其静态图和驱动视频中的某一帧（称之为驱动帧）之间的动作差别很大，而且主体的身份还不一定一致，能确定的只有 一点，就是： **稀疏的关键点可以视为是一一配对的**。 如Fig 1.3所示，蓝色点是人体的稀疏关键点，通常存在一对一的配对映射（暂时不考虑遮挡），如黄色虚线所示，这种稀疏关键点的映射图，我们称之为 **稀疏光流图** （Sparse Optical Flow Map）。我们接下来介绍的文章，都是 **从不同方面考虑从稀疏光流图推理出密集光流图，从而指引图片变形的。 **

![sparse_kp_mapping][sparse_kp_mapping]

<div align='center'>
    <b>
        Fig 1.5 通常输入的是差别较大的静态图（Source）和驱动视频中的某一帧（Driving Frame），蓝点表示的是稀疏的关键点，黄色虚线表示的是对应关键点的配对。
    </b>
</div>

到此为止，我们之前讨论了如何定义一个动作的运动信息，也就是用密集光流图表示。同时，我们也分析了一种情况，在实际任务中，很难直接得到密集光流图，因此需要从一对一配对的稀疏光流图中加入各种先验知识，推理得到密集光流图。我们接下来的章节讨论如何添加这个先验知识。



# 无监督关键点提取

在继续讨论密集光流图提取之前，我们首先描述下如何提取稀疏光流信息，也即是稀疏的关键点信息，如Fig 1.5所示。当然，对于人体而言，目前已经有很多研究可以进行人体姿态估计，比如OpenPose [6]，AlphaPose [7]等，这些研究可以提取出相对不错的人体关键点。就人脸这块的关键点提取而言，也有很多不错的研究[8]，可以提取出相对不错的人脸稀疏或者密集关键点，如Fig 2.1所示。

![face-alignment-adrian][face-alignment-adrian]

<div align='center'>
    <b>
        Fig 2.1 人脸关键点提取方法已经日渐成熟，效果已经相当不错了，原图出自[9]。
    </b>
</div>

但是，我们注意到，为了提取人体或者人脸的关键点，目前的大多数方法都需要依赖于大规模的人体/人脸标注数据集，这个工作量非常大，因此，假如我们需要对某些非人脸/人体的图片进行图片动画化，比如Fig 2.2所示的动画风格的马，我们将无法通过监督学习的方式提取出关键点，因为没有现存的关于这类型数据的数据集。为了让图片动画化可以泛化到人体/人脸之外更广阔的应用上，需要提出一种无监督提取特定主体物体关键点的方法。

![mgif-teaser][mgif-teaser]

<div align='center'>
    <b>
        Fig 2.2 动画风格的马的图片动画化，现存的研究并没有关于这类型数据的关键点提取，也没有相关的数据集，因此依赖于监督学习，在这种数据中提取关键点是一件困难的问题。原图出自[10]。
    </b>
</div>

文献[1,2,11]利用了一种无监督的关键点提取方法，这里简单介绍一下，为之后的章节提供铺垫。





# 零阶动作分解











# 一阶动作分解







# 变形模型





# Reference

[1]. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., & Sebe, N. (2019). Animating arbitrary objects via deep motion transfer. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 2377-2386).

[2]. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., & Sebe, N. (2019). First order motion model for image animation. In *Advances in Neural Information Processing Systems* (pp. 7137-7147).

[3]. https://blog.csdn.net/LoseInVain/article/details/108483736

[4]. Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action recognition in videos. In *Advances in neural information processing systems* (pp. 568-576).

[5]. https://en.wikipedia.org/wiki/Optical_flow

[6]. Cao Z , Hidalgo G , Simon T , et al. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, PP(99):1-1.

[7]. https://github.com/MVIG-SJTU/AlphaPose

[8]. Bulat, Adrian , and G. Tzimiropoulos . "How Far are We from Solving the 2D & 3D Face Alignment Problem? (and a Dataset of 230,000 3D Facial Landmarks)." *IEEE International Conference on Computer Vision* IEEE Computer Society, 2017.

[9]. https://github.com/1adrianb/face-alignment

[10]. https://github.com/AliaksandrSiarohin/first-order-model

[11]. Jakab, T., Gupta, A., Bilen, H., & Vedaldi, A. (2018). Unsupervised learning of object landmarks through conditional image generation. In *Advances in neural information processing systems* (pp. 4016-4027).

[12]. 





[image_animation]: ./imgs/image_animation.png
[vox-teaser]: ./imgs/vox-teaser.gif

[fashion-teaser]: ./imgs/fashion-teaser.gif
[optical_flow]: ./imgs/optical_flow.png
[sparse_kp_mapping]: ./imgs/sparse_kp_mapping.png
[face-alignment-adrian]: ./imgs/face-alignment-adrian.gif
[mgif-teaser]: ./imgs/mgif-teaser.gif





