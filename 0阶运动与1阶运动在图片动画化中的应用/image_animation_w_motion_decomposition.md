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

**注意：本文只是基于[1,2]文献的内容阐述思路，为了行文简练，去除了某些细节，如有兴趣，请读者自行翻阅对应论文细读。**

$\S$   **本文使用术语纪要**：

<1>. 指引视频(Guided Video)，驱动视频(Driving Video)：指的是给定的用于提供动作信息的视频，该视频负责驱动，引导图片的动态信息，这两个术语在本文中将会视场合混用。

<2>. 静态图(Source Image, Source Frame)：需要被驱动的图片，其主体类别通常需要和指引视频中的类别一致，主体身份可以不同。

<3>. 泰勒展开(Taylor Expansion)：将复杂的非线性函数通过展开的方式变换成若干项的线性组合。

<4>. 变形(deformation)：指的是通过某些控制点去操控一个图片的某些部位，使得图片像素发生移动或者插值，从而形成一定程度空间上变化。

<5>. 主体（entity）：指的是图片或者视频中的活动主体，该主体不一定是人体，也可能是其他任意的物体。这里要明确的是本文提到的 主体类别（entity category） 和 主体身份（entity identity），主体身份不同于类别，比如都是人脸，一个张三的人脸，而另一个是李四的人脸。

<6>. 稀疏光流图（Sparse Optical Flow Map）：表示不同帧之间，稀疏的关键点之间的空间变化，是一个向量场。

<7>. 密集光流图（Dense Optical Flow Map）：表示不同帧之间，每个像素之间的空间变化，是一个向量场。

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

通常来说，表征一个主体的运动信息可以通过密集光流图的方式表达，光流（optical flow）[5] 表示的是某个局部运动的速度和方向，简单地可以理解为在时间很短的两个连续帧的某个局部，相对应像素的变化情况。如Fig 1.4所示，如果计算(a)(b)两帧关于蓝色框内的光流，我们可以得到如同(c)所示的光流图，表征了这个“拔箭”动作的局部运动速度和方向，因此是一个向量场，我们通常可以用$\mathcal{F} \in \mathbb{R}^{H \times W \times 2}$表示，其中的$H \times W$表示的是局部区域的空间尺寸，维度2表示的是二维空间$(\Delta x, \Delta y)$偏移。如果该局部区域的每一个像素都计算光流图，那么得到的光流图就称之为 **密集光流图**（Dense Optical Flow Map），如Fig 1.4 (c)所示。密集光流图中的每一个像素对应的向量方向，提供了从一个动作转移到下一个动作所必要的信息，是图片动画化过程中的必需信息。

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

为了以后章节的讨论方便，我们给出图片动画化模型的基本结构，如Fig 1.6所示，需要输入的是驱动视频和静态图，静态图具有和驱动视频相同的主体类别（比如都是人）但是身份可以不同（比如是不同的人），期望生成具有和静态图相同身份和主体，动作和驱动视频一致的视频，通常是提取驱动视频中每帧的动作信息，结合静态图生成期望的视频帧，在拼接成最终的视频输出。

![image_animation_framework][image_animation_framework]

<div align='center'>
    <b>
        Fig 1.6 图片动画化的基本框图，其中Monkey-Net是一种图片动画化的模型，可以替换成其他模型。一般只需要输入驱动视频和某张具有同样主体类别的静态图，通过提取驱动视频中每一帧的动作信息，结合静态图就可以生成具有静态图主体身份和类别，而且具有驱动视频同一个动作的“生成视频”。
    </b>
</div>



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
文献[1,2,11]利用了一种无监督的关键点提取方法，这里简单介绍一下，为之后的章节提供铺垫。如Fig 2.3所示，对于输入的单帧RGB图片$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$来说，利用U-net [12]提取出$K$个热值图$H_k \in [0,1]^{H \times W}, k \in [1,\cdots,K]$，每$k$个热值图表示了第$k$个关节点的分布情况。当然，U-net的最后一层需要用softmax层作为激活层，这样解码器的输出才能解释为每个关键点的置信图(confidence map)。

![kps_detector][kps_detector]

<div align='center'>
    <b>
        Fig 2.3 用U-net分别提取静态图和驱动帧的关键点。
    </b>
</div>

然而，我们还需要从置信图中计算得到关键点的中心位置和关节点的方差[^1]（方差以超像素的角度，表示了对关键点预测的可靠性），因此用高斯分布去对置信图进行拟合，得到均值和方差。对于每个关键点的置信图$H_k \in [0,1]^{H \times W}$，我们有：
$$
\begin{aligned}
\mathbf{h}_k &= \sum_{p\in\mathcal{U}}H_k[p]p \\
\Sigma_k &= \sum_{p\in\mathcal{U}}H_k[p](p-\mathbf{h}_k)(p-\mathbf{h}_k)^{\mathrm{T}}
\end{aligned}
\tag{2.1}
$$
其中$\mathbf{h}_k \in \mathbb{R}^{2}$表示了第$k$个关键点的置信图的中心坐标，而$\Sigma_k \in \mathbb{R}$则是其方差。$\mathcal{U}$表示了图片坐标的集合，而$p\in\mathcal{U}$则是遍历了整个置信图。整个过程如Fig 2.4所示，通过式子(2.1)，最终需要将置信图更新为以高斯分布表示的形式，如(2.2)所示。
$$
H_k(\mathbf{p})=\dfrac{1}{\alpha}\exp(-(\mathbf{p}-\mathbf{h}_k)\Sigma_k^{-1}(\mathbf{p}-\mathbf{h}_k)) \\
\forall p\in\mathcal{U}
\tag{2.2}
$$
其中的$\alpha$为标准化系数。最终得到的置信图如Fig 2.4的右下图所示。

![gaussian_Hk][gaussian_Hk]

![confidence_map][confidence_map]

<div align='center'>
    <b>
        Fig 2.4 通过高斯分布拟合，从置信图中计算关键点的中心位置和方差。
    </b>
</div>

至今，我们描述了如何提取关键点，但是这个关键点还没有经过训练，因此其输出还是随机的，不要担心，我们后续会一步步介绍如何进行无监督训练。不过这一章节就此为止吧，为了后续章节的方便，我们先假设我们的 **关键点提取是经过训练的，可以提取出较为完美的关键点** 。

# 稀疏光流图

在引入动作分解的概念之前，我们先花时间去讨论下稀疏光流图。如Fig 3.1所示，假设我们有训练好的关键点检测器，表示为$\Delta$，那么输入同一个视频中的不同两帧（我们在后面会解释为什么在训练时候是输入同一个视频的不同两帧），其中用$\mathbf{x}$表示静态图，用$\mathbf{x}^{\prime}$表示驱动视频（训练时候是和静态图一样，出自同一个视频）中的其中一帧，那么，检测出的关键点可以表示为:
$$
\begin{aligned}
H &= \Delta(\mathbf{x}) \\
H^{\prime} &= \Delta(\mathbf{x}^{\prime})
\end{aligned}
\tag{3.1}
$$
那么，自然地，这两帧之间的对应关键点的相对变化可以简单地用“代数求差”表示，为：
$$
\dot{H} = H^{\prime}-H
\tag{3.2}
$$
这里的$\dot{H}$称之为稀疏光流图，其表示了稀疏关键点在不同帧之间的空间变化，其中每一个关键点的光流表示为$h_k = [\Delta x, \Delta y]$。可知$\dot{H} \in \mathbb{R}^{K \times 2}$，其中$K$是关键点的数量。

![kps_detector_more][kps_detector_more]

<div align='center'>
    <b>
        Fig 3.1 对关键点进行相减，得到稀疏光流图。
    </b>
</div>
但是得到稀疏光流图只能知道关键点是怎么位移形变的，我们该怎么求出关键点周围的像素的位移变化数据呢？

# 动作分解与泰勒展开

知道了稀疏光流图，我们只知道关键点是怎么变化的，但是对关键点周围的像素的变化却一无所知，我们最终期望的是通过稀疏光流图去推理出密集光流图，如Fig 4.1所示。

![dense_motion_network][dense_motion_network]

<div align='center'>
    <b>
        Fig 4.1 通过稀疏光流图去生成密集光流图。
    </b>
</div>

为了实现这个过程，我们需要引入先验假设，而最为直接的先验假设就是动作分解。

## 零阶动作分解

一种最简单的动作分解假设就是：

> 每个关键点周围的主体部件是局部刚性的，因此其位移方向和大小与关键点的相同，我们称之为动作零阶分解。

这个假设通过Fig 4.2可以得到很好地描述，我们通过关键点检测模型可以检测出对应的关键点位移，根据假设，那么周围的身体部分，如橘色点虚线框所示，是呈现刚体变换的，也就是说该区域内的所有和主体有关的部分的像素的位移向量，都和该关键点相同。

![locally_rigid][locally_rigid]

<div align='center'>
    <b>
        Fig 4.2 关键点周围部件的局部刚体性质：绿色虚线表示的是蓝色关键点周围的主体部件的密集光流，因为假设了关键点周围的刚体性，因此绿色虚线和黄色虚线的大小方向都相同。
    </b>
</div>

那么现在问题就在于，这里谈到的每个关键点的“周围区域”到底有多大，才会使得刚体性质的假设成立。于是问题变成去预测对于每个关节点来说，能使得刚体性质成立的区域了。对于每个关键点，我们通过神经网络预测出一个掩膜$M_k \in \mathbb{R}^{H \times W}$，那么我们有：
$$
\mathcal{F}_{\mathrm{coarse}} = \sum_{k=1}^{K+1} M_k \otimes \rho(h_k)
\tag{4.1}
$$
其中的$\rho(\cdot)$表示对每个关键点的光流重复$H \times W$次，得到$\rho(\cdot)\in\mathbb{R}^{H \times W \times 2}$的张量，该过程如Fig 4.3所示，当然这里用箭头的形式表示了光流向量，其实本质上是一个$\mathbb{R}^2$的向量；而$\otimes$表示逐个元素的相乘。

![repeat_hw][repeat_hw]

<div align='center'>
    <b>
        Fig 4.3 通过在空间上复制每个关键点的光流向量，得到了K个光流图，每个光流图都是单个关键点的简单空间复制。
    </b>
</div>

通常这个掩膜$M_k$通过U-net去进行学习得到，这里的U-net也即是Fig 4.1中的Dense Motion Network，用符号$M$表示，其设计的初衷是可以对某个关键点$k$呈现刚体区域进行显著性高亮，如Fig 4.4所示，并且为了考虑相对不变的背景，实际上需要学习出$K+1$个掩膜，其中一个掩膜用于标识背景，同时也需要$\rho([0,0])$用于表示背景区域不曾出现位移。

![coarse_flow][coarse_flow]

<div align='center'>
    <b>
        Fig 4.4 通过用掩膜去提取出每个关键点的具有局部刚体性质的区域。
    </b>
</div>

除了掩膜之外，模块$M$同样需要预测$\mathcal{F}_{\mathrm{residual}}$，作为$\mathcal{F}_{\mathrm{coarse}}$的补充，其设计的初衷是预测某些非刚体性质的变换，非刚体性质的变换不能通过之前提到的分割主体部分然后进行掩膜的方法得到，因此需要独立出来，通过网络进行预测。于是我们有：
$$
\mathcal{F} = \mathcal{F}_{\mathrm{coarse}}+\mathcal{F}_{\mathrm{residual}}
\tag{4.2}
$$
现在Dense Motion Network的框图如Fig 4.5所示，我们以上阐述了该模块的输出，现在考虑这个模块的输入。输入主要有稀疏光流图$\dot{H}$和静态图$\mathbf{x}$，然而在整个优化过程中，由于$\mathcal{F}$其实是和$\mathbf{x}^{\prime}$对齐的，而输入如果只是$\mathbf{x}$的信息，那么就可能存在优化过程中的困难，因为毕竟存在较大的差别，因此需要显式地先对输入静态图进行一定的变形，可以用双线性采样（Bilinear Sample）进行，记$f_{w}(\cdot)$为双线性采样算符，我们有：
$$
\mathbf{x}_k = f_w(\mathbf{x}, \rho(h_k))
\tag{4.3}
$$
其中的$\mathbf{x}_k$是根据$\rho(h_k)$只对每个关键点光流进行变形形成的，将$\dot{H}$和$\{\mathbf{x}_k\}_{k=1,\cdots,K}$以及$\mathbf{x}$在通道轴进行拼接，然后作为U-net的输入。

![dense_motion_network_Module][dense_motion_network_Module]

<div align='center'>
    <b>
        Fig 4.5 Dense Motion Network的框图，其输入需要考虑密集光流图的对齐问题。
    </b>
</div>

## 一阶动作分解

零阶动作分解的假设还是过于简单了，即便是关键点局部区域也不一定呈现出良好的刚体性质，在存在柔性衣物的影响下更是如此，因此引入了一阶动作分解的假设，除了引入的基本假设不同之外，模型其他大部分和零阶动作分解类似。在一阶动作分解下，基本假设变成了

> 每个关键点周围的主体部件是局部仿射变换[13]的，我们称之为一阶动作分解。







# 变形模型





# 端到端无监督训练



# 论缺陷



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

[12]. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention* (pp. 234-241). Springer, Cham.

[13]. https://blog.csdn.net/LoseInVain/article/details/108454304

[14]. 





[image_animation]: ./imgs/image_animation.png
[vox-teaser]: ./imgs/vox-teaser.gif

[fashion-teaser]: ./imgs/fashion-teaser.gif
[optical_flow]: ./imgs/optical_flow.png
[sparse_kp_mapping]: ./imgs/sparse_kp_mapping.png
[face-alignment-adrian]: ./imgs/face-alignment-adrian.gif
[mgif-teaser]: ./imgs/mgif-teaser.gif

[kps_detector]: ./imgs/kps_detector.png
[gaussian_Hk]: ./imgs/gaussian_Hk.png
[confidence_map]: ./imgs/confidence_map.png
[kps_detector_more]: ./imgs/kps_detector_more.png

[image_animation_framework]: ./imgs/image_animation_framework.png
[dense_motion_network]: ./imgs/dense_motion_network.png
[locally_rigid]: ./imgs/locally_rigid.png
[repeat_hw]: ./imgs/repeat_hw.png
[coarse_flow]: ./imgs/coarse_flow.png
[dense_motion_network_Module]: ./imgs/dense_motion_network_Module.png









[^1]: 这里采用高斯分布拟合的目的还有一个就是，在无监督训练开始时，其预测结果是随机的，将其用高斯分布去拟合，才能给后续的优化提供方便。

