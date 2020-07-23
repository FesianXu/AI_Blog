<div align='center'>
    双目三维重建——层次化重建思考
</div>

<div align='right'>
    FesianXu 2020.7.22 at ANT FINANCIAL intern
</div>

# 前言

本文是笔者阅读[1]第10章内容的笔记，本文从宏观的角度阐述了双目三维重建的若干种层次化的方法，包括投影重建，仿射重建和相似性重建到最后的欧几里德重建等。本文作为介绍性质的文章，只提供了这些方法的思路，并没有太多的细节，细节将会由之后的博文继续展开。如有谬误，请联系作者指出，转载请注明出处。

$\nabla$ 联系方式：
**e-mail**:      [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**:             973926198
github:       https://github.com/FesianXu

----

**注**: 在阅读本文之前，强烈推荐读者先阅读[4]和[5]， 以了解[几何变换的层次——投影变换，仿射变换，度量变换和欧几里德变换的具体区别]和[conic圆锥线和quadric二次曲锥面的定义和应用]，在本文中将会基于这些前置知识进行讨论。同时作为成像的基础知识，相机内外参数的知识[6]也是必须了解的。

# 双目三维重建简介

作为三维重建，我们希望能够得到被重建物体的结构（也就是三维世界中的点的位置信息）。在双目三维重建中，正如其“双目”所言，我们假设有两个摄像机对某个物体进行观测，得到了多视角对同一个物体描述的图片。通常，在三维重建任务中，我们通过一系列的前置算法，可以得到这些多视角图片之间的对应点关系，如Fig 1所示，点$\mathbf{p}_l$和点$\mathbf{p}_r$都是三维世界客体点$\mathbf{P}$的投影，也就是说它们是一对 **对应点**（correspondence），通常我们用$\mathbf{x}_i \leftrightarrow \mathbf{x}_i^{\prime}$表示一对对应点。笔者之前的博文[2,3]中曾经根据对极线约束和图像矫正对对应点进行过介绍，读者有兴趣可自行查阅。

![stereovision][stereovision]

<div align='center'>
    <b>
        Fig 1. 双目多视角图片之间的对应点，其都是对三维世界中客体点P的投影。
    </b>
</div>

通常来说，在三维重建任务中，我们假设对应点对应的三维点$\mathbf{P}$的位置是未知的，需要我们求出，并且我们也不知道相机的方向，位置和内参数矫正矩阵等（也就是外参数和内参数都未知）。整个重建任务就是需要去计算相机矩阵$\mathrm{P}$和$\mathrm{P}^{\prime}$，使得对于三维点$\mathbf{X}_i$有：
$$
\mathbf{x}_i = \mathrm{P} \mathbf{X}_i,   \mathbf{x}_i^{\prime} = \mathrm{P}^{\prime} \mathbf{X}_i \\ \forall i
\tag{1}
$$
其中的$i$表示的是对应点的编号。当给定的对应点太少时，这个任务是显然不能完成的，但是如果给定了足够多的对应点，那么我们就有足够的约束去唯一确定出一个基础矩阵（Fundamental Matrix）[2] 出来，如式子(2)所示（当然，求解这个基础矩阵也并没有那么简单，这个并不在本文中进行阐述）。此时，整个三维重建将会存在一种称之为 **投影歧义** （projective ambiguity）的现象，我们将会在下文进行介绍。作为无矫正的相机来说，在不引入任何先验知识进行约束的情况下，这是双目三维重建能得到的最好结果。在添加了其他对场景的先验知识后（比如平行线约束，垂直线约束，摄像机内参数相同假设等），投影歧义可以被减少到仿射歧义甚至是相似歧义的程度。
$$
\mathbf{x}^{\mathrm{T}} \mathcal{F} \mathbf{x}^{\prime} = 0
\tag{2}
$$
总的来说，重建过程的三部曲可以分为：

1. 根据对应点计算得到基础矩阵。
2. 根据基础矩阵计算得到相机矩阵$\mathrm{P},\mathrm{P}^{\prime}$。
3. 对于每对对应点$\mathbf{x}_i \leftrightarrow \mathbf{x}_i^{\prime}$来说，通过三角测量法，计算其在空间上的三维点坐标位置。

注意到本文对这些重建方法的介绍只是一个概念性的方法，读者需要明晰的是，不要尝试单纯地实现本文介绍的方法去实现重建，对于真实场景的图片而言，重建过程存在着各种噪声（比如对应点对可能不准确），这些具体的方法我们将会在之后的博文介绍。

这里需要单独拎出来提一下的是 **三角测量法**（Triangulation），如Fig 2所示，在计算得到了相机矩阵$\mathrm{P},\mathrm{P}^{\prime}$之后，我们知道$\mathbf{x},\mathbf{x}^{\prime}$满足对极约束$\mathbf{x}^{\mathrm{T}} \mathcal{F} \mathbf{x}^{\prime} = 0$，换句话说，我们知道$\mathbf{x}$在对极线$\mathcal{F}\mathbf{x}^{\prime}$上，反过来这意味着从图像点$\mathbf{x},\mathbf{x}^{\prime}$反向投影得到的射线共面，因此它们的反向射线将会交于一点$\mathbf{X}$。通过三角测量的方法，我们可以测量除了基线上的任意一个三维空间点，原因在于基线上的点的反向射线是共线的，因此不能唯一确定其相交点。

![triangulation][triangulation]

<div align='center'>
    <b>
        Fig 2. 通过三角测量法去确定三维空间中的点。
    </b>
</div>


# 重建歧义性

单纯地从对应点去进行场景，物体的三维重建必然是有一定的歧义性的，只是说引入了对场景一定的先验知识后，这种歧义性会得到缓解。举个例子，光从成对的对应点对（甚至可能是多个视角的点对），都不能计算出场景的绝对位置（指的是地球上的经纬度）和绝对朝向，这点很容易理解，例如Fig 3所示，即便是给定了相机内参数等，也不可能决定b和c这两个走廊的具体的东西走向，还是南北走向，亦或是其经纬度，这些涉及到地理位置的绝对信息无法光从相机重建得到。

![corridor][corridor]

<div align='center'>
    <b>
        Fig 3. 不引入其他任何知识，只从相机得到的照片，无法判断场景的绝对地理信息。
    </b>
</div>

一般来说，基于相机的重建，我们称它最好的情况下，对于世界坐标系来说，都只能是欧几里德变换（包括这旋转和偏移）。当然，如果我们的相机没有标定，也就是内参数未知，就Fig 3的走廊为例子，我们无法确定走廊的宽度和长度，它可能是3米，也可能只是一个玩具走廊，只有30厘米，这些都有可能。在不引入对场景尺度的任何先验而且相机没有标定的情况下，我们称基于相片的场景重建只能最好到相似性变换（也就是是存在旋转，偏移和尺度缩放）。

如果用数学形式去解释这个现象，用$\mathbf{X}_i$表示一系列场景中的三维点，$\mathrm{P},\mathrm{P}^{\prime}$表示一对相机，其将三维点投影到$\mathbf{x}_i, \mathbf{x}_i^{\prime}$。假设我们有相似性变换$\mathrm{H}_S$
$$
\mathrm{H}_S = 
\left[\begin{matrix}
\mathrm{R} & \mathbf{t} \\
\mathbf{0}^{\mathrm{T}} & \lambda
\end{matrix}\right]
\tag{3}
$$
其中的$\mathrm{R}$是旋转矩阵，$\mathbf{t}$是偏移，$\lambda^{-1}$是尺度放缩。假设我们对三维点进行相似性变换，那么我们用$\mathrm{H}_S\mathbf{X}_i$取代$\mathbf{X}_i$，并且用$\mathrm{P}\mathrm{H}_S^{-1}$和$\mathrm{P}^{\prime}\mathrm{H}_S^{-1}$取代原来的相机参数$\mathrm{P},\mathrm{P}^{\prime}$。我们发现，因为有$\mathrm{P}\mathbf{X}_i = (\mathrm{P}\mathrm{H}_S^{-1})(\mathrm{H}_S \mathbf{X}_i)$，因此在图像上的投影点位置是不会改变的。 **通常来说，在不引入其他先验的情况下，我们会发现，在重建过程中，算法只会保证图像上的投影点的位置是投影正确的，没法保证三维空间的其他信息了。**



## 相似歧义性

更进一步，我们对相机参数进行分解，有$\mathrm{P} = \mathrm{K}[\mathrm{R}_P | \mathbf{t}_P]$，那么经过相似性变换之后，有：
$$
\mathrm{P}\mathrm{H}_S^{-1} = \mathrm{K} [\mathrm{R}_P\mathrm{R}^{-1} | \mathbf{t}^{\prime}]
\tag{4}
$$
一般情况下，我们不是很关心偏移$\mathbf{t}^{\prime}$。我们会发现，相似性变换并不会改变其相机内参数，$\mathrm{K}$是不会改变的，也就是说，即便是对于矫正后的相机，最好的重建结果也会存在相似性歧义，我们称之为 **相似性重建** (Similarity reconstruction, metric reconstruction)。 如Fig 4的图a所示，这就是相似性歧义的示意图，我们无法确定场景的绝对大小。

![proj_ambiguity][proj_ambiguity]

<div align='center'>
    <b>
        Fig 4. 重建的相似性歧义性和投影歧义性
    </b>
</div>
## 投影歧义性

如果我们对内参数一无所知，也不知道相机之间的相对位置关系，那么整个场景的重建将会陷入 **投影歧义性** (projective ambiguity)，如Fig 4的图b所示。同样我们可以假设一个不可逆矩阵$\mathrm{H} \in \mathbb{R}^{4 \times 4}$作为投影矩阵，用我们在之前介绍的方法，我们会发现将投影矩阵同时作用在三维点和相机上时，不影响图像上的投影点位置。因此实际的重建三维点将会是投影歧义的。这个称之为 **投影重建** (Projective reconstruction)，投影重建和相似重建的不同之处在于，相似重建因为相机内参数已经知道，因此相机焦点位置是确定的，而投影重建因为没有矫正相机参数，因此焦点位置可能会变化，Fig 4示意图就明确了这一点。



## 仿射歧义性

如果两个相机只是存在偏移上的变化，而内参数完全相同，那么重建过程是最好能到 **仿射重建** （affine reconstruction），相对应的，其会引入 **仿射歧义性** （affine ambiguity）。也就是说，我们知道不同相机之间的焦距都是一样的（因为只存在偏移），因此整个场景可能存在旋转，偏移和尺度放缩或者切变（Shear）[7]。









# Reference

[1]. Hartley R, Zisserman A. Multiple view geometry in computer vision[M]. Cambridge university press, 2003. Chapter 10

[2]. https://blog.csdn.net/LoseInVain/article/details/102665911

[3]. https://blog.csdn.net/LoseInVain/article/details/102775734

[4]. https://blog.csdn.net/LoseInVain/article/details/104533575

[5]. https://blog.csdn.net/LoseInVain/article/details/104515839

[6]. https://blog.csdn.net/LoseInVain/article/details/102632940

[7]. https://blog.csdn.net/LoseInVain/article/details/102756630

[8]. 



[stereovision]: ./imgs/stereovision.jpg
[triangulation]: ./imgs/triangulation.jpg

[corridor]: ./imgs/corridor.jpg
[proj_ambiguity]: ./imgs/proj_ambiguity.jpg





