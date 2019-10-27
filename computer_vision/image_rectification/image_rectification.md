<div align='center'>
    图像校正——使得在对极线上寻找对应点更加容易
</div>

<div align='right'>
    2019/10/27 FesianXu
</div>

[TOC]

-----

# 前言

我们在[1]中曾经谈到了如何在对极线上去寻找对应点，这样会使得算法更鲁棒，而且速度更快。在本文中，我们将会继续介绍一种称之为图像矫正的方法，通过这种方法，我们可以在对极线的基础上，使得寻找对应点变得更为容易。

**如有谬误，请联系指正。转载请注明出处。**

 $\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu 

----

# 为什么我们需要图像矫正

我们在[1]曾经聊到了在对极线上寻找对应点的方法，如Fig 1.1所示。这种方法将对应点可能的范围约束到了一条直线上，而不是一个二维平面，这样大大减小了搜索空间。同时，我们也在[1]中聊过，每个对极面都是会经过基线的，每个对极线也是会经过对应的对极点的，正是因为如此，其实对于单张图的对极线组来说，如果延伸每条对极线，我们会发现每条对极线都会汇聚于对极点上，如Fig 1.2所示。也正是因为如此，我们得到的对极线们都不是平行的，如Fig 1.3所示。

![epipolarplane][epipolarplane]

<div align='center'>
    <b>
        Fig 1.1 在对极线l和l'上寻找对应点。
    </b>
</div>



![epipolar_lines_to_one][epipolar_lines_to_one]

<div align='center'>
    <b>
        Fig 1.2 对极线的延长线将会汇聚在对极点上。
    </b>
</div>



![entity][entity]

<div align='center'>
    <b>
        Fig 1.3 两张图对应的对极线对(epipolar lines pair)，我们发现这两组对极线都是不平行的。
    </b>
</div>

我们在后续的处理中，将会发现不平行的对极线会使得寻找对极点的运算变得复杂，我们期望的是， **如果每对对极线都是平行的，那将会大大减少后续算法的复杂度，特别是在求视差时，平行的对极线将会提供极大的便利。** 理想的对极线对的效果如Fig 1.4所示。我们后续就讨论如何才能从不平行的对极线转换成平行的，也就是  **图像矫正(image rectification)** 的具体操作。

![before_rectification][before_rectification]

![after_rectification][after_rectification]

<div align='center'>
    <b>
        Fig 1.4 上图表示的是未经过图像矫正的对极线配对，我们发现其是不平行的。下图是经过图像矫正的对极线配对，我们发现其平行了，这个效果正是我们所期望的。
    </b>
</div>



----

# 图像矫正

**需要明确的是，进行图像矫正之前需要知道相机的内参数和外参数[4]。**

我们首先要知道，为什么我们的对极线对会不平行呢？如图Fig 2.1所示，其不平行的原因是因为我们的成像平面没有共面(coplanar)，那么自然地，我们的直接做法就是矫正我们的成像平面，使得其共面，效果如图Fig 2.2的黄色面所示。其总体效果，如Fig 2.3所示，我们可以发现，通过这种手段，我们的对应点的搜索空间进一步缩小到了水平线上，只用一个水平线上的参数$x$表示其距离就可以描述对应点了，这样其实是大大减少了运算量的。

![non_coplanar][non_coplanar]

<div align='center'>
    <b>
        Fig 2.1 使得对极线不平行的根本原因是我们的不同摄像机的成像平面之间并不是共面的。
    </b>
</div>

![rectification_coplanar][rectification_coplanar]

<div align='center'>
    <b>
        Fig 2.2 通过矫正成像平面到共面，我们可以使得对极线对平行。
    </b>
</div>

![lines_searching][lines_searching]

<div align='center'>
    <b>
        Fig 2.3 通过矫正到(2)的情况，我们的搜索空间进一步减少了。
    </b>
</div>



我们后续具体分析这个矫正过程，我们要想怎么样才能将成像平面矫正到共面呢？我们观察图Fig 2.4，我们延伸射线$PO$和$PO^{\prime}$分别得到$P\bar{P}, P\bar{P}^{\prime}$，使得$OO^{\prime} // \bar{P}\bar{P}^{\prime}$，同时，我们确保$O, O^{\prime}$到平面$\bar{\prod}, \bar{\prod}^{\prime}$的距离相同，那么我们的平面$\bar{\prod}, \bar{\prod}^{\prime}$就是期望得到的矫正成像平面了。其实这个也不难理解，因为每个对极面都是经过基线的，而基线平行于$\bar{\prod}, \bar{\prod}^{\prime}$，那么$ep,e^{\prime}p^{\prime}$在基线上的投影必然是平行于$\bar{P}\bar{P}^{\prime}$的，同时，我们保证了两个焦点到两个平面的距离相同，确保了其是$\bar{\prod}, \bar{\prod}^{\prime}$的投影。

![parallel_epipolars][parallel_epipolars]

<div align='center'>
    <b>
        Fig 2.4 平行对极线(粉色)平行于基线（黑色OO'），而且焦点分别到其对应的成像平面的距离相同。
    </b>
</div>

我们为什么要加上这个**焦点到两个矫正后的成像平面的距离相同**这个约束呢？让我们看一个例子。

在如图Fig 2.5的系统中，我们假设两个成像平面是共面的，其焦点之间的距离$OO^{\prime} = d$，两个摄像头之间不存在有旋转关系，那么我们可以知道其相机参数为：
$$
\begin{aligned}
\mathbf{R} &= \mathbf{I} \\
\mathbf{T} &= [-d, 0, 0]^{\mathrm{T}} \\
\mathbf{E} &= [\mathbf{T}_{\times}]\mathbf{R} = 
\left[
\begin{matrix}
0 & 0 & 0 \\
0 & 0 & d \\
0 & -d & 0 
\end{matrix}
\right]
\end{aligned}
\tag{2.1}
$$
![example][example]

<div align='center'>
    <b>
        Fig 2.5 在共面的成像平面中的建模。
    </b>
</div>

从[1]中提到的对极约束的代数表达形式，我们有：
$$
\begin{aligned}
\mathbf{p}^{\prime}\mathbf{E}\mathbf{p} &= 0 \\
&\Rightarrow 
[x^{\prime}, y^{\prime}, f] 
\left[
\begin{matrix}
0 & 0 & 0 \\
0 & 0 & d \\
0 & -d & 0
\end{matrix}
\right]
\left[
\begin{matrix}
x \\
y \\
f
\end{matrix}
\right] = 0 \\

& \Rightarrow y = y^{\prime}
\end{aligned}
\tag{2.2}
$$
从中我们发现，我们必须引入这个约束去约束成像平面的位置。

![ex2][ex2]



我们知道极点$e$是本征矩阵的零空间向量，也就是说我们有$\mathbf{E}e = 0$，我们可以解出：
$$
\left[
\begin{matrix}
0 & 0 & 0 \\
0 & 0 & d \\
0 & -d & 0
\end{matrix}
\right] 
\left[
\begin{matrix}
1 \\
0 \\
0
\end{matrix}
\right] = 0
$$
我们如果从齐次坐标的角度去看待$e = \left[
\begin{matrix}
1 \\
0 \\
0
\end{matrix}
\right]$ 那么我们知道其实这个时候**极点是在无限远处了**。(关于齐次坐标系，见[2,3])

于是，其实矫正成像平面的后果就是，我们的极点被挪到了无限远处，这个和我们图Fig 1.2的推论相同。



![inf_epipolar][inf_epipolar]

<div align='center'>
    <b>
        Fig 2.6 经过矫正之后，图像的极点被移到了无限远处。
    </b>
</div>

我们到现在算是对图像矫正有了直观上的印象，接下来我们尝试用算法去描述这个过程。图像矫正算法主要分四步。

1. 用旋转矩阵$\mathbf{R}_{\mathrm{rec}}$旋转左相机，使得左成像平面的极点到无限远处。
2. 用和第一步相同的旋转矩阵旋转右相机。
3. 用外参数中的$\mathbf{R}$旋转继续旋转右相机。
4. 对坐标系调整尺度。

我们首先需要确定旋转矩阵$\mathbf{R}_{\mathrm{rec}}$。









----

# Reference

[1].  https://blog.csdn.net/LoseInVain/article/details/102665911 

[2].  Hartley R, Zisserman A. Multiple View Geometry in Computer Vision[J]. Kybernetes, 2008, 30(9/10):1865 - 1872.

[3].  https://blog.csdn.net/LoseInVain/article/details/102756630 

[4].  https://blog.csdn.net/LoseInVain/article/details/102632940 





[epipolarplane]: ./imgs/epipolarplane.jpg
[entity]: ./imgs/entity.jpg
[epipolar_lines_to_one]: ./imgs/epipolar_lines_to_one.png
[before_rectification]: ./imgs/before_rectification.jpg
[after_rectification]: ./imgs/after_rectification.jpg
[non_coplanar]: ./imgs/non_coplanar.jpg
[rectification_coplanar]: ./imgs/rectification_coplanar.jpg
[lines_searching]: ./imgs/lines_searching.jpg
[parallel_epipolars]: ./imgs/parallel_epipolars.jpg

[example]: ./imgs/example.png

[ex2]: ./imgs/ex2.jpg
[inf_epipolar]: ./imgs/inf_epipolar.jpg