<div align='center'>
    薄板样条插值(Thin Plate Spline)
</div>


<div align='right'>
    FesianXu 2020/09/08 at UESTC
</div>

# 前言

本文是笔者阅读[1]过程中，遇到了关于Thin Plate Spline[5]相关的知识，因而查找若干资料学习后得到的一些笔记，本文主要参考[2,3,4]，希望对大家有所帮助。 **如有谬误，请联系指出，转载请联系作者并注明出处**。

$\nabla$ 联系方式：

**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)

**QQ**: 973926198

github: https://github.com/FesianXu

----


# 薄板样条插值

薄板样条插值（Thin Plate Spline,TPS）是插值方法的一种，是常用的2D插值方法。假如给定两张图片中一些相互对应的控制点，如何将其中一个图片进行特定的形变，使得其控制点可以与另一张图片的控制点重合，如Fig 1.1所示。当然，提供插值的方法也特别的多，TPS是其中一种技术，其有着一个基本假设

> 如果用一个薄钢板（只是一个比喻）的形变来模拟这种2D形变，在确保所有控制点能够尽可能匹配的情况下，怎么样才能使得钢板的弯曲量最小。

几乎所有的生物有关的形变都是可以用TPS来近似，因此TPS也经常被用于脸部关键点形变等相关的应用[1]。

![tps][tps]

<div align='center'>
    <b>
        Fig 1.1 该图演示了TPS的基本任务，(a)图的p点表示的是移动之前的点，而q点表示的是移动之后的点，若干控制点产生了这种移动之后，势必整个平面发生了扭曲，其结果如(b)所示，TPS的目的就是拟合得到每个曲面上的点的变化。
    </b>
</div>



为了描述整个插值过程，按照我们刚才所说的，需要定义两个项，一个是拟合项$\mathcal{E}_{\Phi}$，测量将源点变形后距离目标点的大小；第二个是扭曲项$\mathcal{E}_{d}$，测量曲面的扭曲大小。那么有总的损失函数：
$$
\mathcal{E} = \mathcal{E}_{\Phi}+\lambda \mathcal{E}_{d}
\tag{1.1}
$$
其中的$\lambda$为权值系数，控制允许非刚体形变发生的程度，不同的$\lambda$对于整个拟合效果的影响如Fig 1.2所示。

![diff_lambda][diff_lambda]

<div align='center'>
    <b>
        Fig 1.2 不同的权重系数对于拟合效果的影响，越大的权重变形就越接近仿射变换。
    </b>
</div>

其中有：
$$
\mathcal{E}_{\Phi} = \sum_{i=1}^{N}||\Phi(p_i)-q_i||^2
\tag{1.2}
$$

$$
\mathcal{E}_d = \int \int_{\mathbb{R}^2} \bigg( \bigg( \dfrac{\partial^2\Phi}{\partial \mathrm{x}^2} \bigg)^2
+ 2
\bigg( \dfrac{\partial^2\Phi}{\partial \mathrm{x} \partial \mathrm{y}} \bigg)^2 +
\bigg( \dfrac{\partial^2\Phi}{\partial \mathrm{y}^2} \bigg)^2
\bigg)^2 \mathrm{dx}\mathrm{dy}
\tag{1.3}
$$

其中的$N$为控制点的数量，式子(1.2)很容易理解，是源目标经过形变函数$\Phi$之后和目标之间的距离；而式子(1.3)是曲面扭曲的能量函数，由文献[6]中给出，最小化式子(1.1)的结果，可以推导出形变函数的唯一闭式解结果为：
$$
\Phi(p) = \mathbf{M} \cdot p + m_0+\sum_{i=1}^{N} \omega_i U(||p-p_i||) 
\tag{1.4}
$$
其中$p$为曲面上的任意一个点，有$p = (x,y)^{\mathrm{T}}$，$p_i$是对应域的控制点，而$\mathbf{M} = (m_1,m_2)$，而这里的$U(\cdot)$为径向基函数，表示某个曲面上的点的变形会受到所有控制点变形的影响（当然，不同控制点的影响程度不一样），有
$$
U(x) = r^2\log{r}
\tag{1.5}
$$
而$\omega_i$表示对不同径向基的加权。如Fig 1.3所示，如果我们假设每个控制点都对应一个高度，也就是$(x_i,y_i)\rightarrow v_i$，也就是说控制点是三维空间坐标系中的自变量，而其高度是因变量，那么我们可以再继续分析式子(1.4)中的第一项和第二项。

我们发现第一项其实是尝试用一个平面$y = \mathbf{M} \cdot p+m_0$去拟合所有的目标控制点，当然这个拟合肯定不够好，因此用第二项尝试在该平面的基础上去弯曲（当然是尽可能小的弯曲），从而达到更好的拟合效果，如Fig 1.3所示。此时有未知参数$\mathbf{M} \in \mathbb{R}^2, m_0 \in \mathbb{R}$，和$\omega_i, i \in [1,N]$，因此一共有$1+2+N$个参数，其中$D = 2$是维度，$N$是控制点数目。

![thinplates][thinplates]

<div align='center'>
    <b>
        Fig 1.3 最小程度地扭曲平面，使得该曲面可以符合所有的控制点，而扭曲程度最小。
    </b>
</div>

我们为了求解形式一般化，用以下矩阵代表之前谈到的数值，有：
$$
\mathbf{P} = 
\left[
\begin{matrix}
1 & x_1 & y_1 \\
1 & x_2 & y_2 \\
\vdots & \vdots & \vdots \\
1 & x_n & y_n
\end{matrix}
\right]
\tag{1.6}
$$
其中每一行代表一个控制点坐标，该矩阵称之为控制点矩阵。
$$
\mathbf{Y} = 
\left[
\begin{matrix}
v_1 \\
v_2 \\
\vdots \\
v_n \\
0 \\
0 \\
0 \\
\end{matrix}
\right]
\tag{1.7}
$$
该矩阵称之为高度矩阵，后面三个0是为了形式统一填充的。

$$
\mathbf{K} = 
\left[
\begin{matrix}
U(r_{11}) & U(r_{12}) & \cdots \\
U(r_{21}) & U(r_{22}) & \cdots \\
\cdots & \cdots & U(r_{NN}) 
\end{matrix}
\right]
\tag{1.8}
$$
其中$r_{ij} = ||p_{i}-p_{j}||$表示两个控制点之间的距离。令矩阵$\mathbf{L}$为：
$$
\mathbf{L} = 
\left[
\begin{matrix}
\mathbf{K} & \mathbf{P} \\
\mathbf{P}^{\mathrm{T}} & \mathbf{0}
\end{matrix}
\right] \in \mathbb{R}^{(N+3) \times (N+3)}
\tag{1.9}
$$
那么由式子(1.4)和$\Phi(p_i)=v_i$，有：
$$
\mathbf{Y} = \mathbf{L} (\Omega|m_0,m_1,m_2)^{\mathrm{T}}
\tag{1.10}
$$
其中$\Omega = (\omega_1,\cdots,\omega_N)$。其中的后三行引入了一组对参数的约束（虽然我并不知道这组约束的含义，有了解的朋友请在评论区赐教，谢谢）：
$$
\begin{aligned}
\sum_{i=1}^N \omega_i &= 0 \\
\sum_{i=1}^N x_i\omega_i &= 0 \\
\sum_{i=1}^N y_i\omega_i &= 0
\end{aligned}
\tag{1.11}
$$
那么从式子(1.10)我们有：
$$
(\Omega|m_0,m_1,m_2)^{\mathrm{T}} = \mathbf{L}^{-1}\mathbf{Y} 
\tag{1.12}
$$
当然也可以通过解线性方程组(1.10)得到参数组$(\Omega|m_0,m_1,m_2)^{\mathrm{T}}$，一旦这个参数组计算得到，那么我们的插值函数$\Phi(p)$也就已知了，只要给定平面上任意一个点，就能通过插值函数将其插值到目标平面上。

# 变形(deformation)

这里介绍TPS的一个主要应用，对图片的控制点进行偏移，以达到通过控制点对图像进行特定形变的目的。如Fig 2.1所示，通过拉拽嘴角的控制点（即是蓝色点），使得周围的像素，比如$A$点移动到了$A^{\prime}$点，此时存在位移$(\Delta x, \Delta y)$，此时我们需要插值这个位移。 当然，对应控制点之间的移动偏移是可以知道的，记为$\Delta \mathbf{S} = \{(\Delta x_1, \Delta y_1),\cdots,(\Delta x_N, \Delta y_N)\}$，我们要根据已知的控制点偏移去插值图片上其他任意像素点的偏移。

不妨我们把这两个位移的分量隔离开来，不考虑两个维度之间的相关性，那么可以将第一章提到的“高度”$v_i$在这里理解成每一个位移的分量，那么我们有两个插值函数需要预测，即是：
$$
\begin{aligned}
\Delta x(p) &= \Phi(p)_{\Delta x} \\
\Delta y(p) &= \Phi(p)_{\Delta y}
\end{aligned}
\tag{2.1}
$$
![deformation_face][deformation_face]

<div align='center'>
    <b>
        Fig 2.1 通过拉拽嘴角和眼角的控制点，可以实现图像的内容形变。
    </b>
</div>



假如只是选定6个控制点，分别是图片的四个角落，右眼和右侧嘴角，如Fig 2.2所示。



![herve-smile-points][herve-smile-points]

<div align='center'>
    <b>
        Fig 2.2 红色点表示移动之前的控制点，绿色点表示移动后的控制点，我们发现只是移动了右边眼睛和右边嘴角。
    </b>
</div>





![dxdy][dxdy]







# Reference

[1]. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., & Sebe, N. (2019). First order motion model for image animation. In *Advances in Neural Information Processing Systems* (pp. 7137-7147).

[2]. http://profs.etsmtl.ca/hlombaert/thinplates/

[3]. https://www.jianshu.com/p/2cc189dfbcc5#fn3

[4]. https://www.cse.wustl.edu/~taoju/cse554/lectures/lect07_Deformation2.pdf

[5]. Bookstein, F. L. (1989). Principal warps: Thin-plate splines and the decomposition of deformations. *IEEE Transactions on pattern analysis and machine intelligence*, *11*(6), 567-585.

[6]. Kent, J. T. and Mardia, K. V. (1994a). The link between kriging and thin-plate splines. In: Probability, Statistics and Optimization: a Tribute to Peter Whittle (ed. F. P. Kelly), pp 325–339. John Wiley & Sons, Ltd, Chichester. page 282, 287, 311



[tps]: ./imgs/tps.png
[thinplates]: ./imgs/thinplates.png
[diff_lambda]: ./imgs/diff_lambda.png

[deformation_face]: ./imgs/deformation_face.png
[dxdy]: ./imgs/dxdy.png
[herve-smile-points]: ./imgs/herve-smile-points.png





