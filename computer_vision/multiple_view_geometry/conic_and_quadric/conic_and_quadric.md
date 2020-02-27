<div align="center"> 
    【多视角立体视觉系列】 conic圆锥线和quadric二次曲锥面的定义和应用
</div>


<div align="right">
    20200226 FesianXu
</div>

# 前言

之前我们讨论过一些几何元素，比如点线面等，在本文中，我们将谈到称之为圆锥线和二次曲锥面的几何元素，这种类型的曲线对于讨论计算机视觉中投影是非常有效的，同时也是定义不同几何变换——投影变换，仿射变换，欧几里德变换等区别的要点之一，需要我们很好地掌握。**如有谬误，请联系指出，转载请注明出处。**

$\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu



-------



# 圆锥线

## 定义

我们暂且不管计算机视觉和摄像机成像的这方面的应用背景，在数学的角度上先描述下圆锥线。**圆锥线(conic)**首先是一种在二维平面上的二维点的轨迹，在欧几里德几何中，圆锥线主要分为三种：双曲线(hyperbola)，椭圆线(ellipse)，抛物线(parabola)。名字很熟悉，在高中大家都应该或多或少学过这些几何，而这些不同的曲线之所以都被称之为圆锥线的原因是，它们都可以看成是不同方向的平面切割圆锥体形成的相交平面的边缘的轨迹，如Fig 1.1所示。我们将会发现，这些不同的圆锥线在投影变换(projective transformation)下都是等价的，这个也就是我们指的“不同的角度下观察”的意思，投影变换可以描述这个过程。

![conic][conic]

<div align='center'>
    <b>
        Fig 1.1 圆锥线，用不同方向的平面切割圆锥体，就形成了不同的圆锥曲线，我们会发现，这里所谓的不同方向，其实指的是不同的方向去观察该圆锥体，这点和我们用相机在不同方向成像是异曲同工的，因此在立体视觉中会引入这个概念。
    </b>
</div>

那么，作为解析几何的角度，我们尝试用代数的形式描述这类二次曲线，我们有：
$$
ax^2+bxy+cy^2+dx+cy+f=0
\tag{1.1}
$$
其中曲线轨迹上的每个点为$p=(x,y)^{\mathrm{T}}$，如果用齐次化坐标的形式表达它[1]，令$p = (x_1,x_2,x_3)$即是$x \rightarrow x_1/x_3, y \rightarrow x_2/x_3$，那么我们把(1.1)转成(1.2)，有：
$$
ax_1^2+bx_1x_2+cx_2^2+dx_1x_3+ex_2x_3+fx_3^2=0
\tag{1.2}
$$
用矩阵形式表达就是：
$$
\begin{aligned}
\mathbf{x}^{\mathrm{T}} & \mathbf{C}\mathbf{x} = 0 \\
\mathbf{C} &= 
\left[
\begin{matrix}
a & b/2 & d/2 \\
b/2 & c & e/2 \\
d/2 & e/2 & f
\end{matrix}
\right] \\
\mathbf{x} &= (x_1,x_2,x_3)^{\mathrm{T}}
\end{aligned}
\tag{1.3}
$$
我们注意到描述一个圆锥线，在代数角度下用一个对称矩阵$\mathbf{C} \in \mathbb{R}^{3\times3}$就够了，其矩阵的自由度为5，为什么是5而不是6呢？这里不是有a,b,c,d,e,f六个未知参数吗？那是因为对于一个圆锥线来说，其尺度因子是不重要的，毕竟我们的等式右边是个0，可以除去除了0之外的任意数，因此一般可以对矩阵做归一化处理，如：
$$
\begin{aligned}
\mathbf{C} = \dfrac{1}{f} \mathbf{C} = 
\left[
\begin{matrix}
a/f & b/2f & d/2f \\
b/2f & c/f & e/2f \\
d/2f & e/2f & 1
\end{matrix}
\right]
\end{aligned}
\tag{1.4}
$$
因此自由度就只有5了。因此我们发现，只需要用五个点，就可以确定一个圆锥线，联立方程：
$$
\left[
\begin{matrix}
x_1^2 & x_1y_1 & y_1^2 & x_1 & y_1 & 1 \\
x_2^2 & x_2y_2 & y_2^2 & x_2 & y_2 & 1 \\
x_3^2 & x_3y_3 & y_3^2 & x_3 & y_3 & 1 \\
x_4^2 & x_4y_4 & y_4^2 & x_4 & y_4 & 1 \\
x_5^2 & x_5y_5 & y_5^2 & x_5 & y_5 & 1 
\end{matrix}
\right]
\left[
\begin{matrix}
a \\
b \\
c \\
d \\
e \\
1
\end{matrix}
\right] = 0
\tag{1.5}
$$
可以发现，圆锥曲线的解是这个$5 \times 6$矩阵的零向量[2]。



## 圆锥线的切线

在几何中，我们经常需要使用圆锥线的切线方程，在齐次坐标系的表示下，这种表示特别简单，$l = \mathbf{C}\mathbf{x}$，其中$\mathbf{C} \in \mathbb{R}^{3 \times 3}$为圆锥线的对称矩阵，$\mathbf{x} \in \mathbb{R}^3$是齐次坐标下的点坐标，并且该点在圆锥线之上。这点其实很容易证明：$l = \mathbf{C}\mathbf{x}$首先是经过点$\mathbf{x}$的，有$l^{\mathrm{T}}\mathbf{x} = \mathbf{x}^{\mathrm{T}}\mathbf{C}\mathbf{x} = 0$，所以这个点同时在直线和圆锥线上，接下来我们证明只有一个交点即可。假设还有另外一个交点$\mathbf{y}$，那么我们有$\mathbf{y}^{\mathrm{T}}\mathbf{C}\mathbf{y} = 0$并且因为直线也经过交点$\mathbf{y}$，有$\mathbf{x}^{\mathrm{T}}\mathbf{C}\mathbf{y} = l^{\mathrm{T}} \mathbf{y} = 0$。因此不难得到(1.6)也成立。
$$
(\mathbf{x}+\alpha\mathbf{y})^{\mathrm{T}}\mathbf{C}(\mathbf{x}+\alpha\mathbf{y}) = 0, \alpha \in \mathbb{R}
\tag{1.6}
$$
那么也就是说，整个直线两点$\mathbf{x}, \mathbf{y}$之间的连线的任意点都在圆锥线上，因此这两个点只能是同一个点，即是交点。

![htangent_const][tangent_const]

## 对偶圆锥线

我们之前定义的圆锥线是以如何构成该圆锥线的点的轨迹来定义的，因此形式$\mathbf{x}^{\mathrm{T}} \mathbf{C}\mathbf{x} = 0$中的$\mathbf{x}$是点。然而，我们知道用向量的形式既可以表示点，也能表示直线，如果这里的不是点，而是直线，那么方程形式就变成了$\mathbf{l}^{\mathrm{T}}\mathbf{C}\mathbf{l} = 0$，而数值上来看是和点形式一样的，但是几何含义却完全不同了。在这种情况，如Fig 1.2(b)所示，是无数直线的不断运动的轨迹的切线交点构成了整个圆锥线的轨迹。这个称之为**对偶圆锥线(dual conics)**。同样的，对偶圆锥线也是用对称的$3 \times 3$矩阵来表示，表示为$\mathbf{C}^{*}$。利用共轭矩阵的性质，我们可以求出以点形式的圆锥线$\mathbf{C}$对应的对偶圆锥线的矩阵为$\mathbf{C}^* = \mathbf{C}^{-1}$。这个其实很容易推导，我们知道经过圆锥线的$\mathbf{x}$的切线为$\mathbf{l} = \mathbf{C}\mathbf{x}$，反过来，我们有$\mathbf{x} = \mathbf{C}^{-1}\mathbf{l}$，那么我们有：
$$
\mathbf{x}^{\mathrm{T}}\mathbf{C}\mathbf{x} = (\mathbf{C}^{-1}\mathbf{l})^{\mathrm{T}} \mathbf{C} (\mathbf{C}^{-1}\mathbf{l}) = \mathbf{l}^{\mathrm{T}}	\mathbf{C}^{\mathrm{-1}} \mathbf{l} = 0
\tag{1.7}
$$
![dual_conics][dual_conics]

<div align='center'>
    <b>
        Fig 1.2 圆锥线和对偶圆锥线，(a)圆锥线是以点的轨迹来定义的, (b)对偶圆锥线是以直线的形式来定义的，无数直线的运动构成了整个圆锥线的轨迹。
    </b>
</div>



## 点变换后的圆锥线

![point_transform][point_transform]

假设新的点$\mathbf{x}^{\prime} = \mathbf{H}\mathbf{x}, \mathbf{H} \in \mathbb{R}^{3 \times 3}$，那么新点的圆锥线可以表示为：
$$
(\mathbf{x}^{\prime})^{\mathrm{T}} \mathbf{C}^{\prime} \mathbf{x}^{\prime} = (\mathbf{H}\mathbf{x})^{\mathrm{T}}\mathbf{C}^{\prime}\mathbf{H}\mathbf{x} = \mathbf{x}^{\mathrm{T}}\mathbf{H}^{\mathrm{T}}\mathbf{C}^{\prime}\mathbf{H}\mathbf{x} = \mathbf{x}^{\mathrm{T}} \mathbf{C} \mathbf{x}
$$
因此有：
$$
\mathbf{C} = \mathbf{H}^{\mathrm{T}}\mathbf{C}^{\prime}\mathbf{H}
$$

----



# 二次曲锥面

我们上一章谈到的圆锥线是在二维平面上定义出来的，那么在三维空间中，二维的圆锥线就扩展成为了**二次曲锥面(quadric)**。其数学形式类似(1.3)，可以表达成:
$$
\mathbf{X}^{\mathrm{T}} \mathbf{Q} \mathbf{X} = 0, \mathbf{Q} \in \mathbb{R}^{4 \times 4}
\tag{2.1}
$$
类似于圆锥线，二次曲锥面的自由度为9，去掉了尺度因子。同样，如果$\mathbf{X}$在二次曲锥面上，其切面可以表示为$\pi = \mathbf{Q}\mathbf{X}$。与圆锥线不同的是，平面与二次曲锥面的交叠轨迹不是点，而是一个圆锥线$\mathbf{C}$。当然，二次曲锥面也有其对偶形式，就如同对偶圆锥线一样，表示为$\pi^{\mathrm{T}} \mathbf{Q}^{*} \pi = 0$，其中的$\mathbf{Q}^*$是$\mathbf{Q}$的共轭矩阵，一般是$\mathbf{Q}^* = \mathbf{Q}^{-1}$。 如图Fig2.1所示，我这里贴了几张不同类型的二次曲锥面，我们可以发现，曲锥面的形式要比圆锥线复杂很多。

![non_ruled_quadric][non_ruled_quadric]

![ruled_quadric][ruled_quadric]

## 点变换后的二次曲锥面

同圆锥线，假设有点变换$\mathbf{X}^{\prime} = \mathbf{H} \mathbf{X}, \mathbf{H} \in \mathbb{R}^{4 \times 4}$，那么有$\mathbf{Q} = \mathbf{H}^{\mathrm{T}}\mathbf{Q}^{\prime}\mathbf{H}$。

# 应用

我们在计算机视觉特别是成像中，我们会发现对物体的一些变换，可以体现到对圆锥线或者二次曲锥面的变换上，方便我们分析问题，本文作为预备知识，仅仅介绍了圆锥线和二次曲锥面，先不考虑其他内容了。我们后面的章节再见。



# Reference

[1]. https://blog.csdn.net/LoseInVain/article/details/102756630

[2]. https://en.wikipedia.org/wiki/Null_vector

[3]. Hartley R, Zisserman A. Multiple View Geometry in Computer Vision[J]. Kybernetes, 2008





[conic]: ./imgs/conic.png

[tangent_const]: ./imgs/tangent_const.gif
[dual_conics]: ./imgs/dual_conics.jpg
[ruled_quadric]: ./imgs/ruled_quadric.jpg
[non_ruled_quadric]: ./imgs/non_ruled_quadric.jpg
[point_transform]: ./imgs/point_transform.jpg

