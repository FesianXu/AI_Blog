<div align='center'>
    在pytorch中的双线性采样（Bilinear Sample）
</div>


<div align='right'>
    FesianXu 2020/09/16 at UESTC
</div>
# 前言

双线性插值与双线性采样是在图像插值和采样过程中常用的操作，在`pytorch`中对应的函数是`torch.nn.functional.grid_sample`，本文对该操作的原理和代码例程进行笔记。**如有谬误，请联系指正，转载请联系作者并注明出处，谢谢。**

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**QQ**: 973926198

github: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

----



# 双线性插值原理

插值（interpolation）在数学上指的是 **一种估计方法，其根据已知的离散数据点去构造新的数据点**。以曲线插值为例子，如Fig 1.1所示的曲线线性插值为例，其中红色数据点是已知的数据点，而蓝色线是根据相邻的两个红色数据点进行线性插值估计出来的。

![interpolation_curve][interpolation_curve]

<div align='center'>
    <b>
        Fig 1.1 一个简单的曲线线性插值的例子。
    </b>
</div>

一维的曲线插值的原理可以推广到任意维度的数据形式上，比如我们常见的图像是一种二维数据，就可以进行二维插值，常见的插值方法如Fig 1.2所示。

![interpolation_methods][interpolation_methods]

<div align='center'>
    <b>
        Fig 1.2 常见的1D和2D数据插值方法。
    </b>
</div>

在本文中，我们主要讨论的是双线性采样，而双线性采样和双线性插值紧密相关，因此本章节主要介绍双线性插值。还是以2D图像插值为例子，如Fig 1.3所示，假设图片上给定了红色数据点的像素值，假设待求的绿色点$P=(x,y)$，其中已知每个顶点像素坐标为：
$$
\begin{aligned}
Q_{12} &= (x_{1}, y_{2})^{\mathrm{T}} \\
Q_{22} &= (x_{2}, y_{2})^{\mathrm{T}} \\
Q_{11} &= (x_{1}, y_{1})^{\mathrm{T}} \\
Q_{21} &= (x_{2}, y_{1})^{\mathrm{T}} \\
\end{aligned}
\tag{1.1}
$$
而每个顶点的像素值表示为$f(Q_{ij}), i =1,2, j=1,2$。通过简单的线性插值（按比例划分），我们可以求出蓝色数据点的估计值：
$$
\begin{aligned}
R_2 &= f(x,y_2) = \dfrac{x_2-x}{x_2-x_1}f(Q_{12})+\dfrac{x-x_1}{x_2-x_1}f(Q_{22}) \\
R_1 &= f(x,y_1) = \dfrac{x_2-x}{x_2-x_1}f(Q_{11})+\dfrac{x-x_1}{x_2-x_1}f(Q_{21})
\end{aligned}
\tag{1.2}
$$
然后通过蓝色点，再一次进行线性插值，可以估计出绿色点的值：
$$
\begin{aligned}
f(x,y) &= \dfrac{y_2-y}{y_2-y_1}f(x,y_1)+\dfrac{y-y_1}{y_2-y_1}f(x,y_2) \\
&= \dfrac{1}{(x_2-x_1)(y_2-y_1)}[x_2-x, x-x_1]
\left[
\begin{matrix}
f(Q_{11}) & f(Q_{12}) \\
f(Q_{21}) & f(Q_{22})
\end{matrix}
\right]
\left[
\begin{matrix}
y_2-y \\
y-y_1
\end{matrix}
\right]
\end{aligned}
\tag{1.3}
$$

因为该方法涉及到了两轮（注意不是两次，而是三次）的线性插值，因此称之为双线性插值（Bilinear Interpolation）。

![bilinear_interpolation][bilinear_interpolation]

<div align='center'>
    <b>
        Fig 1.3 给定了四个红色数据点（像素点）的值，通过双线性插值求中间的绿色数据点的值。
    </b>
</div>

# 双线性采样以及grid_sample

在深度学习框架pytorch中提供了一种称之为双线性采样（Bilinear Sample）的函数`torch.nn.functional.grid_sample` [1]，该函数主要输入一个形状为$(N,C,H_{in},W_{in})$的`input`张量，输入一个形状为$(N,H_{out},W_{out},2)$的`grid`张量，输出一个形状为$(N,C,H_{out},W_{out})$的`output`张量。

其中$N$为`batch`批次，我们主要关注后面的维度的代表意义。输入的`grid`是一个$H_{out} \times W_{out}$大小的空间位置矩阵，其中每个元素都代表着一个二维空间坐标$(x,y)$，该坐标指明了在`input`上采样的坐标，而输出张量的每个位置`output[n,:,h,w]`的值，取决于这个输入`input`和采样坐标的值（通过双线性插值形成）。通过这个函数，可以通过指定原图的不同坐标位置，实现图片的变形（deformation）等，在很多研究中有着广泛地应用[2]。

注意到这里的输出张量尺寸和输入张量尺寸是不一定一致的，因此涉及到了插值过程，而且输入的`grid`的每一个坐标都是归一化到了$[-1,1]$之间的，我们举一个简单的代码例子，明晰下细节。

```python
import torch.nn.functional as F
import torch
inputv = torch.arange(4*4).view(1, 1, 4, 4).float()
print(inputv)
'''
输出尺寸为(1,1,4,4)
输出为：tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
'''
# 生成grid，这个grid大小为(1,8,8,2)，空间尺寸而言是原输入图片的两倍。
d = torch.linspace(-1,1, 8)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0) # add batch dim

# 进行双线性采样，其中指定align_corners=True保证了输出的整个图片的角边像素与原输入的一致性。
output = F.grid_sample(inputv, grid,align_corners=True)
print(output)
'''
tensor([[[[ 0.0000,  0.4286,  0.8571,  1.2857,  1.7143,  2.1429,  2.5714,
            3.0000],
          [ 1.7143,  2.1429,  2.5714,  3.0000,  3.4286,  3.8571,  4.2857,
            4.7143],
          [ 3.4286,  3.8571,  4.2857,  4.7143,  5.1429,  5.5714,  6.0000,
            6.4286],
          [ 5.1429,  5.5714,  6.0000,  6.4286,  6.8571,  7.2857,  7.7143,
            8.1429],
          [ 6.8571,  7.2857,  7.7143,  8.1429,  8.5714,  9.0000,  9.4286,
            9.8571],
          [ 8.5714,  9.0000,  9.4286,  9.8571, 10.2857, 10.7143, 11.1429,
           11.5714],
          [10.2857, 10.7143, 11.1429, 11.5714, 12.0000, 12.4286, 12.8571,
           13.2857],
          [12.0000, 12.4286, 12.8571, 13.2857, 13.7143, 14.1429, 14.5714,
           15.0000]]]])
'''

```

在这个过程中，我们生成的采样坐标网格`grid`很简单，单纯只是在x,y两个维度，都把$[-1,1]$均分为了8份。

我们分析下双线性采样后的每个像素的大小计算过程。因为每个输入坐标都是$[-1,1]$，而实际原输入的矩阵大小为$[0,3]$，而且刚好是一个方阵，因此可以计算出从`grid`到实际坐标的映射为:
$$
f_{x} = f_{y} = \dfrac{3}{2}x_{norm}+\dfrac{3}{2}
\tag{1}
$$
这个映射将归一化坐标映射到了实际的原图坐标，如果不是方阵，那么就必须对$x,y$每个维度都计算一个映射方程。

我们暂时只考虑怎么计算其中某一个像素的值，暂时我们考虑`grid`坐标为$[1,1]$的值。我们打印出`grid[0,1,1,:]`，发现这个归一化坐标值为`tensor([[-0.7143, -0.7143]])`，那么通过反归一化映射，也就是式子(1)后，有实际图片坐标为$(0.4285, 0.4285)$，这个时候我们发现这个坐标不是整数，因此为了求出这个坐标的像素值，我们要通过之前谈到的双线性插值去估计。

首先求出每一行的插值结果，有$f(x,y_1) = 0.4285$，这个是在$[0,1]$中插值的结果；有$f(x,y_2) = 4.4285$这个是在$[4,5]$范围内插值的结果，然后再在$[0.4285,4.4285]$中进行插值，有$f(x,y) = (4.4285-0.4285) \times 0.4285+0.4285=2.1428$。这就是整个双线性采样的计算过程。

注意：这个输入`input`也可以是$(N,C,D,H_{in},W_{in})$的5D输入，该输入考虑的是对视频进行处理。本文中只考虑了图片数据，不过原理是类似的，不再赘述。



# Reference

[1]. https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample

[2]. https://blog.csdn.net/LoseInVain/article/details/108710063









[interpolation_curve]: ./imgs/interpolation_curve.png
[interpolation_methods]: ./imgs/interpolation_methods.png
[bilinear_interpolation]: ./imgs/bilinear_interpolation.png

