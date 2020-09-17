<div align='center'>
    [GAMES101学习笔记] 角度与立体角
</div>
<div align='right'>
FesianXu 2020/09/16 at UESTC
</div>

# 前言

本系列文章是笔者学习GAMES101 [1]过程中的学习笔记，如有谬误请联系指出，转载请注明出处，谢谢。

$\nabla$ 联系方式：

**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)

**QQ**: 973926198

github: https://github.com/FesianXu

----



# 立体角

计算机图形学中的光线传播建模是一个非常重要的课题，我们考虑到光线在实际物理空间上的传播是一个空间辐射的过程，因此需要定义出三维空间中的“角度”的概念，当然这里的角度不同于二维情况下的角度容易定义。首先我们先回顾二维平面上角度的定义，给定一个圆形，如Fig 1 (a)所示，我们定义周长与对应半径的比例为角度（弧度制），即是：
$$
\theta = \dfrac{l}{r}
\tag{1}
$$

仿照二维空间中角度的定义，我们定义三维空间中的立体角（Solid angle），如式子(2)所示。
$$
\Omega = \frac{A}{r^2}
\tag{2}
$$
其中的$A$是锥体在球面上围成的面积，如Fig 1 (b)所示。

![solid_angle_and_angle][solid_angle_and_angle]

<div align='center'>
    <b>
        Fig 1. 二维平面的角度定义示意图如(a)所示；三维空间的立体角的定义如（b）所示。
    </b>
</div>
式子(2)中的曲面面积并没有周长那么容易计算，我们需要使用微积分思考如何计算。如Fig 2所示，我们用球面坐标$(\theta,\phi,r)$来表示球面上的任意一点，那么，我们考虑$(\mathrm{d}\theta, \mathrm{d}\phi)$的变化量所围成的曲面的面积大小，因为这个变化量很小，我们可以将曲面视为是一个边长为$H \times W$的矩形。如Fig 3所示，我们可以认为其$H$是一个等腰三角形的底。

![solid_angle_differential][solid_angle_differential]

<div align='center'>
    <b>
        Fig 2. 立体角所围成的曲面的微元。
    </b>
</div>

那么通过简单的几何关系，我们有：
$$
H = 2r\sin(\dfrac{\mathrm{d}\theta}{2})
\tag{3}
$$
因为有等价无穷小关系：
$$
\sin(\dfrac{\mathrm{d}\theta}{2}) \sim \dfrac{\mathrm{d}\theta}{2}
\tag{4}
$$
因此式子(3)(4)联立有：
$$
H = r \mathrm{d}\theta
\tag{5}
$$
![H][H]

<div align='center'>
    <b>
        Fig 3. 矩形的H可以视为是等腰三角形的底。
    </b>
</div>



同理我们可以求出矩形的$W$为：
$$
W = r\sin(\theta) \mathrm{d}\phi
\tag{6}
$$
那么有：
$$
\mathrm{d}A = H \times W = (r\mathrm{d}\theta)(r\sin(\theta)\mathrm{d}\phi) = r^2\sin(\theta)\mathrm{d}\phi\mathrm{d}\theta
\tag{7}
$$
那么立体角的微元为：
$$
\mathrm{d}\Omega = \dfrac{\mathrm{d}A}{r^2} = \sin(\theta)\mathrm{d}\theta\mathrm{d}\phi
\tag{8}
$$
那么，此时对立体角微元进行全积分，我们可以得到立体角的范围最大为：
$$
\Omega = \int_{S^2} \mathrm{d}\Omega = \int_{0}^{2\pi} \int_{0}^{\pi} \sin(\theta)\mathrm{d}\theta\mathrm{d}\phi = 4\pi
\tag{9}
$$



而二维平面的角度范围是最大到$2 \pi$。

# Reference

[1]. https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html





[solid_angle_and_angle]: ./imgs/solid_angle_and_angle.png

[solid_angle_differential]: ./imgs/solid_angle_differential.png
[H]: ./imgs/H.png

