<div align='center'>
   	投影相机，透视相机，弱透视相机和仿射相机的区别和联系
</div>

<div align='right'>
    2019.11.03 FesianXu
</div>

# 前言

相机一般来说是一种从3D到2D的一种投影工具，其按照数学模型可以分为投影相机，透视相机，弱透视相机和仿射相机等，笔者在本文中尝试对其进行区分和联系。**如有谬误，请联系指正。转载请注明出处。**

 $\nabla$  联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu 



----



# 投影相机（projective camera）

相机说到底是一种从3D的现实世界中的点投影到2D平面上点的工具，那么作为这个投影$\mathcal{P}^{3} \rightarrow \mathcal{P}^2$，对其描述最为通用的莫过于在齐次坐标系下，用：
$$
\left[
\begin{matrix}
x_1 \\
x_2 \\
x_3 
\end{matrix}
\right] = 
\mathcal{M} \mathbf{X}
=
\left[
\begin{matrix}
T_{11} & T_{12} & T_{13} & T_{14} \\
T_{21} & T_{22} & T_{23} & T_{24} \\
T_{31} & T_{32} & T_{33} & T_{34} 
\end{matrix}
\right] 
\left[
\begin{matrix}
X_1 \\
X_2 \\
X_3 \\
X_4
\end{matrix}
\right]
\tag{1}
$$
其中3D点$\mathbf{X} = (X_1, X_2, X_3, X_4)^{\mathrm{T}}$和2D点$\mathbf{x} = (x_1, x_2, x_3)^{\mathrm{T}}$采用齐次坐标系描述[6]。不难知道，其非齐次坐标系表达为：
$$
\begin{aligned}
(x, y) &= (\dfrac{x_1}{x_3}, \dfrac{x_2}{x_3}) \\
(X,Y,Z) &= (\dfrac{X_1}{X_4}, \dfrac{X_2}{X_4}, \dfrac{X_3}{X_4})
\end{aligned}
\tag{2}
$$
虽然变换矩阵$\mathcal{M} \in \mathbb{R}^{4 \times 4}$是一个有着12个元素的矩阵，但是因为对于坐标而言，最重要的是比例关系，因此我们可以对$\mathcal{M}$进行尺度缩放，不妨将其全部除以$T_{34}$，得到$\dfrac{1}{T_{34}} \mathcal{M}$ 在这个新的变换矩阵中，原先$T_{34}$的位置变成了1，因此其实其自由度只有11，而不是12.（对于理解这里的尺度变化，我们可以这样认为，我们新投影出来的2D图像的尺度大小是可以变化的，比如原先是$800 \times 600$，尺度变换后可能就变成了$400 \times 300$，其比例还是一样的，因此图中点的坐标其实比例也是不变的）

我们把有这种关系的相机称之为**投影相机(projective camera)**，显然，投影相机是非线性的[7]，这里指的非线性是图像显示的尺寸大小和真实尺寸大小不成线性比例，具体见[7]讨论。



# 透视相机（perspective camera）

投影相机的一种特殊例子也是更为常见的例子是所谓的**透视相机(perspective camera)**，这种相机的投影方式称之为**透视投影（perspective projection）**或者是**中心投影(central projection)**。跟一般的，当其变换矩阵$\mathcal{M}$的最左边$3 \times 3$矩阵是一个旋转矩阵[8]，并且这个旋转矩阵的尺度放缩因子是$1/f$时，透视相机模型成为我们熟悉的针孔模型的表达[9]，如：
$$
\mathcal{M} = 
\left[
\begin{matrix}
T_{11} & T_{12} & 0 & T_{14} \\
T_{21} & T_{22} & 0 & T_{24} \\
0 & 0 & 1/f & T_{34} 
\end{matrix}
\right]
\tag{3}
$$
其中最简单的形式莫过于是
$$
\mathcal{M} = 
\left[
\begin{matrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1/f & 0 
\end{matrix}
\right]
\tag{4}
$$
于是有了熟悉的针孔模型的表达：
$$
\left[
\begin{matrix}
x \\
y
\end{matrix}
\right] = 
\dfrac{f}{Z}
\left[
\begin{matrix}
X \\
Y
\end{matrix}
\right]
\tag{5}
$$
注意到我们是**对每个点的深度Z进行反比放缩的**，这一点很重要，这个导致了透视相机的成像的非线性性，见[7]中的讨论。

![image_pin][image_pin]



# 仿射相机（affine camera）

仿射相机也是投影相机的一种特殊情况，其变换矩阵为：
$$
\left[
\begin{matrix}
x_1 \\
x_2 \\
x_3 
\end{matrix}
\right] = 
\mathcal{M} \mathbf{X}
=
\left[
\begin{matrix}
T_{11} & T_{12} & T_{13} & T_{14} \\
T_{21} & T_{22} & T_{23} & T_{24} \\
0 & 0 & 0 & T_{34} 
\end{matrix}
\right] 
\left[
\begin{matrix}
X_1 \\
X_2 \\
X_3 \\
X_4
\end{matrix}
\right]
\tag{6}
$$
考虑到通常我们也会把$T_{34}$设置为一个常数，比如1，因此现在的自由度变成了8。

于是，投影后的坐标为：
$$
\begin{aligned}
x &= \dfrac{x_1}{x_3} = \dfrac{T_{11}X_1 + T_{12}X_2 + T_{13}X_3 + T_{14}X_4}{T_{34}X_4}  \\
y &= \dfrac{x_2}{x_3} = \dfrac{T_{21}X_1 + T_{22}X_2 + T_{23}X_3 + T_{24}X_4}{T_{34}X_4}  
\end{aligned}
\tag{7}
$$
注意到其分母是$T_{34}X_4$是一个常数，因此其投影后的坐标$(x,y)$是一个只由$(X_1, X_2, X_3, X_4)$决定了的线性关系。这个很特殊，因为在这种情况下，通常透视关系之下不再保存的平行关系，也会在仿射相机中保留下来。一般性的透视关系中不保留平行关系的例子见[8]。因为其线性性，仿射相机可以用更简单的，非齐次坐标系的表达方式，如：
$$
\mathbf{x} = \mathbf{M}\mathbf{X}+\mathbf{t}
\tag{8}
$$
其中$\mathbf{M} \in \mathbb{R}^{2 \times 3}, M_{ij} = \dfrac{T_{ij}}{T_{34}}$，并且$\mathbf{t} \in \mathbb{R}^2$是一个二维向量，表示图像的中心。

在仿射相机中，从不同视角观察到的图像的点之间，可以通过一个简单的$\mathcal{R}$仿射变换进行转换。

![multi][multi]



# 弱透视相机（weak perspective camera）

最常见的，对于相机的假设莫过于是假设其是弱透视相机（weak perspective camera）[10]了。弱透视相机是仿射相机的一种，其最简单的形式是(9)，其中的$Z_{ave}$是相机到物体的平均深度，对于某个场景而言，是一个常数；$f$是焦距。
$$
\mathcal{M} = 
\left[
\begin{matrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & Z_{ave}/f 
\end{matrix}
\right]
\tag{9}
$$
如果转换成公式(8)的那种形式，那么有：
$$
\mathbf{x} = \mathbf{M}_{wp}\mathbf{X} = \dfrac{f}{Z_{ave}} 
\left[
\begin{matrix}
1 & 0 & 0\\
0 & 1 & 0
\end{matrix}
\right] \mathbf{X}
\tag{10}
$$
最终得到投影后的坐标为:
$$
\left[
\begin{matrix}
x \\
y
\end{matrix}
\right] = \dfrac{f}{Z_{ave}} 
\left[
\begin{matrix}
X \\
Y
\end{matrix}
\right]
\tag{11}
$$
在弱透视相机中，我们用平均深度，一个常数$Z_{ave}$去代替了每个点的深度$Z_i$，从而使得分析变得简单。但是要满足弱透视的要求，需要满足几个假设

1. 沿着光轴上的，物体的深度的平均差别，也就是$\Delta Z$，需要远远小于$Z_{ave}$。
2. 视角场(filed of view)（也就是观察某个物体点的夹角，$X/Z_{ave},Y/Z_{ave}$）必须足够小。

下面给出证明。

在考虑所有点的深度情况下，我们在$Z_{ave}$的基础上加上每个点的偏差$\Delta Z$，并且利用泰勒展开，有：
$$
\mathbf{x}_p = \dfrac{f}{Z_{ave}+\Delta Z}
\left[
\begin{matrix}
X \\
Y
\end{matrix}
\right] = 
\dfrac{f}{Z_{ave}} (1 - \dfrac{\Delta Z}{Z_{ave}} + (\dfrac{\Delta Z}{Z_{ave}})^2 - \cdots) 
\left[
\begin{matrix}
X \\
Y
\end{matrix}
\right]
\tag{12}
$$
当$|\Delta Z| << Z_{ave}$时，只有零阶项保留下来了，其他高阶项都趋向于0，其在图像上表现出来的误差就体现在 $\mathbf{x}_{err} = \mathbf{x}_{p} - \mathbf{x}_{wp}$:
$$
\mathbf{x}_{err} = -\dfrac{f}{Z_{ave}}(\dfrac{\Delta Z}{Z_{ave} + \Delta Z})
\left[
\begin{matrix}
X \\
Y
\end{matrix}
\right]
\tag{13}
$$
因此，当焦距$f$比较小时，视角场$\dfrac{X}{Z_{ave}},\dfrac{Y}{Z_{ave}}$足够小，或者物体表面深度的差别$\Delta Z$足够小时，其弱透视模型都可以成立。



-------

# Reference

[1]. http://www.cse.iitd.ernet.in/~suban/vision/affine/node5.html

[2]. http://www.cse.iitd.ernet.in/~suban/vision/affine/node4.html

[3]. http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/UESHIBA1/node5.html

[4]. http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/UESHIBA1/node4.html

[5]. http://www.cse.iitd.ernet.in/~suban/vision/affine/node3.html

[6].  https://blog.csdn.net/LoseInVain/article/details/102756630 

[7].  https://blog.csdn.net/LoseInVain/article/details/102869987 

[8].  https://blog.csdn.net/LoseInVain/article/details/102756630 

[9].  https://blog.csdn.net/LoseInVain/article/details/102632940 

[10].  https://blog.csdn.net/LoseInVain/article/details/102698703 





[image_pin]: ./imgs/image_pin.jpg
[multi]: ./imgs/multi.jpg