<div align='center'>
    从手写字符匹配开始，简要解释局部仿射变换(local affine transformation)
</div>

<div align='right'>
    FesianXu 2020/09/07 at UESTC
</div>



# 前言

最近笔者看论文[1]的时候发现有个术语`local affine transformation`，也就是所谓的局部仿射变换，仿射变换笔者之前有过研究[2]，还算是比较清楚，但是谈到什么是“局部”仿射变换，就没有头绪了。后面笔者查找资料[3]后，终于有所了解，因此简要笔记与此，希望对大家有所帮助。**如有谬误，请联系指出，转载请联系作者并注明出处**。

$\nabla$ 联系方式：
**e-mail**: [FesianXu@gmail.com](mailto:FesianXu@gmail.com)
**QQ**: 973926198
github: https://github.com/FesianXu

----

注意：本文为了描述局部仿射变换的应用场景，会介绍一下手写字符匹配，如果需要快速了解局部仿射变换，请直接跳到最后两章。

# 回顾仿射变换

我们之前在博文[2,4]中谈到过仿射变换，简单来说就是仿射变换是几何图形之间保持平行线，共线性的一种线性变换，在二维情况下，我们用齐次坐标系[4]可以表示为：
$$
\begin{aligned}
\mathbf{T}_{\mathrm{A}} &= 
\left[
\begin{matrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
0 & 0 & 1
\end{matrix}
\right] \in \mathbb{R}^{3 \times 3} \\
\left[
\begin{matrix}
\mathrm{X}^{\prime} \\
\mathrm{Y}^{\prime} \\
1
\end{matrix}
\right] &= 
\mathbf{T}_{\mathrm{A}}
\left[
\begin{matrix}
\mathrm{X} \\
\mathrm{Y} \\
1
\end{matrix}
\right]
\end{aligned}
\tag{1.1}
$$
其中满足约束：
$$
\mathrm{det}(
\left[
\begin{matrix}
a_{11} & a_{12} \\
a_{21} & a_{22} 
\end{matrix}
\right]
) \neq 0
\tag{1.2}
$$
一般来说，仿射变换包括了 旋转，平移，切变，尺度放缩等基本几何操作 [4]。当然，一般我们提到仿射变换时，都默认是全局的仿射变换（global affine transformation）。

----

# 手写字符匹配

在说到局部仿射变换之前，我们先提一下手写字符匹配这个应用场景，在这个场景里面，局部仿射变换有着其作用之地。对于某个手写字符，比如“大”字，我们可以设置一个标准的参考字符，我们分别用$R,S$表示参考字符和输入字符，假设这两个字符的特征点为：
$$
\begin{aligned}
R &= (\vec{r}_1,\cdots,\vec{r}_i,\cdots,\vec{r}_K) \\
S &= (\vec{r}_1,\cdots,\vec{r}_j,\cdots,\vec{r}_{K^{\prime}}) 
\end{aligned}
\tag{2.1}
$$
其中$K,K^{\prime}$表示特征点的数量，如Fig 2.1所示，为了使得匹配关系尽可能一致，我们的目标方程是：
$$
\min_{\Gamma} \sum_{i=1}^{K}|\vec{s_{\Gamma(i)}}-\vec{r_{i}}|
\tag{2.2}
$$
其中$\Gamma(i)$表示了从$R$到$S$的特征点映射关系，即是$\Gamma(i) \rightarrow j$，不失一般性地假设$K = K^{\prime}$，此时$\Gamma(i)$为单位映射，我们有：
$$
\min \sum_{i=1}^{K}|\vec{s_i}-\vec{r_{i}}|
\tag{2.3}
$$
此时变形向量可以表示为：
$$
\vec{d}_{i} = \vec{s}_i-\vec{r}_i, i \in [1,K]
\tag{2.4}
$$
![dvf][dvf]

<div align='center'>
    <b>
        Fig 2.1 “大”字的参考字符和输入字符之间存在一定的匹配关系，当然，这两类字符的特征点数不一定一致，但是可以进行特定的采样使得其特征点数一致。
    </b>
</div>

注意到一种现象，在式子(2.3)中其实考虑的是每个点之间的匹配关系，我们称之为全局匹配关系，而在手写字体和参考字体的真实匹配过程中，其实可能存在局部的匹配关系，即是某些部分的匹配比较接近，而某些部分的匹配不够接近，如果把这两个结果一起优化（即是全局），则优化过程可能对局部优化较好的结果产生不好的影响。为了避免全局信息对局部信息产生影响，此时需要考虑局部的匹配结果优化。

# 局部仿射变换

我们假设输入的字符特征点可以用参考字符特征点的仿射变换表示，那么有：
$$
\vec{s}_i = \mathbf{A}_i\vec{r}_i+\vec{\epsilon}_i, i \in [1,K]
\tag{3.1}
$$
其中$\mathbf{A}_i \in \mathbb{R}^{3 \times 3}$是二维的仿射变换矩阵，用齐次坐标系表示，而$\vec{\epsilon}_i$是残差。那么对于每个特征点$i$而言，都存在这样一个二维仿射变换矩阵$\mathbf{A}_i$，使得参考字符和输入字符的特征点都可以用仿射变换进行关联，如Fig 3.1所示。

可知对于$j \neq i$而言，也有：
$$
\vec{s}_j = \mathbf{A}_j\vec{r}_j+\vec{\epsilon}_j, j \in [1,K], i \neq j
\tag{3.2}
$$
那么为了优化得到第$i$个特征点的仿射矩阵，我们有：
$$
\begin{aligned}
\Psi_{i} &= \sum_{j=1}^{K} \omega_{ij}||\vec{\epsilon}_j||^2 \\
&= \sum_{j=1}^K \omega_{ij}||\vec{s}_j-\mathbf{A}_j\vec{r}_j||^2
\end{aligned}
\tag{3.3}
$$
有最优化过程：
$$
\min_{\mathbf{A}_i} \Psi_{i}
\tag{3.4}
$$
我们发现，该优化过程需要考虑全局的特征点之间的仿射关系，而这个考虑关联的程度由权重系数$\omega_{ij}$控制，一般来说，我们用高斯函数[5]表示这个权重大小，也就是距离$i$越远的特征点$j$其权重越小，反之亦然。有：
$$
\omega_{ij} = \exp(-||\vec{r}_i-\vec{r}_j||^2 / \theta^2)
\tag{3.5}
$$
我们称$\theta$为窗口大小，这个决定了特征点之间距离和权重系数之间的放缩关系。

通过这个权重参数$\omega_{ij}$，我们在优化$\mathbf{A}_{i}$的时候，更多地考虑的是以特征点$i$为圆心，以$\theta$为半径的局部的特征点优化，而不是全局信息，这个就称之为 **局部仿射变换**。

![local_affine_transformation][local_affine_transformation]

<div align='center'>
    <b>
        Fig 3.1 输入字符和参考字符的特征点之间可以用仿射变换进行关联。
    </b>
</div>

容易发现，当$\theta \rightarrow \infty$时，$\omega_{ij} = 1$，此时等价于全局仿射变换，当$\theta \rightarrow 0$时，$\omega_{ij} = 0$，此时意味着所有匹配都准确，这个是不可能的。



# 总结

总结来说，全局仿射变换和局部仿射变换的区别在于，在局部仿射变换过程中，求解仿射变换矩阵的参考点是局部的，在对于某些存在局部变形的应用中，会更为适合用局部仿射变换去求解局部仿射变换矩阵。



# Reference

[1]. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., & Sebe, N. (2019). First order motion model for image animation. In *Advances in Neural Information Processing Systems* (pp. 7137-7147).

[2]. https://blog.csdn.net/LoseInVain/article/details/104533575

[3]. Wakahara, T. (1988, January). Online cursive script recognition using local affine transformation. In *9th International Conference on Pattern Recognition* (pp. 1133-1134). IEEE Computer Society.

[4]. https://blog.csdn.net/LoseInVain/article/details/102756630

[5]. https://blog.csdn.net/LoseInVain/article/details/80339201



[dvf]: ./imgs/dvf.png

[local_affine_transformation]: ./imgs/local_affine_transformation.png



