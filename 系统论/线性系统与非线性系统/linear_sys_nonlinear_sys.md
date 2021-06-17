<div align='center'>
    线性系统与非线性系统
</div>

<div align='right'>
    FesianXu 2021.06.17 at Baidu search team
</div>

# 前言

我们经常在数学和系统论都会谈到`线性`和`非线性`这两个概念，那么这俩到底在系统中有什么应用呢？笔者尝试在本博文简单谈谈自己的看法。**如有谬误请联系指出，本文遵守[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

----

数学中线性的定义很简单，对于$y_n=f(x_n)$，其中$y_n,f(),x_n$ 分别是第$n$个输出变量，函数和输入变量，那么对于输入变量$x_n$的线性组合$x=\sum_{i=0}^{N} \alpha_{i} x_i$有式子(1.1)，我们可以发现输入变量的线性叠加会体现到输出变量的线性叠加上，这个性质取决于系统函数$f()$的性质。
$$
\sum_{i=0}^{N} \alpha_{i} y_i = \sum_{i=0}^{N} \alpha_i f(x_i) = f(\sum_{i=0}^{N} \alpha_{i} x_i)
\tag{1.1}
$$
如Fig 1.1所示，假如某个系统符合线性性质，那么从理论上只需要测量两个输入变量$x_1$和$x_2$以及对应的系统响应$y_1=f(x_1),y_2=f(x_2)$，那么该系统在所有有效范围内的输入下的响应皆可以通过插值的方法进行预测。如可以采样红色点$x_1$和$x_2$，中间的蓝色点$\alpha_1 x_1+\alpha_2 x_2$就是通过采样点进行插值的输入变量，而输出$f(\alpha_1 x_1+\alpha_2 x_2)=\alpha_1 f(x_1)+\alpha_2 f(x_2)$。

![linear_sys][linear_sys]

<div align='center'>
    <b>
    Fig 1.1 系统线性性示意。
    </b>
</div>

然而，现实生活中真正符合线性性的系统少之又少，而大部分系统都是非线性的，如Fig 1.2所示。为了通过采样有限的观察点（observation）对非线性系统$g()$进行估计，可以用若干个局部线性去组合，去模拟建模非线性系统的响应函数。如果通过这种方式，那么最理想的情况下，我们的采样点应该是每个线性曲面的边界点上，如Fig 1.2虚线框内的红色，蓝色，橙色点。遗憾的是，对于一个未知的非线性系统，你无法具体知道局部线性拟合的方式，因此也无法知道理想的采样方式。通过密集均匀采样，即便在不知道非线性系统相应函数的时候，也可以对系统进行拟合，然而密集均匀采样成本极大，在系统输入是高维数据情况下，更是会出现维度灾难的问题。为了解决这种问题，存在许多更为先进的采样方法。

![local_linear][local_linear]

<div align='center'>
    <b>
    Fig 1.2 给定一个未知的非线性系统，可以用局部线性化的方式去对整个非线性系统进行模拟建模。
    </b>
</div>





[qrcode]: ./imgs/qrcode.jpg
[linear_sys]: ./imgs/linear_sys.jpg
[local_linear]: ./imgs/local_linear.png



