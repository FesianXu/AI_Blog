<h1 align = "center">Logistic回归</h1>

## 前言
** 线性回归是一种回归(Regression)方法，常用于预测连续输出，而对于离散的分类输出就难以解决。因此在基于线性回归的前提下，引入了激活函数的概念，形成了Logistic回归。在这里简述自己对Logistic回归的所见所学。 **
**如有谬误，请联系指正。转载请注明出处。**
*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

****

# Logistic回归
线性回归（Linear Regression）是一种回归方法，常用于预测连续输出，而对于离散的分类输出，则爱莫能助。为了解决线性分类问题，引入了sigmoid激活函数，形成了Logistic回归方法。所谓的sigmoid函数，既是形式如：
$$
\sigma(x) = \frac{1}{1+e^{-\alpha*x}}
$$
的式子，通常$\alpha$取1。图像如下：
<div align=center>![sigmoid][sigmoid]</div>

而基本的线性回归的式子我们都知道，如下：
$$
\theta(x) = \theta^Tx+b
$$
其中，$\theta^T=(\theta_1, \theta_2,\cdots,\theta_n)$， $x=(x_1,x_2,\cdots,x_n)$，其中n是特征维数，b为实数。

将线性回归和激活函数sigmoid结合起来，就得到了：
$$
f_\theta(x) = \frac{1}{1+e^{-(\theta^Tx+b)}}
$$
（这里采用的是**机器学习的解释方式**，实际上在**贝叶斯学派**经典PRML中，有对Logistic回归的**贝叶斯解释**，有时间我们以后再谈）
因此，我们现在需要学习的便是$\theta$和$b$的取值了。



*****

# Logistic回归的参数学习










[sigmoid]: ./imgs/sigmoid.jpg







