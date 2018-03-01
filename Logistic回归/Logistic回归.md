<h1 align = "center">Logistic regression(逻辑斯蒂回归)</h1>

## 前言
** 线性回归是一种回归(Regression)方法，常用于预测连续输出，而对于离散的分类输出就难以解决。因此在基于线性回归的前提下，引入了激活函数的概念，形成了Logistic回归。在这里简述自己对Logistic回归的所见所学。 **

**如有谬误，请联系指正。转载请注明出处。**

*联系方式：*
**e-mail**: `FesianXu@163.com`
**QQ**: `973926198`
**github**: `https://github.com/FesianXu`

*******************************************************

# Logistic regression
**线性回归（Linear Regression）**是一种回归方法，常用于预测连续输出，而对于离散的分类输出，则爱莫能助。为了解决线性分类问题，引入了sigmoid激活函数，形成了Logistic回归方法。所谓的sigmoid函数，既是形式如：
$$
\sigma(x) = \frac{1}{1+e^{-\alpha x}}
\tag{1.1}
$$
的式子，通常$\alpha$取1。图像如下：
<div align=center>![sigmoid][sigmoid]</div>

而基本的线性回归的式子我们都知道，如下：
$$
\theta(x) = \theta^Tx+b
\tag{1.2}
$$
其中，$\theta^T=(\theta_1, \theta_2,\cdots,\theta_n)$， $x=(x_1,x_2,\cdots,x_n)$，其中n是特征维数，b为实数。


将线性回归和激活函数sigmoid结合起来，就得到了：
$$
f_\theta(x) = \frac{1}{1+e^{-(\theta^Tx+b)}}
\tag{1.3}
$$

**我们可以看出，其实sigmoid激活函数的作用可以看成是将$\theta^Tx+b$映射到$(0,1)$区间中，注意到，因为这个取值区间，我们可以将sigmoid函数看成是将score映射到概率分布的函数，也就是说$f_\theta(x)$可以看成是概率值。**从这个角度出发，我们定义：

1. $P(y_i=1|x_i) = \sigma(\theta^Tx+b) \tag{1.4}$
2. $P(y_i=0|x_i) = 1-P(y_i=1|x_i)=1-\sigma(\theta^Tx+b) \tag{1.5}$
其中，下标i表示第$i$个样本，$x_i$表示第$i$个样本的特征，$x_i \in R^n$. $y_i$表示第$i$个样本的标签，$y_i \in \{0,1\}$。

# 极大似然估计
**极大似然估计（Maximum Likelihood Estimate,MLE）**是频率学派常用于对给定模型进行参数估计的方法，在logistic回归中用于(估计)学习出$\theta$和$b$的值。极大似然法的基本思想很简单，**假定发生的事情（也就是样本）肯定是符合模型的**，就是**求得参数$\theta$，使得目前所有的样本在这个模型下$f(x;\theta)$发生的概率之积最大**。用到logistic回归上，我们可以得出似然函数：
$$
L(\theta) = \prod_{i=1}^N P(y_i=1|x_i)^{y_i} P(y_i=0|x_i)^{1-y_i}
\tag{2.1}
$$
在实际使用中，因为连乘不如累加好使，我们对等式进行两边求自然对数，得到对数似然函数，有：
$$
\ln L(\theta) = \ln \prod_{i=1}^N P(y_i=1|x_i)^{y_i} P(y_i=0|x_i)^{1-y_i} \\
= \sum_{i=1}^N y_i \ln P(y_i=1|x_i)+(1-y_i) \ln P(y_i=0|x_i) \\
= \sum_{i=1}^N y_i (\ln P(y_i=1|x_i)-\ln P(y_i=0|x_i))+\ln P(y_i=0|x_i) \\
= \sum_{i=1}^N y_i \ln \frac{P(y_i=1|x_i)}{P(y_i=0|x_i)}+\ln P(y_i=0|x_i) \\
= \sum_{i=1}^N y_i \ln \frac{P(y_i=1|x_i)}{1-P(y_i=1|x_i)}+\ln P(y_i=0|x_i) \\
= \sum_{i=1}^N y_i (\theta^T x_i+b)+\ln (1-\sigma(\theta^Tx+b)) \\
= \sum_{i=1}^N y_i (\theta^T x_i+b)+ \ln \frac{1}{1+e^{\theta^Tx_i+b}} \\
= \sum_{i=1}^N y_i (\theta^T x_i+b)-\ln (1+e^{\theta^Tx_i+b}) \\
\tag{2.2}
$$
根据极大似然法的思想，对对数似然函数求最大值，按照传统的方法，我们是对$\ln L(\theta)$求导数后令其为0解得极值点，但是我们会发现$\frac{\partial{\ln L(\theta)}}{\partial{\theta}}=0$没有解析解，所以我们需要通过梯度下降法去近似求得其数值解。关于梯度下降法的介绍见[《随机梯度下降法，批量梯度下降法和小批量梯度下降法以及代码实现》](http://blog.csdn.net/loseinvain/article/details/78243051)。于是现在我们需要求得$\frac{\partial{\ln L(\theta)}}{\partial{\theta}}$。
$$
\frac{\partial{\ln L(\theta)}}{\partial{\theta}} = \sum_{i=1}^N y_ix_i-\frac{1}{1+e^{\theta^Tx_i+b}} x_i e^{\theta^Tx_i+b} \\
= \sum_{i=1}^N y_ix_i-x_i \frac{1}{1+e^{-(\theta^Tx_i+b)}} \\
= \sum_{i=1}^N y_ix_i-\sigma(\theta^Tx_i+b)x_i \\
= \sum_{i=1}^N x_i(y_i-\sigma(\theta^Tx_i+b))
\tag{2.3}
$$
所以参数的更新公式为：
$$
\theta := \theta - \eta \frac{\partial{\ln L(\theta)}}{\partial{\theta}} = \theta - \eta \sum_{i=1}^N x_i(y_i-\sigma(\theta^Tx_i+b))
\tag{2.4}
$$
$$
b := b - \eta \frac{\partial{\ln L(\theta)}}{\partial{b}} = b-\eta \sum_{i=1}^N (y_i-\sigma(\theta^Tx_i+b))
\tag{2.5}
$$






[sigmoid]: ./imgs/sigmoid.jpg







