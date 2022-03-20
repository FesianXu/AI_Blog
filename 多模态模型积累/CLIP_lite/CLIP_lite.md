<div align='center'>
  【论文极速看】CLIP-Lite：一种不依赖于负样本数量的高效多模态学习方法
</div>

<div align='right'>
  FesianXu 20220201 at Baidu Search Team
</div>

$\nabla$ 联系方式：

e-mail: FesianXu@gmail.com

[github](https://so.csdn.net/so/search?q=github&spm=1001.2101.3001.7020): https://github.com/FesianXu

知乎专栏: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

微信公众号：

![qrcode][qrcode]

-----

传统的CLIP [1]对比学习模型依赖于海量的图文对训练数据，以及每个正样本对应的负样本的数量，为了弥补CLIP模型对于负样本数量的极度依赖，而单纯通过当前`batch size`提供足够的负样本又强烈依赖于显卡资源的现况，有些方案提出采用虚拟`batch size`（即是`memory bank`）进行弥补 [2]。MoCo [3]模型提出采用动量编码器和负样本队列的方式，可以利用训练历史上的负样本，从而扩大了参与训练的负样本数量。

在文章[4]中，作者提出了`CLIP-Lite`，该模型通过Jensen-Shannon散度对互信息进行下界估计，而不是像`CLIP`采用infoNCE对互信息进行估计。互信息（Mutual Information,  MI）描述了『在知道某个随机变量$Y$​​后，对于另一个随机变量$X$​​​的不确定性的减少程度』，对于离散随机变量$X,Y$​​而言，其联合概率分布为$P_{X,Y}(x,y)$​​，其互信息可表示为$I(X;Y)$​​，那么：
$$
\begin{aligned}
I(X;Y) &= \sum_{x,y}P_{XY}(x,y)\log(\dfrac{P_{XY}(x,y)}{P_{X}(x)P_{Y}(y)}) \\
&= \mathbb{E}_{P_{XY}}[\log(\dfrac{P_{XY}}{P_{X}P_{Y}})]
\end{aligned}
\tag{1-1}
$$
定义随机变量的熵（entropy）和条件熵（conditional entropy）为：
$$
\begin{aligned}
H(X) &= -\sum_{x} P_{X}(x) \log(P_{X}(x)) = -\mathbb{E}_{P_X}[\log(P_X)] \\
H(X|Y) &= \sum_{y} P_{Y}(y) [-\sum_{x}P_{X|Y}(x|y)\log(P_{X|Y}(x|y))] = \mathbb{E}_{P_Y}[-\mathbb{E}_{P_{X|Y}}[\log{P_{X|Y}}]]
\end{aligned}
\tag{1-2}
$$
其中有:
$$
P_{X|Y}(x|y) = \dfrac{P_{XY}(x,y)}{P_{Y}(y)}
\tag{1-3}
$$
联合(1-2)和(1-3)，我们有：
$$
\begin{aligned}
I(X;Y) &= H(X) - H(X|Y) \\
&= -\sum_{x} P_{X}(x) \log(P_{X}(x)) - \sum_{y} P_{Y}(y) [-\sum_{x}P_{X|Y}(x|y)\log(P_{X|Y}(x|y))] \\
&= -\sum_{x} P_{X}(x) \log(P_{X}(x)) + \sum_{y}\sum_{x}P_{XY}(x,y)\log(P_{X|Y}(x|y)) \\
&= -\sum_{y}\sum_{x}P_{XY}(x,y) \log(P_{X}(x)) + \sum_{y}\sum_{x}P_{XY}(x,y)\log(P_{X|Y}(x|y)) \\
&= \sum_{y}\sum_{x}P_{XY}(x,y) [\log(P_{X|Y}(x|y))-\log(P_{X}(x))] \\
&= \sum_{y}\sum_{x}P_{XY}(x,y) \log(\dfrac{P_{XY}(x, y)}{P_X(x)P_Y(y)})
\end{aligned}
\tag{1-4}
$$
因此互信息可以理解为在知道了额外信息（随机变量$Y$​）后，对于随机变量$X$​的不确定性的减少程度。显然，当且仅当$P_{XY}(x,y)=P_{X}(x)P_{Y}(y)$的时候，有 $I(X;Y)=0$​，此时可以理解为随机变量$X,Y$​​​是完全独立的。

**对互信息进行优化在表征学习中有着广泛地应用，通过最大化互信息可以学习到更好的表征**。不难看出，互信息可以用Kullback-Leibler(KL) 散度表示，为：
$$
I(X;Y) = D_{KL}(P_{XY}(x,y)||P_{Y}(y)P_{X}(x))
\tag{1-5}
$$
然而，对高维的连续随机变量进行互信息估计是一件很困难的事情，特别是当联合概率分布和边缘概率分布都是未知分布的情况下。因此存在有一些方法尝试对互信息进行下界估计，通过对下界进行求最优化从而间接达到对互信息进行优化的目的。目前最常用的有几种下界估计，Donsker-Varadhan（DV）下界，infoNCE下界和Jensen-Shannon（JS）散度下界。DV下界 [5,6] 如(1-6)所示，其中的$T_{\omega}(x,y)$​​是一个参数为$\omega$​的神经网络判别器，$T_{\omega}:\mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$​​​，其对$x$​和$y$​​​的相关程度进行判断。
$$
\begin{aligned}
I(X;Y) &:= D_{KL}(P_{XY}(x,y)||P_{Y}(y)P_{X}(x)) \geq \hat{I}^{(DV)}_{\omega}(X;Y) := \mathbb{E}_{P_{XY}}[T_{\omega}(x,y)]-\log(\mathbb{E}_{P_XP_Y}[\exp(T_{\omega}(x,y))])
\end{aligned}
\tag{1-6}
$$
假设$Y=E_{\psi}(X)$是对输入$X$进行求表征的函数，$\psi$是表征神经网络的参数。那么对$\hat{I}^{(DV)}_{\omega}(X;E_{\psi}(X))$进行最优化，需要同时优化$\omega,\psi$。表示为：
$$
(\hat{\omega}, \hat{\psi}) = \arg \max_{\omega, \psi} \hat{I}_{\omega}(X;E_{\psi}(X))
\tag{1-7}
$$
infoNCE下界如式子(1-8)所示：
$$
\hat{I}_{\omega, \psi}^{infoNCE} (X;E_{\psi}(X)) = \mathbb{E}_{\mathbb{P}}[T_{\omega, \psi}(x, E_{\psi}(x))-\mathbb{E}_{\tilde{\mathbb{P}}}[\log{\sum_{x^{\prime}}\exp{(T_{\omega,\psi}(x^{\prime}, E_{\psi}(x)))}}]]
\tag{1-8}
$$
从式子(1-6)和(1-8)中，我们发现DV下界和infoNCE下界都依赖于负样本数量，越多负样本才能有越好的效果（笔者暂时还不太理解DV下界为啥依赖于负样本数量）。而在CLIP-Lite中作者提出用JS下界去替代CLIP中的infoNCE下界，而JS下界不依赖于负样本数量，因此每个正样本至多只需要一个负样本就可以进行下界优化，如Fig 1.1所示。JS下界见式子(1-9):
$$
I(X;E_{\psi}(X)) \geq \hat{I}_{\omega}^{JS}(X; E_{\psi}(X)) = \mathbb{E}_{P(X,E_{\psi}(X))}[-\log{(1+\exp(-T_{\omega}))}] - \mathbb{E}_{P(X)P(E_{\psi}(X))}[\log{(1+\exp(T_{\omega}))}]
\tag{1-9}
$$
![clip_and_clip_lite][clip_and_clip_lite]

<div align='center'>
  <b>
    Fig 1.1 CLIP-Lite对比CLIP，每个正样本只需要一个负样本即可完成有效的表征对比学习。
  </b>
</div>
其中的$T_{\omega}: \mathcal{X} \times E_{\psi}(X) \rightarrow \mathbb{R}$同样是个以$\omega$​为参数的神经网络判别器，用于判断当前输入的图文对是正样本还是负样本，最后的优化目标同样是同时对$\omega$和$\psi$进行优化：
$$
(\hat{\omega}, \hat{\theta}_i, \hat{\theta}_t) = \arg\max_{\omega, \hat{\theta}_i, \hat{\theta}_t} \hat{I}_{\omega}^{JSD} (f_i(x_i;\theta_{i}), f_t(x_t;\theta_{t}))
\tag{1-10}
$$
其中的$f_i(x_i)$和$f_t(x_t)$​是图片和文本的编码器，如Fig 1.2所示。

![clip_lite_image_text_encoder][clip_lite_image_text_encoder]

<div align='center'>
  <b>
    Fig 1.2 图片编码器和文本编码器，通过优化JS下界间接进行互信息优化。
  </b>
</div>



# Reference

[1]. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., … & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020.

[2]. https://fesian.blog.csdn.net/article/details/119515146

[3]. He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9729-9738).

[4]. Shrivastava, Aman, Ramprasaath R. Selvaraju, Nikhil Naik, and Vicente Ordonez. "CLIP-Lite: Information Efficient Visual Representation Learning from Textual Annotations." *arXiv preprint arXiv:2112.07133* (2021).

[5]. M.D Donsker and S.R.S Varadhan. Asymptotic evaluation of certain markov process expectations for large time, iv. Communications on Pure and Applied Mathematics, 36(2):183–212, 1983.

[6]. Hjelm, R. Devon, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. "Learning deep representations by mutual information estimation and maximization." *arXiv preprint arXiv:1808.06670* (2018).





[qrcode]: ./imgs/qrcode.jpg
[clip_and_clip_lite]: ./imgs/clip_and_clip_lite.png
[clip_lite_image_text_encoder]: ./imgs/clip_lite_image_text_encoder.png

