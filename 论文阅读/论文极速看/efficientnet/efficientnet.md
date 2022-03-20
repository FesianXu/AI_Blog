<div align="center">
  【论文极速读】 Efficient Net：一种组合扩大卷积网络规模的方法
</div>

<div align="right">
  FesianXu 20220313 at Baidu Search Team
</div>

# 前言

最近笔者需要基于Efficient Net作为图片编码器进行实验，之前一直没去看原论文，今天抽空去翻了下原论文，简单记下笔记。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----

前人的研究证实了，一种有效提高卷积网络性能的方法是：

1. 增大卷积网络的深度，比如经典的ResNet [2]
2. 增大卷积网络的宽度，比如Wide Residual Network [3]
3. 增大卷积网络输入图片的分辨率，比如[4,5,6]

然而单独提高深度/宽度/分辨率很容易达到性能的饱和，如Fig 1.1所示，EfficientNet考虑如何以一种合适的方式，组合性地同时提高深度+宽度+分辨率。对于一个卷积网络而言，第$i$层的卷积网络可定义为$Y_i = \mathcal{F}_i(X_i)$，其中$X_i,Y_i$为输入和输出，而$\mathcal{F}_i(\cdot)$为卷积算子，其中$X_i$的形状为$<H_i,W_i,C_i>$。因此一个完整的卷积网络$\mathcal{N}$可以表示为若干个卷积层的层叠，表示为:
$$
\mathcal{N} = \mathcal{F}_k \odot \cdots \mathcal{F}_2 \odot \mathcal{F}_1 (X) = \bigodot_{j=1\cdots k} \mathcal{F}_j (X)
\tag{1-1}
$$
考虑到当前流行的卷积网络设计方法，都会考虑将整个卷积网络划分为多个stage，然后每个stage内进行若干层相同结构卷积层的层叠，那么卷积网络可以表示为(1-2)，其中的$\mathcal{F}_i^{L_i}$表示第$i$个stage的$\mathcal{F_i}$层叠了$L_i$次。那么提高网络的宽度可以认为是提高$C_i$，提高深度就是提高$L_i$，提高分辨率就是在提高$H_i,W_i$，当然这一切的前提是固定卷积算子$\mathcal{F}_i$的架构。即便只考虑$L_i,H_i,W_i,C_i$，整个超参数空间依然是非常地大，作者在本文正是提供了一种策略对这些超参数进行搜索，最终搜索出来的参数$d,w,r$需要满足最优化目标和条件(1-3)。
$$
\mathcal{N} = \bigodot_{i=1\cdots s} \mathcal{F}_{i}^{L_i} (X_{<H_i, W_i, C_i>})
\tag{1-2}
$$

$$
\max_{d,w,r} \mathrm{Accuracy}(\mathcal{N}(d,w,r)) \\
s.t. \ \ \mathcal{N}(d,w,r) = \bigodot_{i=1\cdots s} \mathcal{\hat{F}}_{i}^{d \cdot \hat{L}_i} (X_{<r \cdot \hat{H}_i, r \cdot \hat{W}_i, w \cdot \hat{C}_i >}) \\
\mathrm{Memory}(\mathcal{N}) \leq \mathrm{target\_memory}  \\
\mathrm{FLOPs}(\mathcal{N}) \leq \mathrm{target\_flops}
\tag{1-3}
$$

可看到最终搜索出来的参数$d,w,r$将会对深度进行增大$d \cdot \hat{L}_i$，同时会对分辨率进行增大$r \cdot \hat{H}_i, r \cdot \hat{W}_i$，对宽度进行增大$w \cdot \hat{C}_i$，当然也要满足对计算量FLOP和内存占用的约束。


![standalone_wdr_enlarge][standalone_wdr_enlarge]

<div align='center'>
  <b>
    Fig 1.1 单独提高宽度，深度和分辨率容易达到性能的饱和。
  </b>
</div>

$$
\begin{aligned}
\mathrm{Depth}: & d= \alpha^{\phi} \\
\mathrm{Width}: & w= \beta^{\phi} \\
\mathrm{Resolution}: & r= \gamma^{\phi} \\
s.t. \ \ & \alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2 \\
& \alpha \geq 1, \beta \geq 1, \gamma \geq 1
\end{aligned}
\tag{1-4}
$$

正如(1-4)公式所示，作者通过在小型网络上进行网格搜索，搜索出基本参数$\alpha,\beta,\gamma$，当然这些参数需要满足约束，特别是$\alpha \cdot \beta^{2} \cdot \gamma^{2} \approx 2$。这个约束用于限制网络的计算复杂度FLOPs，我们知道一般的卷积网络的计算复杂度正比于$d,w^2,r^2$，也即是$flops \sim d \cdot w^2 \cdot r^2$，通过约束其约等于2，可以保证最后的计算量控制在$(d \cdot w^2 \cdot r^2)^{\phi}$，也即是约等于$2^{\phi}$。作者采用的参数搜索策略是这样的：

1. 首先控制$\phi=1$，然后以MBConv [7]为基础模块，对$\alpha, \beta,\gamma$进行网格搜索，最后搜索出$\alpha=1.2, \beta=1.1, \gamma=1.15$，这个$\phi=1$的网络命名为`EfficientNet B0`。
2. 固定$\alpha, \beta,\gamma$，通过选择不同的$\phi$从而实现对EfficientNet的尺度增大，从而得到`EfficientNet B0` ~`EfficientNet B7`。

最终得到的`EfficientNet B0`网络见Fig 1.2所示，其网络结构图可见Fig 1.3，最终各个版本的`EfficientNet`的$d,w,r$见Table 1。

![efficientnet_b0][efficientnet_b0]

<div align='center'>
  <b>
    Fig 1.2 Efficient Net B0基础网络的深度，宽度和分辨率设置，其基础模块采用MBConv。
  </b>
</div>
<div align='center'>
  <b>
    Table 1 搜索出来的各个维度的参数系数 [9]。
  </b>
</div>


| Model-type      | width_coefficient | depth_coefficient | resolution | dropout_rate |
| --------------- | ----------------- | ----------------- | ---------- | ------------ |
| Efficientnet-b0 | 1.0               | 1.0               | 224        | 0.2          |
| Efficientnet-b1 | 1.0               | 1.1               | 240        | 0.2          |
| Efficientnet-b2 | 1.1               | 1.2               | 260        | 0.3          |
| Efficientnet-b3 | 1.2               | 1.4               | 300        | 0.3          |
| Efficientnet-b4 | 1.4               | 1.8               | 380        | 0.4          |
| Efficientnet-b5 | 1.6               | 2.2               | 456        | 0.4          |
| Efficientnet-b6 | 1.8               | 2.6               | 528        | 0.5          |
| Efficientnet-b7 | 2.0               | 3.1               | 600        | 0.5          |
| Efficientnet-b8 | 2.2               | 3.6               | 672        | 0.5          |
| Efficientnet-l2 | 4.3               | 5.3               | 800        | 0.5          |

![efficientnet_b0_detail][efficientnet_b0_detail]

<div align='center'>
  <b>
    Fig 1.3 Effcient Net B0的网络结构图，其中的MBConv1和MBConv6结构有些细微区别。[8]
  </b>
</div>



# Reference

[1]. Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." In *International conference on machine learning*, pp. 6105-6114. PMLR, 2019.

[2].He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. CVPR, pp. 770–778, 2016.

[3]. Zagoruyko, S. and Komodakis, N. Wide residual networks. BMVC, 2016.

[4]. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. CVPR, pp. 2818–2826, 2016.

[5]. Zoph, B., Vasudevan, V., Shlens, J., and Le, Q. V. Learning transferable architectures for scalable image recognition. CVPR, 2018.

[6]. Huang, Y., Cheng, Y., Chen, D., Lee, H., Ngiam, J., Le, Q. V., and Chen, Z. Gpipe: Efficient training of giant neural networks using pipeline parallelism. arXiv preprint arXiv:1808.07233, 2018.

[7]. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. CVPR, 2018.

[8]. https://www.researchgate.net/figure/The-structure-of-an-EfficientNetB0-model-with-the-internal-structure-of-MBConv1-and_fig2_351057828

[9]. https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py





[qrcode]: ./imgs/qrcode.jpg
[standalone_wdr_enlarge]: ./imgs/standalone_wdr_enlarge.png

[efficientnet_b0]: ./imgs/efficientnet_b0.png

[efficientnet_b0_detail]: ./imgs/efficientnet_b0_detail.png
[efficientnet_b0_detail]: ./imgs/efficientnet_b0_detail.png

