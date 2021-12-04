<div align='center'>
  【论文极速读】MoCo v3: MoCo机制下Transformer模型的训练不稳定现象
</div>

<div align='right'>
  FesianXu 20211015 at Baidu search team
</div>

# 前言

之前笔者在[1]中介绍过MoCo v1模型通过解耦`batch size`和负样本队列大小，从而实现超大负样本队列的对比学习训练方案；在[2]中我们提到了当前对比学习训练中提高负样本数量的一些方法；在[3]中提到了将MoCo扩展到多模态检索中的方案。在本文，我们介绍下MoCo v3，一种尝试在Transformer模型中引入MoCo机制的方法，并且最重要的，介绍其中作者得到的一些训练的小技巧（Trick）。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]



----



MoCo的基本原理，包括其历史来龙去脉在前文中[1,2,3]中已经介绍的比较充足了，本文就不再进行赘述。本文主要介绍下MoCo v3 [4]中的一些新发现。MoCo v3中并没有对模型或者MoCo机制进行改动，而是探索基于Transformer的ViT（Visual Transformer）模型[5,6]在MoCo机制下的表现以及一些训练经验。作者发现ViT在采用MoCo机制的训练过程中，很容易出现不稳定的情况，并且这个不稳定的现象受到了学习率，`batch size`和优化器的影响。如Fig 1.所示，在batch size大于4096的时候已经出现了明显的剧烈抖动，如Table 1.所示，我们发现在`bs=2048`时候取得了最好的测试性能，继续增大batch size反而有很大的负面影响，这个结论和MoCo v1里面『batch size越大，对比学习效果越好』相悖，如Fig 2.所示。这里面的大幅度训练抖动肯定是导致这个结论相悖的罪魁祸首。这个抖动并不容易发现，因为在`bs=4096`时候，模型训练最终能收敛到和`bs=1024,2048`相同的水平，但是泛化效果确实会存在差别。

![train_curve_bs][train_curve_bs]

<div align='center'>
  <b>
    Fig 1. ViT在MoCo训练过程中，不同batch size情况下的训练曲线，我们发现在大batch size情况下很容易出现稳定性问题。
  </b>
</div>
<div align='center'>
  <b>
   Table 1. ViT在不同batch size下训练出的模型测试结果。
  </b>
</div>


| batch       | 1024 | 2048     | 4096 | 6144 |
| ----------- | ---- | -------- | ---- | ---- |
| linear acc. | 71.5 | **72.6** | 72.2 | 69.7 |

![mocov1_result][mocov1_result]

<div align='center'>
  <b>
    Fig 2. 在MoCo v1中，随着batch size的增大，对比学习的结果也逐渐变好。
  </b>
</div>

不仅仅是batch size，学习率也会导致ViT训练的不稳定，如Fig 3.所示，我们发现在较大的学习率下训练曲线存在明显的抖动，而最终的训练收敛位置却差别不大。在测试结果上看，则会受到很大的影响。如果将优化器从`AdamW`更换到`LAMB`优化器，那么结果也是类似的，如Fig 4.所示，只是可以采用更大的学习率进行训练了。

![train_curve_lr][train_curve_lr]

<div align='center'>
  <b>
    Fig 3. ViT以MoCo机制训练，在不同学习率下的训练曲线和对应测试结果。
  </b>
</div>

![train_curve_opt][train_curve_opt]

<div align='center'>
  <b>
    Fig 4. ViT以MoCo机制训练，在采用LAMB优化器的情况下，不同学习率下的训练曲线和对应测试结果。
  </b>
</div>

这种出现训练时的剧烈抖动，很可能是梯度剧变导致的，因此作者对ViT的第一层和最后一层的梯度的无穷范数进行了统计。注意到无穷范数相当于求所有梯度值中绝对值的最大值，也即是如(1)所示。结果如图Fig 5.所示，我们发现的确会存在有梯度的骤变，而且总是第一层先发生，然后经过约数十个step之后传递给了最后一层。因此，导致训练曲线剧烈抖动的原因可能是ViT的Transformer的第一层梯度不稳定导致。
$$
||x||_{\infty} = \max_{1 \leq i \leq n} |x_{i}|
\tag{1}
$$
![grad_vit][grad_vit]

<div align='center'>
  <b>
    Fig 5. ViT在训练过程中第一层和最后一层的梯度无穷范数。
  </b>
</div>

考虑到在ViT中的第一层是将`patch`映射到`visual token`，也就是一层FC全连接层，如图Fig6.所示。作者在MoCo v3里面的做法也很直接，直接将ViT的第一层，也即是从`Patch`到`Visual Token`的线性映射层随机初始化后固定住，不参与训练。

![vit][vit]

<div align='center'>
  <b>
    Fig 6. 在ViT中通过FC层将图片patch线性映射到了visual token，从而输入到Transformer。
  </b>
</div>

这个做法挺奇怪的，但是实验结果表明在固定住了线性映射层之后，的确ViT的训练稳定多了，如Fig 7.所示，训练曲线的确不再出现诡异的剧烈抖动，最主要的是其测试结果也能随着学习率的提高而增大了，并且同比`learned path proj.`的情况还更高。

![train_curve_fixed_fc_lr][train_curve_fixed_fc_lr]

<div align='center'>
  <b>
    Fig 7. 在固定住了patch映射到visual token的线性映射层之后，训练曲线不再出现明显的剧烈抖动。
  </b>
</div>

这种现象还是蛮奇怪的，也就是说即便不训练这个`patch projection layer`，模型的性能也不会打折，而且还会更加稳定。作者给出的解释就是目前这个映射是完备的（complete），甚至是过完备（over-complete）的，以$16 \times 16 \times 3$​的patch，$768$的`visual token`为例子，那么这个映射矩阵就是$\mathbf{M} \in \mathbb{R}^{768 \times 768}$​的。也就是说对于所有可能的`patch`来说，可能在随机的$\mathbf{M}$​​​​​中就已经有着近似的唯一输出对应，即便这个映射可能不保留太多的视觉语义信息，但是也保留了原始的视觉信息，不至于损失原始信息。但是正如作者最后所说的，这个『trick』只是缓解了问题，但是并没有解决问题，显然这个问题出现在了优化阶段，而固定FC层减少了解空间提高了其稳定性。在更大的学习率下，还是会受到相同的不稳定现象，对该现象的研究值得继续深究。

笔者在大规模的对比学习训练过程中也遇到过类似的训练曲线抖动（虽然没有那么剧烈），但是笔者发现可能是温度系数的剧烈变化导致的，我们以后再继续讨论下温度系数的影响。




# Reference

[1]. https://fesian.blog.csdn.net/article/details/119515146

[2]. https://fesian.blog.csdn.net/article/details/120039316

[3]. https://fesian.blog.csdn.net/article/details/120364242

[4]. Chen, Xinlei, Saining Xie, and Kaiming He. "An empirical study of training self-supervised vision transformers." *arXiv preprint arXiv:2104.02057* (2021).

[5]. https://blog.csdn.net/LoseInVain/article/details/116031656

[6]. Dosovitskiy, Alexey, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani et al. “An image is worth 16x16 words: Transformers for image recognition at scale.” arXiv preprint arXiv:2010.11929 (2020).









[qrcode]: ./imgs/qrcode.jpg
[exp_result]: ./imgs/exp_result.png

[train_curve_bs]: ./imgs/train_curve_bs.png

[mocov1_result]: ./imgs/mocov1_result.png
[train_curve_lr]: ./imgs/train_curve_lr.png
[train_curve_opt]: ./imgs/train_curve_opt.png
[grad_vit]: ./imgs/grad_vit.png
[vit]: ./imgs/vit.png
[train_curve_fixed_fc_lr]: ./imgs/train_curve_fixed_fc_lr.png

