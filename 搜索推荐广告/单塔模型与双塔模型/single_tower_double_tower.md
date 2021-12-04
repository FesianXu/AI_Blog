<div align='center'>
   图片/视频搜索中的单塔模型与双塔模型杂谈
</div>

<div align='right'>
    FesianXu 20210919 at Baidu Search Team
</div>

# 前言

目前深度学习已经在搜索系统中得到了广泛地应用，通常用于对用户Query，网络文档Doc进行Embedding特征提取，随着Transformer的兴起，单塔模型在搜索系统中也得到了崭露头角的机会。本文简单介绍下单塔模型和双塔模型在商业搜索系统中的一些应用场景和一些局限性。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]

-----

我们之前在博文[1]中曾经极为粗略地介绍过搜索系统的基本组成部分，本文我们以图文搜索，视频搜索中的相关性建模作为应用场景例子，简单介绍单塔模型和双塔模型的一些主要应用。

# 双塔模型

相关性指的是衡量用户Query和候选Doc之间的相关程度度量，通常可以划分为若干个档位，比如常见的四个档位：完全相关（3），相关（2），部分相关（1），完全不相干（0）。在一个信息检索系统中，保证Query和Doc之间的强相关性是最基础，也是最根本的技术要求，具有核心的地位。很难想象会有用户，愿意长期对一个检索不到期望的相关文档的搜索引擎买单。衡量一个用户Query是否和某个图片或者视频相关，其实是一件很难的事情，图片和视频的视觉语义复杂，通常会在多个信息源上进行特征建模，比如对视频的标题进行Query和Title文本相关性度量是最为常见和经典的方法。如Fig 1.1所示，文档的标题通常能对文档内容进行较好的概括，因此大多数的搜索系统都会尝试提取Query和Title的相关性特征，我们称之为QT特征。最为基础的QT特征是字面上的短词匹配（Term Matching），通常会对Query和Title进行短词分割，对于Query “人工智能的应用与理论”，可能会被拆分为“人工智能”“理论”“应用”等短词，形成Query Term集合$\mathcal{D}_{Q}$，同时也会拆分文档的Title形成Title Term集合$\mathcal{D}_{T}$，按照$\mathcal{D}_{Q}$和$\mathcal{D}_{T}$的重合度，可以定义出QT的短词匹配度作为基础性特征信号。同时，图片和视频不同一般的网页，网页通常文字内容很丰富，可以通过TF-IDF（Term Frequency–Inverse Document Frequency）特征衡量Query和文档内容中词频作为相关性信号。当然，作为图片和视频，比起一般的网页内容天然有着更为丰富的视觉语义，单纯用Query和Title，或者图片/视频的其他文本信号（比如网页中图片的上下文文本信息，视频的Tag信息，视频生产者对视频的文本摘要等）可能不足以描述整个视频的内容。我们知道网络中存在很多标题党的内容，也即是题文不匹配，很多Title-视频内容，Title-图片内容不一致的问题无法通过这类型的特征解决。目前随着深度学习在计算机视觉领域的兴起，已经有很多特征信号尝试去提取视频/图片的视觉语义从而进行Query-Vision匹配了。

![video_title][video_title]

<div align='center'>
    <b>
        Fig 1.1 无论是什么形式的信息检索，doc的标题大多数时候能提供较为准确的相关性信息。
    </b>
</div>









# Reference

[1]. https://fesian.blog.csdn.net/article/details/116377189

[2]. 





[qrcode]: ./imgs/qrcode.jpg
[video_title]: ./imgs/video_title.png



