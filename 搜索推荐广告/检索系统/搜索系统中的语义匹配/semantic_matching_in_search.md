<div align='center'>
    搜索系统中的语义匹配
</div>

<div align='right'>
    FesianXu 20210707 at Baidu search team
</div>

# 前言

由Hang Li大佬写的《Semantic Matching in Search》[1]是一篇很好的关于搜索系统相关性的综述性文章。本文主要翻译自该文，并加上一些笔者的读后感和注释。**如有谬误请联系指出，本文遵循[ CC 4.0 BY-SA ](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请附上原文出处链接和本声明并且联系笔者，谢谢**。

$\nabla$ 联系方式：
**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：

![qrcode][qrcode]



----

注：笔者曾在博文[2]中简单介绍了搜索系统主要的包含部分，读者有兴趣可移步阅读。本文成文较早，为2010年，有些技术或者表述已经过时，但是仍不失一般性，是一篇了解搜索系统的好文章。



# 相关性

相关性（Relevance）是搜索系统中保证用户体验的重要因素，其很大程度上决定了一个搜索引擎的成功与否，很难想象会有用户选择一个搜索不出相关检索结果的搜索系统。然而在实际搜索过程中，经常会出现因为query和doc之间不匹配导致的bad case，比如“ny times”这个query并不能很好地和一个仅包含有“New York Times”的doc进行匹配，即便我们知道这两个其实是同一个东西。在搜索系统中，相关性还很大程度上由词语匹配（Term match）决定（笔者：该文章成文2010年，目前据笔者了解，大多数商业搜索系统已经引入了语义匹配，但是词语匹配仍然是具有重要作用。），词语匹配例如词袋技术（bag-of-word）仍然在现代搜索系统中起着主要的作用。可以毫不夸张地说，query和doc之间的不匹配是当今搜索系统中的主要挑战。理想情况下，如果query和doc是主题上相关的，那么搜索系统应该能返回和query精密匹配的doc。

最近，研究者花了很大努力去解决这类型的相关性问题，其成果主要是引出了语义匹配（semantic matching），即是在query和doc匹配过程中，实现更多地对query和doc的内容理解，以及实现对内容丰富后（enriched）的query和doc之间实现更好的匹配。在海量的log数据和机器学习技术的加持下，语义匹配取得了重要的进展。该文主要关注网页搜索中的相关性问题，特别是语义相关性问题，主要介绍了在Query-Doc匹配中涉及到的形式匹配（form aspect），短语匹配（phrase aspect），词义匹配（word sense aspect），主题模型（topic aspect）和结构匹配（structure aspect）。

同时我们要注意到，query和doc的匹配问题不仅仅局限在搜索系统这个应用，在其他诸多相似的问题中同样有着广泛应用，比如问题回答系统（Question Answering， QA），在线计算广告（online advertising），跨语言信息检索（cross-Language information retrieval），机器翻译，推荐系统，连接预测，图像标注，药物设计等等，我们可以认为其最通用的任务模式是：在两个异质物体的匹配都可以通过这个技术进行建模。因此，本文介绍到的技术不仅仅可用于搜索系统，还能在其他领域大施拳脚。



# Query-Doc的错误匹配

一个成功的搜索系统必须在相关性，覆盖率（coverage），时鲜性（freshness）和响应时间（response time），用户界面（user interface）之间都做好。而在这些中，相关性无疑是最为重要的因素，而这也是本文所关注的。搜索系统仍然对词袋技术或者基于词语（term）的技术有着很重的依赖。也就是说，用词袋的方式将Query和Doc都表征为一个词袋向量（terms），









# Reference

[1]. Li, Hang, and Jun Xu. "Semantic matching in search." *Foundations and Trends in Information retrieval* 7, no. 5 (2014): 343-469.

[2]. https://fesian.blog.csdn.net/article/details/116377189

[3]. 



[qrcode]: ./imgs/qrcode.jpg



