<div align='center'>
    Kosmos-1: 通用接口架构下的多模态大语言模型
</div>



<div align='right'>
    FesianXu 20230513 at Baidu Search Team
</div>

# 前言

在大规模语言模型（Large Language Model, LLM）看似要带来新一番人工智能变革浪潮之际，越来越多尝试以LLM作为通用接口去融入各种任务的工作，之前我们在[2]中曾经对其进行过简单介绍，比如尝试用LLM去控制浏览器、搜索引擎甚至是机械臂等。本文介绍的工作kosmos-1是LLM与多模态信号结合的一种尝试，对笔者有所启发，在此给大家进行推荐。**如有谬误请见谅并联系指出，本文遵守[CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

**e-mail**: FesianXu@gmail.com

**github**: https://github.com/FesianXu

**github page**: https://fesianxu.github.io/

**知乎专栏**: [计算机视觉/计算机图形理论与应用](https://zhuanlan.zhihu.com/c_1265262560611299328)

**微信公众号**：机器学习杂货铺3号店

![qrcode][qrcode]



-------



笔者曾在博文 [3] 中对MetaLM [4] 进行过介绍，而本文待要介绍的Kosmos [1,5]系列工作则是在MetaLM的设计思路下进行的进一步优化，具体来说就是继承了MetaLM中采用因果语言模型作为通用任务接口，采用各种子任务编码器[^1]对各类型输入数据进行编码的范式。在Kosmos系列中，范式保持了一致，模型也无特别变化，主要升级点在数据方面，Kosmos 1.0在MetaLM纯文本数据和图文对数据的基础上，引入了图文交织数据（image-text interleave data），使之具有了更强的in-context learning能力；而Kosmos 2.0则在Kosmos 1.0的基础上引入了图文基准数据（image-text grounding data），使之具有了图文基准（grounding）、图文指代（referring）的能力，我们下文就依次进行讨论。

图文交织数据，指的是多张相关的图片穿插在文本中，笔者在博文 [6] 中曾经讨论过Flamingo模型对于交织数据的使用，感兴趣的读者可移步阅读。为何在Kosmos 1.0中要引入图文交织数据呢？故事回到了该论文的标题，“Language is not all you need: **Aligning perception** with language models.”，想必大家都关注到了加粗的那两个词，"对齐感知"，而这是当前LLM中很火的一个话题。图文交织数据比起图文对数据，前者的上下文信息更为充足，能从多个角度对穿插在文中的图片进行多角度解释，也即是“对齐感知”的程度更加彻底，反观后者，图文对数据多是对互联网中图片的alt-text文本等进行处理后收集得到的，亦或是在搜索引擎中对用户的行为分析后收集得到，具有很大的噪声，光用图文对数据训练的模型，对齐能力因此也会受到极大的限制。图文交织数据有如此大的好处，那么我们要如何采集这类型数据呢？如Fig 1 (a)所示，一种可行的方法是采集用户在聊天软件中的对话，由于当前主流聊天软件已经支持非常丰富的多模态输入（如图片、语音、视频、文本等），因此通过合适的筛选可以获取非常丰富的图文交织数据。当然，从聊天软件中采集需要对聊天纪录进行爬取，对于个人或者研究机构来说，都可能面临资源和法律风险，因此适合于某些聊天软件/社区大厂的使用（比如腾讯的QQ和微信）。对于一般的研究者和机构来说，采集网页中的图文交织数据更有性价比，如Fig 1 (b)所示，通过爬取公开的网页，对DOM进行解析后可以清理出图文交织数据。显然，来自于聊天和网页的图文交织数据分布差异很大，目前公开论文工作看到的都是后者，而笔者暂时没发现以前者方式收集的，猜测是多在大厂内部使用而未公开，因此笔者暂无法对这两者的优劣特点进行分析。从笔者的猜测来看，基于聊天的图文交织数据会更加的口语化，因此用于训练chat bot等基于多轮对话的应用模型来说，是一个更好的选择，而基于网页的内容则可能更加权威，适合做一些世界模型的探索。

![fig1_interleave_data][fig1_interleave_data]

<div align="center">
    <b>
        Fig 1. 来自于聊天的图文交织数据和来自于网页的图文交织数据示例。
    </b>
</div>
在kosmos 1.0中，作者在原始20亿级别的网页快照中筛选了7100万网页（英文网页）[^2]，然后从挑选出的网页中提取出文字和图片，对于每个网页会将提取图片的数量限制在5张以内以减少噪声和冗余，同时随机舍弃掉了一半只含有一张图片的网页，最终将这些提取出来的文字和图片构建成图文交织数据。最终kosmos 1.0将这些图文交织数据和纯文本数据、图片文本对数据一起用于训练，如Fig 2所示，从公开数据看都是采用的英文语料进行训练的。

![fig2_data_table][fig2_data_table]

<div align="center">
    <b>
        Fig 2. Flamingo 1.0所采用的的数据类型。
    </b>
</div>

在经过这些数据进行预训练后，作者将kosmos 1.0在很多语言任务、跨模态迁移、IQ测试、感知——语言任务、视觉任务等上进行了zero-shot/few-shot测试，这些指标大多都达到了SOTA水平，具体的实验结果笔者就不在博客里面陈列了，笔者注意到的是在试验部分展现得到几个点。第一，作者将kosmos 1在OCR-free 语言理解任务中进行了测试，所谓的**OCR-free**指的是不另外对图片中的文本信息进行OCR提取后处理，而是直接将原始图片输入到模型中端到端计算。作者在 Rendered SST-2和HatefulMemes中进行了测试，如Fig 3所示，可以看到Flamingo 1.0在OCR-free文本理解任务上有一定的优势，这意味着Flamingo 1.0的视觉模型能一定程度上感知到图片中OCR语义。笔者之前在训练CLIP的时候已经发现了CLIP模型具有一定的OCR-free能力，从Flamingo 1.0的实验结果来看将视觉模型和语言模型结合后，同样能继承这种能力，这是否意味着以后文本输入甚至也可以作为视觉输入的一种进行统一化呢？毕竟人类对于文本还是图片的感知都由眼睛作为感受器接受信息，本质上都是视觉信息，这值得我们思考。

![fig3_ocr_free][fig3_ocr_free]

<div align="center">
    <b>
        Fig 3. OCR-free文本理解任务。
    </b>
</div>

作者在将kosmos 1.0用在评估多模态任务时候，采用了多模态思维链（Multimodal Chain of Thought, MCoT）技术，如Fig 4所示，通过对图文输入进行CoT提示后，能提高一些问答的准确性。笔者理解其本质是通过对图片的视觉信息进行CoT提示后，能获取很多额外和图片相关信息，这些信息可能并不完全是对图片的直接视觉信息描述，而是会包含一些和图片的底蕴、历史背景、社会背景等相关的信息 [7]，笔者称之为**延伸语义**。如Fig 4的例子所示，通过CoT提示词让模型介绍图片细节，可以知道这幅图片是来自于电影WALL-E中，这个信息可能来自于大量的图文预训练数据中，而从大量的文本预训练数据中，模型又可以知道WALL-E这部电影由皮克斯动画工作室出品，从而最终能得到正确答案。因此在多模态思维链技术中，笔者认为是结合了图文、文本预训练数据的综合优势的，而这优势对于模型性能而言可能并不是线性提升的，而是指数级提升的。如Fig 5所示，采用了多模态思维链技术后，在Render SST-2任务上有5.8%的可观提升。

![fig4_mcot_prompt][fig4_mcot_prompt]

<div align="center">
    <b>
        Fig 4. 标准prompt技术和多模态思维链prompt技术对比。
    </b>
</div>
![fig5_mcot_result][fig5_mcot_result]

<div align="center">
    <b>
        Fig 5. 采用了多模态思维链技术后，在Render SST-2任务上有5.8%的贡献。
    </b>
</div>

当然，多模态大语言模型首先是一个语言模型，因此评估MLLM的纯语言任务能力也是一个值得探索的事情，这能让我们观察LLM在引入多模态能力的过程中是否会灾难性遗忘掉其语言建模的能力。如Fig 6所示，作者对比了kosmos-1和LLM结果的对比，可以发现大多数任务上的结果和LLM持平（平均值偏低些），而有些任务上甚至还有优势，这说明引入多模态信息不会影响LLM对于文本能力的建模。

![fig6_language_task][fig6_language_task]

<div align="center">
    <b>
        Fig 6. kosmos-1模型与LLM模型在纯语言任务上的对比。
    </b>
</div>

同时，作者尝试引入了纯文本的指令微调，如Fig 7所示，即便只是采用了纯文本的指令微调，在大部分数据集上都能带来客观的性能收益，这似乎证实了指令微调数据的重要作用，能有效提高模型指令跟随（Instruction-following）的能力，在后续的一些工作，如InstructBLIP [8], LLaVa[9]中尝试引入多模态的指令微调数据，而这又是后话了。	

![fig7_instruction_tuning][fig7_instruction_tuning]

<div align="center">
    <b>
        Fig 7. 从实验中能看到指令微调在Flickr30k、VQAv2、VizWiz等数据集下的收益。
    </b>
</div>

如Fig 8所示，其实kosmos-1在论文中是作为一种通用模型接口框架下的多模态大模型进行叙述的，如之前在MetaLM [3]中的解释，通过一个因果语言模型去承接来自不同模态的输入，因此理论上kosmos-1的输入除了文本和图片外，还可以是视频、音频等，然而本作中并没有进一步试验。同时，笔者似乎也没在原文中看到有关于对交织图文数据有效性的消融试验，但是笔者还是愿意相信该数据带来的收益，特别是in-context任务上的收益，未来如何更好的收集图文交织数据也是值得关注的点。

![fig8_kosmos_framework][fig8_kosmos_framework]

<div align="center">
    <b>
        Fig 8. Kosmos-1是一个通用接口框架下的多模态大模型。
    </b>
</div>







# Reference

[1]. Huang, Shaohan, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv et al. "Language is not all you need: Aligning perception with language models." *arXiv preprint arXiv:2302.14045* (2023). short for Kosmos 1

[2]. https://blog.csdn.net/LoseInVain/article/details/130500648，增强型语言模型——走向通用智能的道路？！？

[3]. https://blog.csdn.net/LoseInVain/article/details/136161262， 《【论文极速读】MetaLM：一种融合因果语言模型和非因果语言模型的方法》

[4]. Hao, Yaru, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, and Furu Wei. “Language models are general-purpose interfaces.” arXiv preprint arXiv:2206.06336 (2022). aka MetaLM

[5]. Peng, Z., Wang, W., Dong, L., Hao, Y., Huang, S., Ma, S., & Wei, F. (2023). Kosmos-2: Grounding Multimodal Large Language Models to the World. *arXiv preprint arXiv:2306.14824*. aka Kosmos 2

[6]. https://blog.csdn.net/LoseInVain/article/details/136072993, 《【论文极速读】Flamingo：一种交织图文的视觉语言大模型方法》

[7]. https://fesianxu.github.io/2023/03/04/story-of-multimodal-models-20230304/， 《视频与图片检索中的多模态语义匹配模型：原理、启示、应用与展望》

[8]. Liu, Haotian, Chunyuan Li, Yuheng Li, and Yong Jae Lee. "Improved baselines with visual instruction tuning." *arXiv preprint arXiv:2310.03744* (2023). aka InstructBLIP 

[9]. Liu, Haotian, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. "Visual instruction tuning." *Advances in neural information processing systems* 36 (2024). aka llava



[^1]: 此处的子任务编码器，由于在MetaLM中主要是采用基于MLM训练的Transformer模型进行文本和图片编码，因此也被称之为非因果语言模型。

[^2]: 筛选规则包括过滤掉所有非英文网页，将无图片的网页去除，将网页中所有分辨率小于$64 \times 64$的图片去除，将所有单色图片去除等，同时也会将无意义的网页去除，比如垃圾邮件等。



[qrcode]: ./imgs/qrcode.png

[fig1_interleave_data]: ./imgs/fig1_interleave_data.png

[fig2_data_table]: ./imgs/fig2_data_table.png
[fig3_ocr_free]: ./imgs/fig3_ocr_free.png
[fig4_mcot_prompt]: ./imgs/fig4_mcot_prompt.png

[fig5_mcot_result]: ./imgs/fig5_mcot_result.png
[fig6_language_task]: ./imgs/fig6_language_task.png
[fig7_instruction_tuning]: ./imgs/fig7_instruction_tuning.png
[fig8_kosmos_framework]: ./imgs/fig8_kosmos_framework.png

