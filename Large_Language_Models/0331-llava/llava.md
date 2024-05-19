<div align="center">
    【论文极速读】 LLava: 指令跟随的多模态大语言模型
</div>


<div align="right">
    FesianXu 20240331 at Tencent WeChat Search Team
</div>

# 前言

如何将已预训练好的大规模语言模型（LLM）和多模态模型（如CLIP）进行融合，形成一个多模态大语言模型（MLLM）是目前很火热的研究课题。本文将要介绍的LLava是一个经典的工作，其采用了指令微调的方式对MLLM进行训练，笔者在此笔记，希望对诸位读者有所帮助。**如有谬误请见谅并联系指出，本文遵守[CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/)版权协议，转载请联系作者并注明出处，谢谢**。

$\nabla$ 联系方式：

e-mail: FesianXu@gmail.com

github: https://github.com/FesianXu

github page: https://fesianxu.github.io/

知乎专栏: 计算机视觉/计算机图形理论与应用

微信公众号：机器学习杂货铺3号店

----

之前笔者在博文中曾经介绍过不少MLLM的工作 [2-4]，这些工作在模型结构和数据采集、利用上都有所创新。本文将会介绍LLava (Large Language and Vision Assistant) [1]，一个视觉指令微调的工作，在这篇工作中收集了一个大规模的指令微调数据集`llava-instruct-150k`，可以供给下游的MLLM任务进行指令微调。这个工作为我们采集数据的实践提供了有价值的指导，让我们看看他是如何做的。

LLava采集多模态指令微调数据的思路很直接：先将图片的视觉信息转化为文本描述，然后通过ChatGPT/GPT4的强大文本理解能力，去构建指令微调数据。由于视觉信息和文本信息之间存在信息鸿沟，为了尽可能减少信息差，如Fig 1所示，作者不仅采用image caption对图片进行描述， 同时采用object detection模型对图中的物体进行检测和定位，丰富的文本信息有利于尽可能全面地描述图片的视觉上下文信息，为后续GPT4通过文字去理解图片的视觉信息提供了重要基础。

![fig_1_image_context_to_text][fig_1_image_context_to_text]

<div align="center">
    <b>
        Fig 1. 两种不同的文本化的图片上下文信息，分别采用image caption和object detection模型进行处理。
    </b>
</div>

考虑到MLLM的下游应用可能有多种多样，比如聊天机器人，信息抽取器等多种场景，因此在指令微调数据的构建上也需要尽可能的多样化。如Fig 2所示，作者在构建指令微调数据的时候考虑了三种可能的类型，对话、细节描述和复杂推理，其中对话属于多轮交互，而其他则是单轮交互。引入对话形式的指令微调数据，有利于后续将MLLM应用到聊天机器人应用中，同时也为MLLM提供了多轮对话的能力。细节描述的问题，可以采样自固定的问题集合，如下所示，但是围绕图片展开讨论的对话显然不可能存在固定的问题集合，作者于是采用GPT4去围绕图片信息，同时产生问题和回答。

> • "Describe the following image in detail"
> • "Provide a detailed description of the given image"
> • "Give an elaborate explanation of the image you see"
> • "Share a comprehensive rundown of the presented image"
> • "Offer a thorough analysis of the image"  
>
> ...

为了更好地让GPT4产生的问题更符合指令微调的需求，除了在prompt中对任务进行清楚地定义外，如下prompt所示，作者还提供了例子（需要人工设计提供）去辅助GPT4生成问题和回答，即是采用了in-context learning的方式。最终，作者采集到了158k个图文指令微调数据，其中包括58k个对话，23k个细节描述和77k个复杂推理。

> 你是一个人工智能视觉助理，你在查看一张图片。你将看到五句话，用于描述你正在看到的同一幅图像。在你看到图像时回答所有问题。
>
> 设计一个你和一个询问这张照片的人之间的对话。答案应该是一个视觉人工智能助理看到图像并回答问题的语气。提出不同的问题并给出相应的答案。包括询问图像视觉内容的问题，包括对象类型、对象计数、对象动作、对象位置、对象之间的相对位置等。请提出仅包括有明确答案的问题：
>
> （1） 人们可以看到问题所问的图像中的内容，并且可以自信地回答；
>
> （2） 可以根据图像自信地确定它不在图像中。不要问任何没把握回答的问题。
>
> 你还可以提出包括与图像中的内容相关的复杂问题，例如，询问图像中对象的背景知识、询问讨论图像中发生的事件等。同样，不要询问不确定的细节。在回答复杂问题时提供详细答案。例如，给出详细的例子或推理步骤，使内容更有说服力和条理。如有必要，可以包括多个段落。

![fig_2_response_types][fig_2_response_types]

<div align="center">
    <b>
        Fig 2. 三种不同类型的指令微调数据类型，对话，细节描述和复杂推理，其中对话是多轮交互，而其他是单轮。
    </b>
</div>

LLava的模型建模，如Fig 3所示，采用了类似于Frozen [5] 的visual prompt的方式，将图片进行视觉特征提取后，通过投影矩阵$\mathbf{W}$将其映射到LLM同维度的特征空间之中，即是$\mathbf{H_v} = \mathbf{W} f_v(\mathbf{X}_v)$，其中$f_v(\cdot)$是视觉提取模型，比如CLIP，而$\mathbf{X_v}$是输入的图片。最后将视觉侧的特征$\mathbf{H_v}$和文本侧的特征$\mathbf{H}_q$拼接在一起，即可送给LLM。在训练过程中，对于多轮对话的数据$(\mathbf{X}^{1}_{q}, \mathbf{X}^{1}_{a},\cdots,\mathbf{X}^{T}_{q}, \mathbf{X}^T_{a})$，其中$T$是对话的轮次，将所有的回答$\mathbf{X}_a^{t}$都视为LLM的待预测内容，那么第$t$轮的LLM的指令输入$\mathbf{X}_{instruct}^{t}$则是：
$$
\mathbf{X}^{t}_{instruct} = 
\begin{cases} 
\mathrm{Rand\ choose\ } [\mathbf{X}^1_q, \mathbf{X}_v] \ or \  [\mathbf{X}_v, \mathbf{X}^1_q] & t=1 \\
\mathbf{X}^{t}_q & t \gt 1
\end{cases}
\tag{1}
$$
在第一个轮次$t=1$的时候，会随机选择图片$\mathbf{X}_v$前置或者图片后置，这样有利于增加数据的多样性。如Fig 4所示，模型的输入包含有一个系统提示词(system prompt)，在本文是`X_{system message} = A chat between a curious human and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the human’s questions.   `，`<STOP> = ###`，注意到只有绿色字样部分的才会进行损失计算，不难发现都是`<STOP>`部分和$\mathbf{X}_{a}^{t}$部分。整体损失就是LLM的自回归损失，如公式(2)所示：
$$
p(\mathbf{X}_a|\mathbf{X}_v, \mathbf{X}_{instruct}) = \sum_{i=1}^{L} p_{\theta}(x_i|\mathbf{X}_v, \mathbf{X}_{instruct,<i}, \mathbf{X}_{a, <i})
\tag{2}
$$
注意到，在第$i$个令牌（token）之前的所有令牌（包括指令和回答部分）都会作为输入，去预测第$i$个令牌。

![fig_3_visual_prompt][fig_3_visual_prompt]

<div align="center">
    <b>
        Fig 3. 采用visual prompt的形式引入多模态向量。
    </b>
</div>



![fig_4_multiturn_input][fig_4_multiturn_input]

<div align="center">
    <b>
        Fig 4. 模型的输入示例，只有绿色字样部分才会进行损失计算。
    </b>
</div>

在训练范式上，由于引入了投影矩阵$\mathbf{W}$去对齐视觉特征和LLM文本特征，因此作者设计成两阶段训练，在第一阶段引入预训练的方式，除了投影矩阵外其余所有参数都固定住，其数据采用的是CC3M中过滤出来的595k个图文对数据，采用如下所示最简单的提示词输入

> X_q, X_v <STOP> \n Assistant: X_a <STOP> \n

其中的$\mathbf{X}_a$直接采用图文对中的文本部分，即是图片的caption，而$\mathbf{X}_q$则随机采样自以下几个问题。

> • "Describe the image concisely."
> • "Provide a brief description of the given image."
> • "Offer a succinct explanation of the picture presented."
> • "Summarize the visual content of the image."
> • "Give a short and clear explanation of the subsequent image."
> • "Share a concise interpretation of the image provided."
> • "Present a compact description of the photo’s key features."
> • "Relay a brief, clear account of the picture shown."
> • "Render a clear and concise summary of the photo."
> • "Write a terse but informative summary of the picture."
> • "Create a compact narrative representing the image presented."  



在第二阶段的训练中，作者只对视觉编码器的参数就行固定，而LLM和投影矩阵的参数都进行端到端的训练，训练数据就来在于之前采集的158k个指令微调数据。







# Reference

[1]. Liu, Haotian, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. "Visual instruction tuning." *Advances in neural information processing systems* 36 (2024). aka llava

[2]. https://blog.csdn.net/LoseInVain/article/details/136428429, 《Kosmos-1: 通用接口架构下的多模态大语言模型》

[3]. https://blog.csdn.net/LoseInVain/article/details/136072993， 《【论文极速读】Flamingo：一种交织图文的视觉语言大模型方法》

[4]. https://blog.csdn.net/LoseInVain/article/details/136013909，《BLIP2——采用Q-Former融合视觉语义与LLM能力的方法》

[5]. Tsimpoukelli, Maria, Jacob L. Menick, Serkan Cabi, S. M. Eslami, Oriol Vinyals, and Felix Hill. "Multimodal few-shot learning with frozen language models." Advances in Neural Information Processing Systems 34 (2021): 200-212. aka Frozen




[fig_1_image_context_to_text]: ./imgs/fig_1_image_context_to_text.png
[fig_2_response_types]: ./imgs/fig_2_response_types.png
[fig_3_visual_prompt]: ./imgs/fig_3_visual_prompt.png
[fig_4_multiturn_input]: ./imgs/fig_4_multiturn_input.png

