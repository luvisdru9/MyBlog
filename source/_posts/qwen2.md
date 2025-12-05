---
title: Qwen2技术报告阅读笔记
author: luvisdru9
#date: 2025-07-09 00:00:00
#updated: 2025-07-09 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: Qwen2技术报告阅读笔记
keywords:
  - NLP
  - LLM
  - DL
  - ML
#top_img:
#comments:
cover: img_2.png
#toc:
#toc_number:
#toc_style_simple:
#copyright:
#copyright_author:
#copyright_author_href:
#copyright_url:
#copyright_info:
#mathjax:
#katex:
#aplayer:
#highlight_shrink:
#aside:
#abcjs:
#noticeOutdate:
---

## Abstract

继Qwen后，阿里又推出了Qwen2，同样是基于**Transformer架构 + Next-token Prediction自回归训练范式**。Qwen2提供了Base和Instruct两个版本，总共发布了5个版本的模型，包括四个**dense model**（0.5B、1.5B、7B和72B）和一个57B的**MoE**模型（每个token的计算只激活14B的参数）。

Qwen2使用超过**7万亿tokens**的训练数据，其中包含了更高质量的代码和数学推理数据，采用**SFT + DPO**进行post-training，在多个benchmark上取得了优异的性能表现。

<br>
<br>
<br>

## Tokenizer & Model

### Tokenizer

Qwen2沿用了Qwen1的tokenizer方案，基于开源快速分词库tiktoken进行分词，其底层为BBPE。该方案具有很高的tokenization效率，且相较于其他方案而言具有更高的压缩率，除了上述优势以外，采用这种方案也能够使得Qwen2的多语言能力得到大幅提升。

Qwen2所有尺寸的模型都使用相同的词表，包括151643个normal token以及3个special token（也称为control token），参照Qwen1的技术报告，这里的special token是因为使用了OpenAI的**ChatML**数据格式，分别是```<|endoftext|>```, ```<|im_start|>```, ```<|im_end|>```。

需要注意的是，文中提到为了考虑到**分布式训练**的需求，实际用于embedding的词表大小会更大一些。

<br>

### Model Architecture

Qwen2是经典的基于Transformer架构且带有Causal Mask的LLM，上述提到了，Qwen2包含了四个dense model和一个57B的MoE模型，下面进行简要介绍：

#### Dense Model

Qwen2的dense model由多个Transformer层组成，每一层都包括Causal Attention和FFN，相比于Qwen1而言，大部分架构保留了下来，主要具有以下几处变动：

- GQA：Qwen2 dense model采用Grouped Query Attention（GQA）替代传统的MHA，通过这种方法优化了推理中的 KV cache，使吞吐量显著提升；

- DCA + YARN：为了扩展上下文的长度，Qwen2使用了Dual Chunk Attention（DCA）+ Yet another RoPE extensioN method（YaRN）的组合trick。前者通过对长序列token进行chunk划分，多层次注意力捕获依赖关系，高效处理长文本；后者则重新缩放了注意力权重，解决 RoPE 在长上下文下“高频旋转失真”问题，两者组合后，使模型在极长上下文下仍保持稳定、可控、推理能力不掉点。

其他则沿用Qwen1的配置，例如SwiGLU、带有QKV bias的Attention计算、RMSNorm + Pre-norm。

<br>

#### Mixture-Of-Experts Model

Qwen2 MoE 模型的架构与 Qwen1.5-MoE-A2.7B中的高度一致。在MoE模型中，MoE FFN 取代了原始 FFN，由 n 个独立 FFN 组成，每个 FFN 为一个Expert。每个 token 会根据 gated network G 给出的概率被路由到特定Expert $E_i$：
$$\mathbf{p} = softmax(G(\mathbf{x}))$$
$$\mathbf{y} = \sum_{i \in top_k(\mathbf{p})} \mathbf{p}_i E_i(\mathbf{x})$$


下面介绍在Qwen2中关于MoE的关键设计。

##### Expert Granularity

MoE模型与Dense模型最大的区别就是MoE层包含多个FFN，每个对应一个Expert，那么可以联想到，从Dense模型迁移到MoE模型最简单直接的方式便是“使每个Expert FFN的参数 = 原有Dense模型中单个FFN的参数”。比如从Mistral-7B到Mixtral 8×7B，而后同时激活 8 个Expert中的 2 个。

但是在Qwen2中则采用一种fine-grained experts（细粒度专家）的方案，详细可见[《DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models.》](https://arxiv.org/abs/2401.06066)。大致来说，在基础的MoE架构基础上，将每个Expert的中间维度减少至其原始维度的$\frac{1}{m}$，并将每个Expert细分为m个更细粒度的Tiny Expert，最后增加同时激活Expert的数量至原来的m倍，在保持一致的专家参数数量和计算成本的同时，通过更细粒度地分割专家，使得激活的专家组合更加灵活和适应。


##### Expert Routing

路由机制的设计对MoE的性能影响巨大，有很多研究者已经在 MoE 中同时使用 shared experts（共享专家）和 routing-specific experts（路由专用专家），前者所有token都可以使用且不依赖路由选择，后者则是只有当路由选择它们时才会激活，例如下图中关于DeepSeekMoE对这两者的混合使用。

<div align=center>
	<img src="img.png"/>
</div>

在Qwen2中也采用了这种方案，shared experts在各种任务中进行共享，而routing-specific experts则用于特定任务场景的选择，使 MoE 路由机制更加灵活与高效。


##### Expert Initialization

Qwen2 MoE中的初始化类似于一种“upcycling”的方法，以dense模型权重为基础，若有n个Expert则复制n份来初始化Expert。但是由于Qwen2采用了fine-grained experts，每一个Expert FFN的中间维度会比原来dense模型中FFN的中间维度要更小，在Qwen2中是这样做的：

- 假设Expert中间层大小为$h_E$，Expert数量为$n$，原始dense模型FFN的中间维度为$h_{FFN}$；
- 将原始FFN复制$$\lceil \frac{n \times h_E}{h_{FFN}} \rceil$$次，以保证适配指定Expert数量与大小，使得总维度能够覆盖所有的Expert的需求；
- 这些复制体都是相同的，如果按照原有的FFN内在统计规律顺序切片分配给每个Expert，会存在许多Expert出现几乎一致或者完全一致的情况，为了给每个Expert引入多样性的表达能力，Qwen2在中间层维度上对参数进行shuffle，使得每个Expert组合了不同来源的参数，具有独特的参数；
- 最后，为了引入一些随机性，每个Expert 50%的参数将被随机初始化而不是依靠原始的FFN

<br>

### Model Configuration

<div align=center>
	<img src="img_1.png"/>
</div>

相比于 Qwen1.5，Qwen2显著减少了 KV Heads 大小，内存占用减小，使得长上下文推理，低显存推理，多轮 reasoning 更加高效。


<br>
<br>
<br>

## Pre-training

在Qwen2的预训练中，主要关注了**优化改善数据集**与**探索有效处理长上下文的方法**两个方面。

### Pre-training Data

Qwen2的预训练基于一个全新的大规模高质量数据集，相比于Qwen1和Qwen1.5所使用的预训练数据集有了显著地提升。主要体现在以下三个方面：


#### 数据质量提升

Qwen2的数据同样来自各类渠道，原始数据中不可避免存在低质量文本、重复内容和噪声信息等冗余信息，需要对数据进行过滤。

Qwen2结合了**启发式过滤**与**模型过滤**两种过滤方案。前者是基于经验规则、统计或简单算法的过滤手段（删除长度过短或过长的文本/删除重复行或文档/删除包含过多特殊符号或乱码的文本/删除不符合语言规范或语法的句子等），这种方案快速、简单、可控，
可保证初步的基础质量。后者则利用 Qwen 模型打分筛掉低质量数据，此时的评分维度会更加丰富（语义完整性/文本可读性/与主题相关性/代码是否可运行/数学公式是否合理等），并得到高质量预训练数据。

#### 数据内容扩展

Qwen2的预训练数据增加了代码、数学和多语言文本，提升了推理和跨语言能力：

- 收集了大量高质量的代码、数学、以及多语言文本；

- 支持约 30 种语言，包括英语、中文、西班牙语、法语、德语、阿拉伯语、俄语、韩语、日语、泰语和越南语。

#### 数据分布优化

为了确定数据配比，选择一种更符合人类实际接触和使用语言的方式，Qwen2团队选择先在一些小模型上快速反复试验不同的数据比例配方，找出最优的数据混合策略，最终将这个比例用于训练大模型。

<br>

在上述三种方案的改善优化下，Qwen2 将预训练数据规模从 Qwen1.5 的 3T 扩大到 7T tokens。Qwen2团队尝试进一步**放宽质量标准**构建 12T tokens 的更大数据集，但模型效果并未优于 7T，除了小模型自身能力的影响外，这也能说明预训练数据集的质量影响之大。因此为了在性能与成本间平衡，最终选择使用高质量的 7T 数据集，最终得到的预训练数据方案如下：

- 除了0.5B之外的 Qwen2 dense模型均使用 7T tokens 训练；

- Qwen2-0.5B 使用 12T tokens 数据训练，因为0.5B模型规模小、训练成本低；

- Qwen2-MoE在继承dense模型参数的基础上，额外增加 4.5T tokens 的训练数据，为 MoE部分提供更多学习信号；

- 在预训练中加入高质量多任务指令数据，用于增强模型的 **In-Context Learning 与 Instruct Follow** 能力。

<br>
<br>

### long-content training

在长文本推理能力的提升方面，除了之前提到过的DCA + YaRN的方案，Qwen2还将RoPE的基础频率**从 10000 调整到 1000000**。同时，在预训练时将上下文长度**从 4096 tokens 增加到 32768 tokens**，引入大量高质量长文本数据。



<br>
<br>
<br>

## Post-training

Qwen2的post-training与传统依赖大量人工监督的方式不同，聚焦于如何高效获取用于SFT与RLHF的高质量示范与偏好数据，在大幅降低人工标注需求的同时，保障数据的优质与可靠。

### Post-training Data

对于SFT而言，所使用的数据为给定的“指令-回答”对$\{ (x_i, y_i) \}$；而对于RLHF而言，所使用的数据为“指令-更偏好的回答-较低偏好的回答”对$\{ (x_i, y_i^+, y_i^-) \}$。后训练数据构建遵循一个“two-step”方案，由 **collaborative data annotation** 和
 **automated data synthesis**组成。

#### collaborative data annotation（协作式数据标注）

分为自动本体提取（Automatic Ontology Extraction）、指令选择（Instruction Selection）、指令演化（Instruction Evolution）和人工标注（Human Annotation）四个环节：

- Automatic Ontology Extraction（这里指的是描述指令类型与任务结构的体系）：这里使用了InsTag，这种方法能够给指令标注细粒度标签，把“杂乱的指令”系统化为有层级、有细粒度的任务类别（写作/代码/数学/翻译/对话/扮演等），再根据标签构建指令的层级结构（hierarchical ontology）；
- Instruction Selection：得到InsTag标注后的指令集后，根据文章[《How abilities in large language models are affected by supervised fine-tuning data composition》](https://arxiv.org/abs/2310.05492)中的思路，根据**标签多样性**、**语义丰富性、复杂度**和**意图完整性**等角度进行指令选取，最终选取了选取一组具有代表性的指令。
- Instruction Evolution：再选定代表指令集后，基于Qwen模型进行指令集的self-evolution，例如：增加边界条件、增加限制以及增加逻辑要求，对指令集进行复杂化并且保证了指令集各类难度的多样性；
- Human Annotation：这一步针对给定指令的回答进行选择，具体我们给定一条指令，选择不同版本\规模\生成策略的Qwen模型生成多条回答，人工根据人类偏好对这些回答进行排序，确保最优的回答符合标准之后，分别得到SFT和RLHF所需要格式的数据。

#### automated data synthesis（自动化数据合成）

大规模数据的合成通常成本十分高昂，对于需要专业知识、经验、严谨性或耐心的任务更是具有巨大的挑战，Qwen2团队提出了一套自动化数据合成的方法以应对上述问题：

- 拒绝采样（Rejection Sampling）：针对数学问题等具有**确定性答案**的指令任务，采用拒绝采样的思路，要求LLM为这个指令生成多条推理路径，保留得出准确结果且模型判定为合理的路径作为SFT的数据对，并通过对比正确与错误路径为RLHF生成偏好数据；
- 执行反馈（Execution Feedback）：针对 code 与 constraint-following 等指令任务，通过一个Execution Feedback来判断生成回答的质量。例如code任务中，我们可以执行生成的代码，而后检查执行结果是否符合要求；而对于constraint-following类型的任务也可以参照这种思路。比如``请生成不超过20字的文本``的指令可以直接判断生成文本的size，或者``以JSON格式列出数据``则可以使用类似``json.loads()``这种方法进行验证...
- 数据再利用（Data Repurposing）：针对角色扮演与文学写作等指令任务，充分利用互联网已有数据进行数据合成。对于文学写作任务，收集已发表的高质量作品，使用LLM自动生成指令，将原作品作为标准回答后构建数据对。而对于角色扮演类的任务，用Wikipedia抽取真实的（这里的真实是指在互联网真实存在的）人物信息，让 LLM 生成角色扮演的指令+回复；
- 合法性反馈（Constitutional Feedback）：根据现代社会的法律与道德约束构建一个“原则集”（比如安全、价值、中立性），让模型根据原则写出符合或违反原则的回答，作为SFT和RLHF的数据来源。


<br>
<br>

### SFT & RLHF

在SFT阶段，Qwen2使用了超过 50 万条高质量指令数据，涵盖指令跟随、代码、数学、推理、多语种和安全等多种能力，以 32k 的序列长度训练 2 个 epoch，学习率从 7e-6 逐步降到 7e-7，并采用 0.1 的weight decay 和 gradient clip（max=1.0）来防止过拟合并稳定训练。

在RLHF阶段，Qwen2的训练分为offline和online两个阶段，offline阶段基于预构建好的偏好数据集，通过DPO最大化$y_i^+$和$y_i^-$之间的差异；online阶段，模型实时进行响应，生成多条回答，由Reward Model选取最好和最差的回答构建为新的$y_i^+$和$y_i^-$，而后再通过DPO进行训练。在训练中OMO（Online Merging Optimizer） 减少 “alignment tax”，保证模型符合人类偏好的同时不丢失性能。

<br>
<br>
<br>

## Evaluation

实验部分可以自行查看原文，这里不进行记录。

<br>
<br>
<br>

## 参考资料

- [《QWEN2 TECHNICAL REPORT》](https://arxiv.org/pdf/2407.10671)
