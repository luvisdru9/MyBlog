---
title: Qwen2.5技术报告阅读笔记
author: luvisdru9
#date: 2025-08-20 00:00:00
#updated: 2025-08-20 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: Qwen2.5技术报告阅读笔记
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

## Introduction

<div align=center>
	<img src="img.png"/>
</div>

在 AGI 与 开源 LLM 快速发展的时代，Qwen Team推出了Qwen的2.5版本，具备**Better in Size**、**Better in Data**和**Better in Use**三个特性。旗舰模型 Qwen2.5-72B-Instruct 的性能可与最先进的开放权重模型 Llama-3-405B-Instruct 相匹敌，而发布的专有的MoE模型——Qwen2.5-Turbo 和 Qwen2.5-Plus，也分别具备与 GPT-4o-mini 和 GPT-4o 竞争的性能。

<br>
<br>
<br>

## Architecture

Qwen2.5与Qwen2类似，可分为dense model和MoE model，其中dense model包含``Qwen2.5-0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B``，用于社区开源；MoE模型包含``Qwen2.5-Turbo``和``Qwen2.5-Plus``，主要用于API服务。


### Dense Model

qwen2.5 dense model沿用了与Qwen2 dense model一致的模型架构：
- 使用transformer decoder结构；
- 带有QKV bias的GQA；
- SwiGLU激活函数；
- RoPE进行位置编码
- RMSNorm + Pre-norm进行Normalization

dense model开源系列模型参数如下：

<div align=center>
	<img src="img_1.png"/>
</div>

<br>

### MoE Model

在dense model的基础上，通过将原始的FFN替换为MoE 层，将其扩展为MoE model。

每个 MoE 层由多个 FFN Expert和一个路由机制组成，该路由机制将 token 分派给 Top-k 个最相关的Expert（top-K routing）。参照Qwen2 MoE的做法，Qwen2.5 MoE也使用了shared experts（共享专家）和 routing-specific experts（路由专用专家）的混合机制，shared experts在各种任务中进行共享，routing-specific experts用于特定任务场景的选择。

<br>
<br>
<br>

## Tokenizer

Qwen2.5沿用了Qwen的tokenizer，基于开源快速分词库tiktoken进行分词，其底层为BBPE。

为了提升一致性并减少潜在兼容性问题，所有的Qwen2.5使用同一个词表。与Qwen2一致，Qwen2.5词表也包含151643个normal token，但是special token数量**从原来的3个增加至22个**，其中新增2个用于工具功能，其余用于其他模型能力。

<br>
<br>
<br>

## Pre-training

### Pre-training Data

Qwen2.5的预训练数据相较于Qwen2的有显著的优化提升，主要体现在以下几点：

#### 更好的数据过滤

使用 Qwen2-Instruct 作为“数据过滤器”，由于 Qwen2-Instruct 在更大的多语言语料库上训练过，具备更细致的质量判断能力，因此，它可以更好地保留高质量数据，并过滤掉多语言中低质量数据。相比 Qwen2，Qwen2.5所使用的数据过滤器功能更为强大。

#### 更好的math和code相关数据

在 Qwen2.5 的预训练中，加入了来自 **Qwen2.5-Math** 和 **Qwen2.5-Coder** 的训练数据，使得Qwen2.5在数学和编程任务上达到SOTA水平。

#### 更好的合成数据

为了生成高质量的数学、代码和知识类合成数据，分别使用**Qwen2-72B-Instruct**和**Qwen2-Math-72B-Instruct**生成相应的数据，将生成的数据再经过**通用 reward model**和 **Qwen2-Math-RM-72B**进行进一步评判过滤，保证合成数据的质量。

#### 更好的数据混合

Qwen Team使用Qwen2-Instruct 分类数据，发现各个领域的数据存在不平衡现象：

- 电商、社交媒体、娱乐类内容占比过高，这些领域包含大量重复、模板化、低质量甚至机器生成内容；
- 科技、科学、学术研究等高价值领域反而稀少。

于是，Qwen2.5在预训练数据中对这些占比过高的领域进行下采样，对高价值的领域进行上采样，最终形成更加平衡且价值更高的预训练集。

最终形成了比Qwen2更大、质量更优的预训练数据集，从Qwen2的7T tokens扩展至18T tokens。

<br>

### 超参数的 Scaling Law

[Kaplan等人](https://arxiv.org/abs/2001.08361)提出的Scaling Law最主要的一个作用便是预测模型性能，指导算力资源配置。而Qwen Team基于Qwen2.5的预训练数据开发了一种**关于超参数的Scaling Law**，并将其用于寻找不同规模下的dense model和MoE model的最优训练超参数，例如batch size、学习率$\mu$等。

于是，Qwen Team系统性地研究了如下两个超参数：**最优学习率**$\mu_{opt}$和**最优batch size**$B_{opt}$，看它们是如何随着模型规模$N$和数据规模$D$变化的。

相关实验设置：
- 44M ~ 14B规模参数的 dense 模型；
- 44M ~ 1B 规模激活参数的 MoE 模型；
- 数据规模从 0.8B 到 600B tokens。

基于实验得到的这些最优超参，对最终的 loss 随模型规模与训练数据规模的变化规律进行建模，得到超参数的Scaling Law，并利用Scaling Law：

1. 预测不同 MoE 模型与 dense 模型的性能对应关系；
2. 帮助确定 MoE 的激活参数量和总参数量。

从而确保例如 Qwen2.5-72B 和 Qwen2.5-14B 的 dense 版本与其对应 MoE 版本性能相当。


<br>

### Long-context Pre-training

为了训练效率最优，Qwen2.5采用两阶段方法进行训练：

1. 初始阶段，使用4096的上下文长度；
2. 后续扩展阶段，使用更长的上下文长度。

根据上述规则，规定除了Qwen2.5-Turbo 外的所有Qwen2.5模型的训练上下文长度由4096 tokens变为32768 tokens，将同时将 RoPE 基频使用 ABF 技术从10000使用1000000。这里稍微解释一下RoPE的ABF：


#### 基于ABF的RoPE变频

RoPE的角速度计算公式为$ω(d)=\theta^{-\frac{2d}{D}}$，其中$\theta$为基频，$d$代表总维度为$D$的向量的第$d$个维度。

RoPE的$w$越大，旋转地越快，周期重叠速度越快，可区分\理解的上下文长度就越短，反之亦然。逐渐增大RoPE的基频会使得其旋转速度变慢，对长上下文的理解能力也更强。

因此，我们需要进行ABF，即自适应基频，在扩展阶段的训练过程中，基频 $\theta$ 不再是一个常数，而是一个随着训练步数（或者上下文长度）逐渐增大的变量。


<br>

回到正文，对于Qwen2.5-Turbo而言，采取了与其他模型不同的预训练策略，上下文长度扩展分为四个阶段：``32768 tokens->65536 tokens->131072 tokens->262144 tokens``，RoPE基频高达10000000，每个阶段训练数据包含40%当前最大长度上下文数据和60%较短长度上下文数据，平滑过渡，提升模型对不同上下文长度的泛化能力。

此外，类似Qwen2，也使用了YaRN、DCA等策略继续稳定模型长上下文的表现。

综合这些技术后，不仅降低了长上下文的perplexity，还不影响短上下文的性能。Qwen2.5在长上下文的表现得到了有效优化，Qwen2.5-Turbo 支持 100w tokens，其他模型支持 131072 tokens。


<br>
<br>
<br>

## Post-training

Qwen2.5的post-training相比于Qwen2主要有两点提升：

- SFT的数据覆盖更大；
- 双阶段RL（offline + online）。

### SFT

Qwen2.5在SFT环节做的增强点还是很充足的，分为以下九点：

#### 高质量长序列生成

为了构建长文本数据集，Qwen2.5通过**回译增强技术**生成长文本的queries，并对生成文本的长度进行约束，生成**{query, long-text data}**数据对，并通过Qwen2过滤掉低质量的数据对。

#### Math

加入来自 Qwen2.5-Math 的CoT数据，数据来自公开数据、K12 数学、合成题。为了确保高质量的数学推理，使用基于奖励模型的拒绝采样筛选高质量数据，并提供的正确推理步骤作为参考，指导模型学习“正确的思考方式”。

#### Code

为了提升模型代码相关能力，整合了Qwen2.5-Coder 的指令微调数据，利用多编程语言的 agent 协作生成 40 种不同编程语言相关的高质量对话。此外，从包含代码相关的Q&A网站和带有算法/代码片段的GitHub仓库收集数据，增加了数据的多样性。

使用多语言的沙箱执行环境进行静态检查和自动单元测试，以确保代码正确性和质量。

#### 指令跟随

为了确保SFT包含高质量的指令跟随数据，Qwen2.5使用一种严格的“代码式验证框架”进行指令跟随数据生成。具体步骤如下：

- 数据生成所用的LLM需要生成三样东西：1、生成指令，2、生成验证程序代码，3、生成若干单元测试用例用于交叉验证；
- 而后给Qwen2.5指令，得到其生成的代码后，执行验证程序和完整的单元测试；
- 如果测试通过，保留为SFT数据，反之丢弃。

这种基于**执行反馈**的拒绝采样能够严格筛选 SFT 数据，从而保证模型能够忠实遵循意图指令。

#### 结构化数据理解

对于结构化或者半结构化的数据理解在大模型的企业应用中非常重要，Qwen Team开发了一个包含许多这类数据理解任务（数据表分析、表格问答、JSON 编辑、半结构化网页信息读取）的数据集，并通过引入CoT，更大幅度地在SFT这个环节强化了模型对于这类数据的深入理解。

#### 逻辑推理

为增强模型逻辑推理能力，构建了包含7w条跨领域查询的多样化数据集（涵盖选择题、判断题、开放式问题），模型使用多种方法对这些问题进行推理（演绎/归纳/类比/因果/统计推理），推理后将迭代地进行数据筛选，剔除错误答案与缺陷推理，显著提升了模型在复杂推理任务中的准确性与鲁棒性。

#### 跨语言迁移

高资源语言指如英文、中文这类能够收集到大量数据的语言，低资源则反之。为了促进模型的通用能力在不同语言之间迁移，使用一个翻译模型，将高资源语言的指令翻译成多种低资源语言，并由此生成对应的回答。

#### System Prompt 鲁棒性

当用户拿到一个模型进行使用时，总会自己自定义各类的System Prompt，如果模型对于System Prompt过于敏感导致性能方差极大，会十分影响用户的体验。

Qwen Team构建了数百个通用的System Prompt模板，增加后训练中System Prompt的多样性，并且确保System Prompt与对话内容的语义上的一致性。

后续评估表明模型在不同的System Prompt下仍能保持优异的表现和较小的方差，增强了模型的鲁棒性。

#### 回答过滤

使用多种自动化注释方法进行模型回答评估，包括专门的评论模型和多智能体协作评分系统。所有回答都经过这些系统的严格筛选，只有被所有评分系统认为完美的回答才会被保留，从而保证了模型输出的高质量标准。

<br>

最终，Qwen2.5的SFT数据集包含了超过100w条高质量数据，基于32K tokens的上下文长度进行两轮SFT，学习率由$7 \times 10^{-6}$变为$7 \times 10^{-7}$，使用0.1的weight decay，梯度范数被限制在最大值为1.0的范围内。

<br>

### Offline RL

在SFT阶段，广泛使用了**执行反馈**和**答案匹配**两种思想确保模型回答的质量，在offline rl中继续复用这个步骤，将通过质量筛选的数据设为正样本，反之设为负样本，而后构建数据对直接进行DPO训练。

为了进一步提高数据的可靠性和安全性，采用了人工与模型自动化相结合的审查流程，这种双重审查方法确保训练数据既可学习，又更符合人类的预期。

最后得到了150000 对训练样本，使用 OMO 进行 1 个 epoch 的训练，学习率为 $7 \times 10^{-7}$。

<br>

### Online RL

online rl中奖励模型十分重要，为了构建一个稳健的奖励模型，Qwen2.5的online rl遵循一套精心定义的标注标准，这些标准确保模型生成的回答不仅质量高，而且符合伦理道德约束与用户中心原则。分别包括以下几点：

- 真实性（Truthfulness）：回答必须基于事实，坚定地回答给定的上下文和指令，不得生成虚假或缺乏依据的信息；

- 有用性（Helpfulness）：回答应真正解决用户问题，内容积极、有吸引力、具有教育意义并与主题相关，同时严格遵循指令；

- 简洁性（Conciseness）：回答应简明扼要，避免不必要的冗长；

- 相关性（Relevance）：回答的所有部分都必须与用户查询、对话历史和助手上下文直接相关；

- 无害性（Harmlessness）：必须避免可能导致非法、不道德或有害行为的内容，始终遵循负责任的沟通方式；

- 去偏（Debiasing）：回答应避免偏见，包括但不限于性别、种族、国籍和政治方面，遵循广泛认可的伦理道德标准。


用来训练奖励模型的queries来自两个数据集，一个是公开可得的开源数据，另一个为更复杂的私有数据集。 相关的回答由 Qwen 在不同训练阶段通过不同方法（SFT、DPO、RL）微调后的模型checkpoint生成。为了增加回答的多样性，这些回答在不同的temperature下得到。 而后，偏好数据对通过人工和自动标注流程生成，同时 DPO 的训练数据也被整合进该数据集中。


在online rl中使用 GRPO 进行训练，其中用于训练奖励模型的queries集合与 GRPO 训练阶段使用的queries集合完全一致。训练过程中query的处理顺序根据其所有回答的分数（由奖励模型评估）的方差决定，方差越大的query被优先处理，以提高训练效果。 为每个query采样8个回答，一个episode采样2048个sample，一个更新的global batch size为2048，将每对query和回答视为一个样本。

<br>

### LongContext Fine-tuning

为了进一步扩展 Qwen2.5-Turbo 的上下文长度，在其后训练阶段引入了更长上下文的 SFT 示例数据，使Trubo模型能够在处理长上下文查询时更好地符合人类偏好，这里采用两阶段的SFT方案：

1. 模型仅使用短指令进行微调，每条指令最多包含 32768 个 token。这一阶段使用的数据和训练步骤与其他 Qwen2.5 模型相同，以保证在短任务上的强性能；

2. 微调过程结合短指令（最多 32768 token）和长指令（最多 262144 token）。这种混合方式有效提升了模型在长上下文任务中的指令跟随能力，同时保持了短任务的性能。

在 RL 阶段，采用与其他 Qwen2.5 模型类似的训练策略，仅关注短指令。这一方案的选择主要有两个原因：

1. 对长上下文任务进行 RL 训练计算成本十分高昂；

2. 当前缺少能够为长上下文任务提供合适奖励信号的奖励模型。

此外，Qwen Team发现，**仅对短指令进行 RL 训练，也可以显著提升模型在长上下文任务中对人类偏好的对齐能力。**个人分析，原因可能有如下：

- rl强化了模型的instruct-follow能力，这对长短上下文都起作用；
- rl使模型学习到了“偏好”这种抽象的模式，使其能够关注有用的信息，使得模型的思考方式更为贴近人类了；
- ......

<br>
<br>
<br>

## Evaluation

具体的评估实验可参照原文自行查看，此处不展开说明。

回看Qwen2.5的成功，还是感叹一句其数据构建方面的严谨与全面，果然究其根本数据才是LLM良好性能的基石......