---
title: Qwen技术报告阅读笔记
author: luvisdru9
date: 2025-09-13 00:00:00
updated: 2025-09-13 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: Qwen技术报告阅读笔记
keywords:
  - NLP
  - LLM
  - DL
  - ML
#top_img:
#comments:
cover: img.png
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

QWEN（千问）是阿里发布的一个全面的LLM系列，涵盖了不同参数规模的各类模型，这些模型之间的关系网如下：

<div align=center>
	<img src="img_1.png"/>
</div>

- QWEN（基础预训练模型），使用多达3万亿tokens的多样化文本和代码数据进行了大规模预训练，涵盖广泛领域；

- QWEN-CHAT系列，包含Qwen-Chat和Qwen-Chat-RLHF，基于基础预训练模型QWEN，使用SFT + RLHF的技术进行微调，数据集涵盖任务执行、对话、安全性等各个方面；

- QWEN-CODE系列，基于Qwen基础模型，在大规模代码数据集上进行二次预训练，得到代码基础模型Code-Qwen，并进一步在一些代码生成、调试与解释相关的语料上进行微调得到Code-Qwen-Chat；

- 阿里也专门设计了数学相关的LLM —— Math-Qwen-Chat，不论是7B还是14B的模型，在数学相关的benchmark上（例如GSM8K和MATH）都接近了GPT-3.5的表现；

- QWEN-VL系列，阿里开源了多模态预训练模型Qwen-VL和其微调版本Qwen-VL-Chat，在理解视觉与语言指令的多样化能力方面，于多个benchmark上超越了现有的开源视觉语言模型，并支持中英文的文字识别与视觉定位，还能实现多图对话和故事生成功能。

<br>
<br>
<br>

## Pre-training

### Pre-training Data

为了构建一个有效且高质量的预训练数据集，阿里提供了一套全面的数据收集与预处理流程：

- 首先，从多个source收集多样化的数据，包括网页、百科、书籍、代码，且保证多语言特性，其中大量为中文与英文，保证覆盖全球化和本土化应用；

- 对于HTML网页数据，从其中抽取数据，并利用语言识别工具识别数据所属的语言；

- 基于精确去重和模糊去重，避免数据集存在大量重复的信息。前者在对语料进行规范化（Normalization），例如统一格式、处理一些标点符号等操作后，如果文本完全相同则进行去重；后者则使用MinHash + LSH的方案进行重复文本匹配删除；

- 对低质量或者有害数据进行过滤，使用基于规则和基于模型（质量评估模型 + 有害内容检测模型）两种方案进行过滤。此外，也会人工进行采样审核以确保质量；

- 对优质数据进行上采样，使模型有更多机会基于优质的信息进行学习；

- 在预训练时，加入多任务指令数据，已经有研究表明，在预训练中加入多任务指令学习能够提高模型的zero-shot和few-shot能力。


<br>

### Tokenizer

使用OpenAI开源的分词库``tiktoken``中的``cl100k base``（GPT-3.5、GPT-4所使用的）作为基础分词库，这个库底层是基于BBPE算法进行tokenization的。

针对中文和其他语言，特别补充了高频汉字和词汇的token，这种优化让Qwen 在中文场景下更高效，也更贴合多语言任务。

此外，采用“数字拆分为单个字符”的策略。比如 “123” 会变成 “1 2 3 ”。原因是数字在推理、算术、代码任务中比较敏感，拆分后能提升泛化性和计算相关的准确性。

最终扩展后的词表大小约152K，但是更大的词表并没有降低下游性能，而且在``压缩效率``（更少 token 表达更多信息） 方面优于很多模型，这样也能使得推理更为高效。

<div align=center>
	<img src="img_2.png"/>
</div>

<br>

### Model Structure

Qwen的模型基于Transformer架构，选择在LLaMA架构上进行修改。 

- 使用Untied Embedding而不是Tied Embedding，使得输入输出需要训练两个不同的embedding权重，增大显存开销，但是这样输入输出可以学习到不同的语义映射方式，有可能能换取更好的性能；

- 使用RoPE作为位置编码方案，但是在逆频率矩阵计算时采用FP32的精度，提升了模型的精度与性能表现；

- 选择SwiGLU作为激活函数，此外，将FFN的放大倍数改为8/3（而不是4倍）；

- 使用不用计算均值、更轻量的RMSNorm来代替LayerNorm，结合Pre-norm的形式，保证了Qwen的训练稳定。

大多数的网络层去除bias，但是QKV注意力层保留了bias，用于增强模型的外推能力。这里苏神有在自己的博客中进行阐述，我们分析Key部分的bias，对于token：$m$和$n$不使用RoPE的注意力计算而言有：$a_n= \frac{e^{q_m \cdot (k_n+b)}}{\sum_t e^{q_m \cdot (k_t+b)}}$，可以拆分为：$a_n= \frac{e^{q_m \cdot k_n } \cdot e^{q \cdot b } }{\sum_t e^{q_m  \cdot k_t } \cdot e^{q_m \cdot b }}$。此处$b$与$n$无关，可以看出对于此处token$n$对于token$m$的注意力计算而言，$e^{q_m \cdot b}$为一个常数，所以可以约掉。但是对于含有RoPE的注意力而言，会对Q和K进行旋转变换：$a_n= \frac{e^{(q_m+a)R_m^T \cdot R_n (k_n+b)}}{\sum_t e^{(q_m+a)R_m^T \cdot R_t(k_t+b)}}$，展开后得到：$a_n= \frac{e^{q_mR_m^TR_nk_n+aR_m^TR_nk_n+q_mR_m^TR_nb+aR_m^TR_nb}}{\sum_t e^{q_mR_m^TR_tk_t+aR_m^TR_tk_t+q_mR_m^TR_tb+aR_m^TR_tb}}$。可以发现，最终的$b$这几项都与$Rt$相关，而$R_t$是不同的，无法约掉，因此这部分bias最好进行保留。

Qwen相关模型大小、架构以及超参数设置如下图所示：

<div align=center>
	<img src="img_3.png"/>
</div>

<br>

### Pre-training Method

Qwen的预训练遵循自回归语言建模方法，即根据上下文来预测下一个token。

训练时指定上下文长度为2048，在构建batch时，会对文档进行打乱与合并，然后截断到指定的上下文长度，避免模型按指定顺序看到数据。

为了提高计算效率并减少显存占用，在注意力模块中采用了Flash Attention，Qwen的所有模型都使用BFloat16混合精度来保证训练稳定性。

在优化器的选择上，Qwen与LLaMA相同，都使用了$\beta_1=0.9$和$\beta_2=0.95$的AdamW作为优化器。学习率调度采用cosine lr schedule，并且为每一个不同规模的模型都设置了峰值学习率，学习率按照schedule逐步衰减到峰值学习率的10%。

<br>

### Long Context

基于Transformer结构的LLM在扩展上下文长度时有很大挑战，因为注意力机制的时间复杂度为$O(n^2)$，很容易OOM。Qwen在训练时采用2048的上下文长度，但是在应用时需要支持更长的上下文，因此引入一些技巧来解决。

#### NTK-aware

Qwen在推理时引入一种training-free的方法，即NTK-aware插值。对于普通的位置插值（Position Interpolation）而言，其进行简单缩放后，会丢失高频信息，影响长文本建模。而NTK-aware能够调整RoPE的基数，更好保留位置信息，更为稳定。

为了进一步提升性能，Qwen中还实现了一种简单的扩展方法，即动态NTK-aware插值，通过 分段动态调整缩放，从而避免了性能的严重退化。


#### 两种注意力机制 

Qwen还结合了两种注意力机制：LogN-Scaling和Window Attention。Qwen团队还观察到，模型的长上下文建模能力在不同层次间存在差异：低层比高层对上下文长度扩展更敏感。基于这一观察，为各层分配了不同的窗口大小：低层使用较短的窗口，高层使用较长的窗口。

<br>


### Evaluation

预训练Qwen的评估涵盖了7个常用的benchmark，在评估中，关注未经过Alignment的模型。结果表示，三种Qwen模型在所有下游任务中均展现了优异表现：

<div align=center>
	<img src="img_4.png"/>
</div>

实验也验证了，Qwen采用的Dynamic NTK、LogN-Scaling和Window Attention技术能够在上下文长度增大时有效保持甚至提升模型性能。此处用困惑度来衡量模型性能：

<div align=center>
	<img src="img_5.png"/>
</div>

<br>
<br>
<br>

## Alignment

Qwen使用了SFT + RLHF的方案进行Alignment，强化模型指令遵循能力以及有害内容过滤能力，使得模型输出更符合用户意图。

### SFT

#### SFT Data

传统的数据集一般包含大量的Prompt（带有questions,、instructions和answers），但是Qwen团队进一步标注了一种``human-style conversations``，模型学习的不再是孤立地回答一个问题，而是如何在多轮交互中持续地、有上下文地提供帮助，提升了模型的对话记忆、推理以及主动引导能力，也使得模型的回答看起来“更有人味”。

而为了在用户进行各类方式提问时不限制模型的能力，Qwen团队对Prompt模版格式化数据进行删除，这些僵化、固定的格式数据会引导模型对高度结构化的信息进行学习，影响其泛化能力。


Qwen团队发现，在训练阶段，数据的“组织方式”和数据的质量同等重要，即使拥有高质量的人工标注数据，如果以错误的方式喂给模型，最终性能也会大打折扣。

为了解决这个问题，Qwen团队采用了OpenAI提出的ChatML（Chat Markup Language）格式，这是一种结构化的元数据标记，将对话的角色清楚的标记出来，例如：

<div align=center>
	<img src="img_6.png"/>
</div>

在训练时引入这类标签，模型通过学习将这种模式内化，推理时，用户只需要输入自然语言指令，将这些指令添加上ChatML模板后提交至模型，而后输出回复内容。Qwen团队在技术报告中也提到了类似的方案``Anthropic``，其利用``\n\nhuman``和``\n\nassistant``来区分user和assistant，但是由于这些特定短语是常见词汇，模型可能难以区分其他上下文中的这些词汇。


#### Training

采用与预训练相同的自回归学习方法，在ChatML的数据格式上，对于对系统和用户输入部分应用损失掩码，使模型重点关注assistant输出部分。使用$\beta_1=0.9$、$\beta_2=0.95$、$\epsilon=10^{-8}$的AdamW作为优化器. sequence length限制为2048，batch size为128。训练共进行4000个epoch，学习率在前1430个epoch内逐步提升至峰值$2 \times 10^{-6}$。为防止过拟合，采用0.1的weight decay、0.1的dropout和1.0的gradient clip。


<br>

### RLHF

SFT虽然有效, 但是容易过拟合导致泛化性有限. Qwen团队使用了RLHF进一步与人类偏好进行对齐。

#### Reward Model

和InstructGPT利用GPT-3微调得到Reward Model相比, Qwen则是像训练一个完整的大模型一样采用预训练+微调的方案构建一个Reward Model. 其中的预训练模型步骤称为偏好模型预训练(PMP), 这一步骤需要大量的数据进行对比学习, 即给定一个query, 数据包含两个不同的response(A/B), 且包含人类对这两个response的偏好标注。 预训练基于这类数据让模型初步、广泛地学习“什么是好，什么是坏”。这个数据集可能混合了来自不同领域、不同风格的偏好数据。

而在微调阶段, 则使用标注质量更高的数据进行训练，类似的，Qwen团队收集各类Prompt，让人类根据Qwen模型输出的回答进行反馈，用这些反馈来调整Reward Model。为了保证Prompt的多样性与复杂性，Qwen团队对Prompt所属任务类型创建了一个大约包含6600个详细标签的分类系统，并实施了一种平衡采样算法，该算法在选择Prompt供奖励模型训练时，同时考虑多样性和复杂性。

微调时为了让这些Prompt得到更为多样的response，Qwen团队使用了不同大小的Qwen模型和不同的解码采样策略，从而使得模型生成更为多样的回复。Qwen团队使用与待Alignment的模型大小一致的Qwen作为初始化的Reward Model。

Qwen团队在Reward Model的原始模型结构中加入了池化层，lr恒定为，批次大$3 \times 10^{-6}$，batch size为64。sequence length设为2048，训练进行单个epoch。


#### Reinforcement Learning

Qwen使用PPO，其中涉及四个模型：

1. Policy Model：这是要训练的主要模型，根据Prompt生成response，在训练过程中，该模型进行学习以生成更高奖励的response；


2. Reward model：接收一个Prompt和Policy Model生成的response，并给出一个Score，在PPO中该模型参数冻结；


3. Value Model：估计给定Prompt的预期累积奖励（价值）。这个估计值用于帮助更新策略模型，判断当前生成的回复是比预期更好还是更差；


4. Reference Model：通常是最初始的模型，其参数在PPO中完全冻结，作用是为PPO过程提供一个基准。


在正式进行PPO迭代之前，先暂停Policy Model的更新，只用初始的奖励信号在50个epoch内训练Value Model，使得Value Model先学会准确地估计状态预期价值。

#### 双响应采样

对于训练集中的每一个Prompt，策略模型同时生成两个不同的response。这种方法提供了更多样化的数据供策略学习，类似于在同一个问题上尝试两种不同的解题思路然后得到反馈。内部实验证明，这种策略比只生成一个回复更有效，可能因为它能更好地探索策略空间，减少训练方差。


<br>

还有一些超参数相关设置：KL散度系数设为0.04、使用运行均值对奖励进行归一化、：使用了低学习率(Policy: 1e-6, Value: 5e-6)、进行价值损失裁剪（clip=0.15）、解码策略使用0.9的Top-p而不是完全随机。


#### Alignment Tax

RLHF使得LLM获得了符合人类期望的能力，但是这个过程中可能是LLM遗忘以往学习到的多样性能力（代码生成、数学推理等），KL散度惩罚应对阅读理解与常识理解部分能力的衰退已足够，但是对于这部分精细化能力的衰退还是无能为力。

Qwen团队为了缓解Alignment Tax，在PPO的损失函数上额外加入一个来自预训练数据的损失项，这相当于一种“正则化”，不断提醒模型保持其原有的广泛语言能力和知识，从而有效减轻对齐税，这个环节需要远大于PPO所需偏好数据量的预训练数据。

<br>

### Evaluation

实验将三种不同的Qwen微调模型和GPT-4这四种模型同GPT-3.5进行对比试验，由图可知，RLHF后进行Alignment的模型性能更好：


<div align=center>
	<img src="img_7.png"/>
</div>
