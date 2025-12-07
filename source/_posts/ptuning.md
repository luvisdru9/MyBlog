---
title: Prompt Tuning -《The Power of Scale for Parameter-Efficient Prompt Tuning》论文阅读笔记
author: luvisdru9
date: 2025-10-25 00:00:00
updated: 2025-10-25 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: Prompt Tuning -《The Power of Scale for Parameter-Efficient Prompt Tuning》论文阅读笔记
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

## Introduction（Hard Prompt & Soft Prompt）

自大模型出现以来，各路研究人员都在寻找合适的方案讲这些模型适配到下游任务重，例如Peters等人提出的ELMo则冻结预训练模型参数，只学习其中各层表示进的特定任务加权。但是自 GPT 和 BERT 出现以来，最主流的下游任务适配方法变成了模型微调（finetune），即在下游任务中重新调整全部模型参数。

在GPT-3的技术报告中，作者通过Few-shot的方式对GPT-3进行In-context Learning，从而向我们展示了Hard Prompt设计在模型下游任务适配上的可能性（这里关于Hard Prompt和Soft Prompt的区别推荐观看这篇综述文章：[提示学习Prompt Tuning：面向研究综述](https://zhuanlan.zhihu.com/p/524383554 "提示学习Prompt Tuning：面向研究综述")）。

但是这些Hard Prompt的方法都存在几个关键的不足之处：
1. 需要人为不断调试，Prompt编写容易出错； 
2. 模型的上下文长度限制，导致了Prompt的长度也受到限制，意味着Prompt能表达的下游任务信息也受到了限制；
3. 受到上面两点的限制，Hard Prompt-based的模型在下游任务中的性能仍明显低于全参数微调模型。例如，GPT-3（175B）在 SuperGLUE 上的 few-shot 表现比 finetune 后的 T5-XXL（11B）低了 175 分，即使GPT-3的参数比后者多了 16 倍。

知道了Hard Prompt的不足之处，一些研究人员开始着手设计基于Soft Prompt（也可以叫Continuous Prompt）的适配方法，简单来说，就是把Prompt的生成本身作为一个任务进行学习，经典的有类似“Prefix Tuning”这类方法，且这类Soft Prompt方法最后可以利用“不同下游任务Prompt + 冻结原始模型”的方案适配不同的下游任务，保留了原始模型的部署高效性，如下图所示。而本文提出了一种名为“Prompt Tuning”的方案，可看作为Prefix Tuning的简化形式。


<div align="center">
    <img src="img_1.png"/>
</div>


<br>
<br>
<br>

## 任务描述

作者使用的是$T5$模型，也遵循其中“text-to-text”的方法，将原本的模型分类任务$Pr(y|X)$（y为一个类别label）建模为$Pr_{\theta}(Y|X)$（Y为一个类别label的token序列）。

GPT-3中，Prompt Token是由固定的词汇表得到的，模型的参数$\theta$也是冻结的，此时，生成任务可建模为$Pr_{\theta}(Y|[P;X])$。此时如果想要找到最优的Prompt，则需要使用人工搜索/不可微分的搜索方法。

而Prompt-tuning则消除了这种“固定参数$\theta$”的限制，使得Prompt拥有可更新的参数$\theta_P$。具体来说，相比于原来“从词表中抉择如何选取token”而言，Prompt-tuning则是固定token的选择，而不断学习它们的embedding，此时生成任务可以建模为$Pr_{\theta; \theta_P}(Y|[P;X])$。此时不再需要人工进行搜索，而是可以通过反向传播在主模型参数$\theta$冻结的情况下，更新参数$\theta_P$。

下面对Prompt-tuning的任务进行详细描述：输入的$n$个token序列为$\{x_1, x_2, ..., x_n\}$，而后将这$n$个token通过原有模型的embedding变为维度为$e$的向量$X_e \in \mathbb{R}^{n \times e}$。可调的Soft Prompt表示为$P_e \in \mathbb{R}^{p \times e}$，$p$意味着Prompt Token序列的长度，后续训练中直接使用梯度更新这个矩阵$P_e$。最后训练收敛后，输入数据可以按“Prompt + Input”拼接为$[P_e;X_e] \in \mathbb{R}^{(p+n) \times e}$。

对于$P_e$的初始化而言，我们可以选择随机初始化从0开始训练；也可以将每个 Prompt Token 初始化为模型词表中某个词的 embedding；对于分类任务而言，因为我们希望模型在输出中生成这些类别的token sequence，用那么我们也可以用合法label token sequence的 embedding 来初始化 prompt 可以帮助模型倾向于生成正确类别的输出。


<br>
<br>
<br>


## $T5$中的Span Corruption

$T5$基于Span Corruption（片段掩码）任务进行训练，例如给定一个句子：``Thank you for inviting me to your party last week``。

我们可以构建数据对：
- 输入：``Thank you <X> me to your party <Y> week``；
- 输出：``<X> for inviting <Y> last <Z>``。

其中的``<X>``、``<Y>``、``<Z>``都是**Sentinel Token**，这是一种特殊标记。由于这种标记的存在，使用这种方法进行预训练，模型从未见过真正自然的输入文本，也从未见过真正自然的输出文本，比起进行自回归语言建模预训练，这种方法会产生一定的“非自然语言输出倾向”。

对于这种“非自然语言输出倾向”，一般的finetune能够调整模型所有参数，改变模型的输出风格与偏好，能够很有效地消除这种不自然。但是对于Prompt-tuning而言，并不会对原模型参数进行调整，无法改变模型的先验，只能在现有风格上做微小引导，极大可能保留这种不自然，从而影响模型表现。为了探究这一个问题，于是作者使用三种方案展开了对比实验：

- Span Corruption：直接使用原版预训练 T5 作为冻结模型；

- Span Corruption + Sentinel：在下游任务的输出中也手动加 sentinel 特殊标记，使其更接近预训练目标格式；

- LM Adaptation：继续在自然语料上训练 T5，让模型使用类似 GPT-3 的自回归方式进行再训练，这个方案只需执行一个epoch，就能得到一个可以在任意任务上 Prompt-tuning 的模型。

作者希望通过这种$LM Adaptation$的方式将T5转换为更像GPT-3这样的模型，能够具有更强的自然语言能力，对应地也就对Prompt的影响更为敏感。但是相比于GPT-3的从零开始的自回归预训练，这种方案相当于后期再训练，是否能够在性能上真正替代前者尚未确定，所以作者不只是将LM Adaptation进行一个epoch，而是测试了不同的epoch数，上限为100K。


<br>
<br>
<br>


## 对比实验

下图展示四种模型适配方法的对比实验，其中绿色曲线代表默认配置的Prompt-tuning，其中``LM Adaptation Step``为100K、使用类别标签embedding初始化$P_e$、Prompt Token Sequence长度为100。每个任务分别训练一个Prompt，实验在SuperGLUE上进行评估，纵轴为评估分数，横轴为模型参数量:

<div align="center">
    <img src="img_2.png"/>
</div>

可以看到，随着模型参数量的增大，Prompt-tuning的性能和Model Tuning(finetune)的性能差距越来越小，且远远超过Prompt Design(Hard Prompt)。在模型参数量达到T5-XXL（11B）的规模时，Prompt-tuning以远远小于Model Tuning的训练量获得了几乎一致的性能，且在T5-Small(60M)和T5-Large(770M)上进行Prompt-tuning的模型表现也比在GPT-3 XL(1.3B)和GPT-3(175B)上使用Prompt Design要更好。


<br>
<br>
<br>

## 消融实验

文中从三个维度进行Prompt-tuning的消融实验，实验结果具体如下图所示：


<div align="center">
    <img src="img_3.png"/>
</div>

<br>
<br>
<br>

### Prompt 长度

在不同规模的模型上分别用不同长度的Prompt进行Prompt-tuning，分别设置为$1、5、20、100、150$，如上图(a)所示，相比于长度为1的Prompt，更长的Prompt能够取得更好的表现。值得注意的是，对于XXL规模的模型而言，Prompt长度为1似乎也能获得不错的效果，这可以侧面反映一个现象：即模型越大，实现较好下游任务性能所需要的提示越少，因为模型本身已经容纳了足够的知识量。另一方面，从实验来看，长度为20后再往后提升Prompt长度似乎对模型性能的提升没有产生明显效果，甚至有负面影响。


<br>


### Prompt 初始化

在不同规模的模型上分别用不同方式初始化embedding的Prompt进行Prompt-tuning，如上文所述，包括随机初始化（在[-0.5, 0.5]区间均匀采样）、从词表中采样（从T5的词表中最常用的5000个token中选取）、基于类别标签初始化（这里具体的做法是：对下游任务的每个类别，取其文本形式对应的 embedding 用作 prompt 中的一部分；如果类别文本由多个 token 组成，则取平均，如果 Prompt 比类别数长时，剩余 token 从词表中采样）。

如图(b)所示，我们可以看出，基于类别标签初始化进行Prompt-tuning的效果最好，且在小规模模型上表现出的各方案的性能差异到了大规模模型上都几乎消失了，这里应该也和大规模模型海量的知识有关。此外，作者还发现，使用类别标签初始化训练后，最后的Prompt Token Embedding向量语义仍然与初始的类别向量语义十分靠近，只是整体的Prompt经过训练后已经无法解释了。


<br>

### 预训练目标

图(c)在前文所述的Span Corruption、Span Corruption + Sentinel和LM Adaptation三种方案得到的预训练模型上进行Prompt-tuning，并对比性能。可以看出T5这种Span Corruption预训练出来的模型完全不适合进行Prompt-tuning，哪怕在下游任务中主动迎合偏好，加上特殊标记符号，效果也不尽人意。而按照自回归预方法对T5模型进行100K epoch的再训练则明显取得了更好效果，且由图(d)可知，epoch轮次越大效果越好。作者还提到，对于规模较大的模型，即使原本的预训练目标不适合Prompt-tuning，仍然能够通过 Prompt 得到正常、合理的输出，但是对于中小规模的模型来说，可能就会学不会任务甚至全错。

<br>
<br>
<br>


## 与其他适配方案训练参数量的对比

<div align="center">
    <img src="img_4.png"/>
</div>

<br>
<br>
<br>


## 领域迁移的鲁棒性

作者在两个任务上研究zero-shot的领域迁移问题。对于同一个任务，在一个领域A数据集上进行训练，而后在领域B数据集上进行相同的任务验证其性能根据实验结果，发现在领域差异更大的情况下，Prompt-tuning的收益更大，一般的模型微调对模型大部分甚至全部的参数进行训练，更容易过拟合所训练的下游任务，不利于相似任务在不同领域的迁移。


<div align="center">
    <img src="img_5.png"/>
</div>

<div align="center">
    <img src="img_6.png"/>
</div>

<br>
<br>
<br>


## Prompt 集成

神经模型在相同数据上以不同初始化训练后再进行集成通常可以提升任务性能，并且将各个模型的表现进行对比有助于估计模型的不确定性。然而，随着模型规模变大，模型集成会变得不切实际。除了需要存储 N 份模型，在推理时还需要运行 N 个独立模型，这会导致非常高的计算成本。而Prompt-tuning则提供了十分高效的集成方案，通过在同一任务上训练 N 个不同的 Prompt，我们实际上得到了 N 个“任务专用模型”，但它们共享底层语言模型的所有参数。

文中进行了相关的实验验证集成Prompt的性能，对于同一个任务而言，训练得到5个Prompt，并将输入也复制5份，拼接得到一个batch的数据：$\{[P_1; X_1], [P_2; X_2], [P_3; X_3], [P_4; X_4], [P_5; X_5]\}$，进行一次forward后，采用简单多数投票进行集成，实验结果如下表所示，结果证明：Prompt集成模型均优于以下几种方案：

1. 单独使用一个Prompt；
2. 多个Prompt单独性能的平均；
3. 单独使用的Prompt中最好的性能.

<div align="center">
    <img src="img_7.png"/>
</div>

<br>
<br>
<br>


## Prompt 可解释性

一般来说，一个可解释的 Prompt 应该是自然语言写成的，清晰地描述任务，明确要求模型进行某种输出，使我们可以理解模型为什么会得到特定行为。但 Prompt-tuning 工作在连续 embedding 空间中，而非离散词空间，因此非常难进行解释。

文中为了探索这种Soft Prompt的可解释性，计算了每一个训练得到的Prompt Token在词表中余弦距离最近邻的词，结果发现：每个 Prompt Token 的前 5 个最近邻词会形成语义聚类，例如词汇上就一致的：``{Technology / technology / Technologies / technological / technologies}``。 或是词汇上不相似但是具有强语义联系的：`` {entirely / completely / totally / altogether / 100% }``。这说明Soft Prompt也学会了一定的类词汇级别的语义信息。

当Prompt-tuning使用类别标签初始化时，作者发现，在训练后这些类别标签往往仍保留在这些Soft Prompt的最近邻词中，证明这些Prompt记忆了这些类别信息。

如果Prompt长度很长，会发现多个位置学到的语义重复，这可能意味着：1、Prompt 的容量过大，产生冗余；2、Prompt 序列中缺乏结构，模型不能很好地将所需要的信息定位到特定位置，这似乎也解释了前文实验中为什么过长的Prompt并不能提升模型性能，甚至产生负面作用。


<br>
<br>
<br>


## 参考资料

- [《The Power of Scale for Parameter-Efficient Prompt Tuning》](https://arxiv.org/abs/2104.08691)
- https://zhuanlan.zhihu.com/p/524383554

