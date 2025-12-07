---
title: Prefix Tuning -《Prefix-Tuning:Optimizing Continuous Prompts for Generation》论文阅读笔记
author: luvisdru9
date: 2025-11-07 00:00:00
updated: 2025-11-07 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: Prefix Tuning -《Prefix-Tuning:Optimizing Continuous Prompts for Generation》论文阅读笔记
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

在“大模型的下游任务适配”这个方向上，研究者已经开发出来若干种可用的方案，例如全量微调（finetune），但是这类方法对于大规模模型所需的计算量过大；也有类似Adapter-tuning的方法，冻结大部分参数，只添加少量可训练的层，通常能够以2–4%的参数接近全量微调的性能；而GPT-3则使用了更极端的方案——只依赖Prompt Design进行in-context learning。


### Hard Prompt Design && Prefix-tuning

一般人工进行Prompt Design对模型进行引导通常具有如下的局限性：

1. Prompt表达能力有限，只能用现成词表中的单词组合表达条件，同时，这种离散的Prompt空间也难以进行优化；
2. 模型难以理解指令，类似于“请生成文章的摘要”这类的Prompt在人类看来已经足够清晰，但是对于模型而言可能并未如此，这也是为什么这类方案的效果时常有效。

Prefix-tuning的思想则是：与其费力寻找离散的词，不如直接优化连续的embeddings，即在模型的输入前加入可训练优化的embedding序列，也即“Prefix”。根据这个特点，可以只为每一个下游任务学习一个“Prefix”而不是全量finetune，这样只需要推理时在输入前加上这个“Prefix”，便实现了下游适配。


<div align="center">
    <img src="img_1.png"/>
</div>


<br>
<br>
<br>


## 模型任务适配

对基于Transformer架构的自回归模型来说，训练时，在某个“隐态层”上，我们将“Prefix”直接加在输入的前面，如下图所示，$h_1$、$h_2$是直接添加的“Prefix”，$h_{3-15}$则是正常输入和输出的拼接，可描述为：$z = [PREFIX;x;y]$。


<div align="center">
    <img src="img_2.png"/>
</div>

对基于Encoder-Decoder结构的模型（如BART）来说，训练时，在Encoder和Decoder中的某个“隐态层”上，我们都分别将“Prefix”直接加在输入和输出的前面，如下图所示，$h_1$、$h_2$是直接添加在Encoder中的“Prefix1”，$h_9$、$h_{10}$是直接添加在Dncoder中的“Prefix2”，可描述为：$ z =[PREFIX;x;PREFIX;y]$。

<div align="center">
    <img src="img_3.png"/>
</div>

<br>
<br>
<br>


## Training

将最终需要加上去的若干个Prefix向量表示为一个矩阵：$P_{\theta}$，所有需要训练的参数即为$\theta$，这个矩阵可能会非常大。作者发现，如果我们将梯度用于直接训练这个向量矩阵$P_{\theta}$会很不稳定并导致性能有小幅度地下降，因为这种直接优化$P_{\theta}$的形式，对于初始化和学习率设置都非常敏感。

于是文中使用了一种重参数化的技巧，假设$P_{\theta}$维度为$n \times d$，那么每一个Prefix向量$P_{\theta}[i,:]$维度则为$1 \times d$。而后引入一个更低维的向量$P'_{\theta}[i,:]$，其维度为$1 \times d', d' \lt d$，再引入一个较大前馈网络$MLP_{\theta}$，最后我们可以由：$P_{\theta}[i,:] = MLP_{\theta}(P'_{\theta}[i,:])$得到我们需要的$P_{\theta}$，在训练时，P'_{\theta}和$MLP_{\theta}$都需要进行更新，训练结束后，我们只需要拿到最后的结果$P_{\theta}$即可。

作者也在文中阐述了这样做的合理性：Aghajanyan等人在2020年发表的著作[《Intrinsic dimensionality explains the effectiveness of language model fine-tuning》](https://arxiv.org/abs/2012.13255)中写道，模型参数存在几个主要的内在方向（Intrinsic Dimension），在这些方向上进行低维训练也能达到与全量微调几乎一致的效果，作者将这种观念迁移到了$P_{\theta}$的学习上。


### 任务选择

在文章中，作者主要关注两个任务：

- 输入线性化的数据表，输出该表的文字描述，数据集使用了E2E, WebNLG, DART，模型为GPT-2；

- 输入是一篇文章，输出的是该文章的简短摘要，数据集使用了XSUM，模型为BART。

具体如下图所示：

<div align="center">
    <img src="img_4.png"/>
</div>

<br>
<br>
<br>


## 一些实验结果

### 不同适配方案的表现

在table-to-text的任务中，Prefix-tuning只需要$0.1\%$的参数，性能超过了Adapter-tuning，与全量微调一致甚至超过其性能表现。

<div align="center">
    <img src="img_5.png"/>
</div>

在summarization任务中，Prefix-tuning的表现略逊于全量微调，但是性能上相差不算过大，结合其极小参数量训练的优势，也具有一定的应用场景。

<div align="center">
    <img src="img_6.png"/>
</div>


<br>


### 低数据量下不同适配方案的表现

由于所需参数量更小，较小的数据集也能够在Prefix-tuning上产生较好的泛化能力，适配表现更好。而相比之下Finetune远大于Prefix-tuning的参数量也带来了远超过后者的数据量需求，否则会出现欠拟合的现象。


<div align="center">
    <img src="img_7.png"/>
</div>

<br>

### 模型外推（Extrapolation）能力

作者设置了两种不同的跨领域测试实验，1、将模型在新闻类数据上训练，而后在体育类数据上测试（news-to-sports），属于跨领域外推；2、在新闻类数据集中挑选不同的主题数据进行训练和测试（within-news），属于领域内外推。实验结果证明，Prefix-tuning这种模式相比于全量微调，具有更好的外推能力。

<div align="center">
    <img src="img_8.png"/>
</div>

<br>

### 消融实验

#### Prefix Length

作者探究了Prefix长度对模型性能的影响，实验结果表示，过大过小的长度都会影响模型的表现，模型表现会随着长度到达一定值后达到峰值。

<div align="center">
    <img src="img_9.png"/>
</div>

同时，作者发现，一般随着Prefix长度的增加，模型的推理速度并不会受到太大的影响，因为所有Prefix的Attention计算在GPU中都是并行的。


#### Full vs Embedding-only && Prefixing vs Infixing

作者对比了两种Prefix-tuning的方法：

1. Full：每一层上都设置可训练的Prefix，这意味着Prefix可以干预每一层的Attention计算；
2. Embedding-only：只在Embedding层设置可训练的Prefix，后续的计算随着每一层的前馈自动进行。实验结果表示，Full相比于Embedding-only拥有更好的表现。

同时，作者设置了一个挺有意思的实验，将trainable的向量放在不同的上下文位置，prefixing：$[PREFIX;x;y]$，infixing：$[x;INFIX;y]$，实验结果表示Prefixing的性能略好Infixing。作者简单提到是因为Prefix能够影响x和y的激活表示，而Infix只能够影响到y的激活表示，这里其实和自回归性质的LM有关，自回归的训练方法决定了Prefix的放置方法在计算Attention时影响范围更广。

<div align="center">
    <img src="img_10.png"/>
</div>

#### 初始化

作者在少数据量的情况下讨论了Prefix的初始化问题，发现如果使用随机初始化（通常是从高斯分布或均匀分布中随机采样数值）会导致模型表现性能差和方差大两个问题

针对这个问题，作者用真实单词在预训练模型中的激活值（Embedding 或 隐层状态）作初始化选择，最后发现，不同的方案初始化后模型性能由低到高分别为：``随机初始化``、``任务无关的实词``、``任务有关的实词``。因为这些真实的单词更符合预训练模型中的语义空间分布，这样Prefix更容易融入基础预训练模型的知识体系内，不易破坏当前LM的知识结构。


<div align="center">
    <img src="img_11.png"/>
</div>

<br>
<br>
<br>


## 参考资料

- [《Prefix-Tuning:Optimizing Continuous Prompts for Generation》](https://arxiv.org/abs/2101.00190)