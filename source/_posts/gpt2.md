---
title: GPT-2技术报告阅读笔记
author: luvisdru9
date: 2025-08-20 00:00:00
updated: 2025-08-20 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: GPT-2技术报告阅读笔记
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

GPT-2文章中指出了监督学习的核心弱点：脆弱性与敏感性，监督学习在训练数据分布上表现优异，但是数据分布一旦稍有变化，则性能急剧下降，这样训练出来的系统称为Narrow Expert，单任务单领域的训练范式无法进行举一反三的泛化功能。因此，文章主要宣传的是下游任务中Zero-shot的思想。

<div align=center>
	<img src="img_1.png"/>
</div>

<br>
<br>
<br>

## 任务转换

对于以往的单任务建模而言，任务框架可以被描述为对条件概率``p(output|input)``进行建模，而一个通用的系统应该能够执行多个任务，则应该以“task”为条件，即建模为``p(output|input, task)``。以往的多任务学习的实现方式通常体现在模型架构方面，为不同任务设计不同的架构，这种方式复杂、不灵活，且难以扩展。

但是根据McCann等人在2018年的工作可知，自然语言本身就是一个极其灵活的元语言，可以用来指定任务、输入和输出，例如：

- 翻译任务可以写成一个序列：``translate to french, english text, french text``
- 问答任务可以写成：``answer the question, document, question, answer``

所有的监督学习任务都可以被重新表述为一个“符号序列”。一旦做到了这一点，所有这些不同的任务都被“压平”到了同一个维度——``预测序列中的下一个符号``。那么从形式上而言，如果一个LLM很好地掌握了无监督语言建模，那么它也就很好地掌握了序列中蕴含的各类下游任务，可以看做语言建模任务的一个子集。

但实际上在McCann等人的假设中，序列数据非常干净整洁，互联网上的数据并不如这样干净而是非常杂乱，而gpt-2认为在这种数据模式下，任务范式仍然存在，模型为了更准确地预测出这些文章后续的文本（比如，预测出步骤中的下一个词），它被迫去理解“提问”和“回答”之间的逻辑关系。为了更好地完成“预测下一个词”这个简单任务，它必须学会隐藏在文本深处的各种复杂任务，且初步小规模实验也证明了有效性，但是出现了``训练慢``的情况。那么多任务学习的目标就从``模型结构设计``转换到了``工程上是否能实现这个问题``：

1. 能否构建一个足够大的模型？
2. 能否收集足够多、足够多样的训练数据？
3. 能否有足够的计算资源（算力）来训练这个模型？

<br>
<br>
<br>

## Training

### Data

OpenAI爬取了一个新的网络数据集，强调文档质量，只爬取了经过人类策展/筛选的网页。数据来自社交媒体平台 Reddit的所有出站链接，这些链接至少获得了3个karma（声望值），这可以被看作是一种启发式指标，用于判断其他用户是否认为该链接有趣、有教育意义或只是好笑，从而获取高质量的内容。

同时，移除了所有来自Wikipedia的数据，因为维基百科中一般存在很多QA对的信息，会对Zero-shot产生影响。这个最终的数据集名为WebText，包含略超过 800 万个文档，文本总量为 40 GB。

<br>

### Tokenizer

使用BPE进行分词，但是BPE会包含常见单词的许多变体，例如 ``dog, dog., dog!, dog?``。这导致了对有限的词汇表位置和模型容量的次优分配。因此引入了约束：不允许BPE将属于不同Unicode类别（如字母、数字、标点符号、空格）的字节合并在一起，阻止跨类别合并。具体的流程：

1. 将文本编码为 UTF-8 字节序列；

2. 将每个字节映射为 Unicode 字符；

3. 使用 BPE merge rules 对字符序列进行合并。

<br>

### Model Structure

沿用了gpt-1的transformer-based结构，但是进行了一些小改动：

- 使用Pre-norm作为归一化方案，且在最终的自注意力层之后添加了一个额外的LN；
- 防止残差叠加导致的梯度爆炸，初始化每个残差层权重时缩小一个因子`` 1/sqrt(N)``，N为残差层个数；
- 训练数据量增大，batch_size从 64 增加到 512，seq_length大小从 512 增加到 1024。

文章对比了四种不同参数量的模型，最小的对标gpt-1，medium则对标BERT-large，最大的则是gpt-2，文章经过人工调整lr，使得各个模型在保留样本上都取得了各自的最小困惑度，但是显示仍然欠拟合，这证明当前的模型参数量仍然不足够，需要更大规模的模型，感觉这里已经出现了Scaling Law的雏形。

<div align=center>
	<img src="img_2.png"/>
</div>

<br>
<br>
<br>

## 参考资料

- [《Language Models are Unsupervised Multitask Learners》](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)
