---
title: LLaMA技术报告阅读笔记
author: luvisdru9
date: 2025-09-08 00:00:00
updated: 2025-09-08 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: LLaMA技术报告阅读笔记
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

GPT-3基于Few-shot展示了一个现象：模型的能力随着其规模的增大而获得提升。然而，Hoffmann等人在2022年的工作——[《Training Compute-Optimal Large Language Models》](https://arxiv.org/abs/2203.15556)中提到：在固定的计算预算下，最佳性能并不是由最大模型取得的，而是由较小的模型在更多数据上训练得到的。Hoffmann等人修正后的scaling law表明：在特定的训练计算预算下，确定数据集规模和模型规模的最佳配比。

然而，上述的目标却忽视了推理预算，而推理预算在大规模部署语言模型时至关重要。例如大模型虽然能够在训练时快速地达到某个性能，在部署推理时成本却较高；而小模型虽然本身能力有限，需要多轮训练才能收敛到某个性能，但是在推理时成本更低，长期来看更为实用。

Meta的目标是：使用比通常训练更多的数据，来在不同推理预算下都能找到“性能最佳”的模型。LLaMA的规模从7B到65B不等，同时，其训练数据均是开源的，为开源社区提供了又一个重要资源。


<br>
<br>
<br>

## Pre-training

### Pre-training Data

LLaMA的预训练数据来自多个数据源，覆盖了多个领域，其中也不乏包含了其他LLM的开源预训练数据，这里介绍一下：

#### English CommonCrawl [67%]

Meta收集了2017-2020年间的五份Common Crawl dump，使用CCNet Pipline进行数据处理。（CCNet是一个Facebook提出的，从CommonCrawl网页数据中自动构建高质量语料的处理流水线。包含如下几个步骤：

1. 去重：这里使用的是行级别去重，将网页数据按照行划分，使用Minihash、Simhash等算法进行重复文本检测，最后去除这些重复的内容；
2. 语言识别：只保留目标语言的文本，其中LLaMA中主要使用的是英文。具体而言，CCNet使用 fastText 的语言分类器，能在几毫秒内识别一句话的语言。这样做能够收敛数据中包含的语言种类，提高模型的专注性；

3. 质量过滤：使用n-gram进行计算，从而判断句子流畅度，过滤掉低质量的语料。

#### C4 [15%]

在探索性实验中，Meta发现使用多样化的预处理CommonCrawl数据集能够提升模型性能，于是Meta在数据集中加入了一定量的C4数据集。C4也包含去重与语言识别两个步骤，但是在质量过滤步骤使用了启发式的方法，通过标点、长度、字符比例等经验规则来快速去掉低质量语料。


#### Github [4.5%]

Meta使用了 Google BigQuery 上公开可用的 GitHub 数据集，只选公开可再利用的项目，进行许可证筛选。而后进行数据清洗，包括对低质量文件的过滤（如果某些文件的行过短或过长，或字母数字字符比例异常，说明可能是无效代码、二进制/乱码，直接丢弃）和模版代码的去除（通过正则表达式剔除“样板部分”，比如版权声明、自动生成的文件头注释等，这些部分通常与代码主题关系不大）。最后进行文件级别的去重，去掉一些拷贝、fork之类的文件。


#### Wikipedia [4.5%]

引入了 20 种语言的 Wikipedia 数据，清洗掉格式信息后（去掉超链接、注释和格式化样板，比如维基百科里的 HTML 标签、编辑痕迹、表格标记等）作为高质量百科知识来源。



#### Gutenberg and Books3 [4.5%]

这部分数据来源于 Project Gutenberg （多是文学经典、历史文献、语言偏正式、结构完整）和 Books3 （包含现代刊物、更符合当前时代的写作风格与主题）。此外，也进行了书籍层面的去重。


#### ArXiv [2.5%]

从 arXiv 的 LaTeX 源文件中提取科学论文相关语料，去除非正文部分、参考文献和注释部分并统一宏定义。


#### Stack Exchange [2%]

Stack Exchange 是一个问答平台，选择 28 个最大子站点，这些站点具有覆盖面广的特点，但避免小众站点带来稀疏数据。而后去除网页的html标签，得到干净文本。在 Stack Exchange 中，答案具有投票机制，按分数从高到低排序，使得模型优先接触更优的回答。


<div align="center">
    <img src="img_1.png"/>
</div>



<br>


### Tokenizer

使用基于SentencePiece实现的BPE算法进行分词，并做了两点优化：数字拆分为单个字符，提升泛化与数值处理能力；在遇到未知的 UTF-8 字符时回退到字节级别进行分解（如一些表情符号）。

<br>


### Model

Meta在架构上从一些工作上获得灵感，组建了LLaMA的架构：

- 为了提高训练的稳定性，与GPT-3一样，LLaMA使用了Pre-norm作为归一化选择，选择RMSNorm作为归一化函数；

- 从PaLM中获得灵感，使用SwiGLU作为激活函数来代替ReLU，但是使用$\frac{2}{3}4d$维度，而不是PaLM中的$4d$；

- 和GPT-Neo一样，LLaMA移除了绝对位置编码，使用RoPE进行位置编码；

- 使用Loshchilov等人在17年提出的AdamW优化器，超参数设置为：$\beta_1=0.9, \beta_2=0.95$。使用 cosine lr schedule 将学习率衰减最大学习率的10%。此外，LLaMA还是用了0.1的 weight decay、1.0的 gradient clipping、2000步的warmup。Batch size 和 lr 会随着模型规模的变化而变化，具体如下表：

<div align="center">
    <img src="img_2.png"/>
</div>


<br>

### Efficient implementation

Meta为了提高LLaMA的训练效率，采取了一系列措施：

1. 基于$xformers$，使用了一种高效的因果注意力实现方式；
2. LLaMA中手写了backward，避免PyTorch的自动微分保存所有激活值，LLaMA只保存昂贵的激活（比如线性层输出），而不是全部保存，实现定制化checkpoint； 
3. 通过模型并行（model parallelism）和序列并行（sequence parallelism）降低单张GPU的显存压力。同时，将GPU之间的通信与部分激活值的计算一同进行，节省训练时间。


<br>
<br>
<br>

## 参考资料

- [《LLaMA:OpenandEfficient Foundation Language Models》](https://arxiv.org/abs/2302.13971)