---
title: GPT-1技术报告阅读笔记
author: luvisdru9
date: 2025-08-20 00:00:00
updated: 2025-08-20 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
  
categories: NLP
description: GPT-1技术报告阅读笔记
keywords:
  - NLP
  - LLM
  - DL
  - ML
  
#top_img:
#comments:
cover: img_9.png
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

OpenAI由2018年介绍了一种名为“生成式预训练”（Generative Pre-Training，简称GPT）的新型语言模型，该模型通过在大规模语料库上进行训练，能够学习自然语言的模式和规律，从而实现更好的语言理解。

## 模型结构

GPT-1模型由12层transformer decoder组成，masked self-attention的hidden_state为768，MHA包含12个attention heads，FNN中间层会对768先进行升维至3072，参数量大小为117M。

<div align=center>
	<img src="img.png"/>
</div>

<div align=center>
	<img src="img_1.png"/>
</div>


<br>
<br>
<br>

## Pre-train

作为一个自回归语言模型，GPT-1预训练使用前n个token预测后一个token的无监督预训练架构形式，进行自回归预训练。

<div align=center>
	<img src="img_2.png"/>
</div>

在预训练阶段，GPT-1采用BooksCorpus数据集进行预训练，该数据集包含了大量不同类别的书籍，其中具有大量long-range信息。文章将大小大致相同的数据集ELMo进行比较，不同的是ELMo在句子层级上被打乱，丧失了long-range结构，模型在这个数据集上性能交叉，困惑度达18.4。

<div align=center>
	<img src="img_3.png"/>
</div>

<div align=center>
	<img src="img_4.png"/>
</div>

### 预训练中的一些设置：
1. 使用Adam作为优化器；
2. 在前2000步中采用warmup将lr从0线性增加到最大值2.5e-4，而后则按照余弦函数的曲线逐渐下降为0；
3. 训练进行100个epoch，每一个minibatch包含64个样本，每一个样本长度为512个token；
4. 因为transformer中采用LN对输出分布进行稳定，所以权重可以初始化为简单的均值为0、标准差为0.02的正态分布；
5. 使用BPE作为tokenizer，缓和了OOV问题，通过40000次merge合并操作构建了这个词汇表，词汇表大小约为4万；
6. 在残差、embedding、attention等位置使用概率为0.1的dropout，同时还使用系数为0.01的L2正则化惩罚除bias和LN参数外的参数；
7. 使用GELU作为激活函数；使用可学习的位置编码；
8. 在数据预处理中，使用ftfy修复文本中的编码错误和怪异格式，统一标点和空格，最后使用spaCy工具进行初步的分词。

<div align=center>
	<img src="img_5.png"/>
</div>


<br>
<br>
<br>

## Post-train

GPT-1使用有监督微调架构进行下游适配，给定带有标签的数据集，添加线性输出层进行，使用标准交叉熵损失函数进行微调，同时微调时引入如预训练环节中的辅助语言建模损失函数，提高泛化能力和加速收敛。

<div align=center>
	<img src="img_6.png"/>
</div>

此外，对于不同的有监督微调任务，需要进行输入格式的转换，可分为文本分类、文本蕴含（判断一个前提Premise是否包含一个假设Hypothesis）、文本相似度、文本问答（给定一个上下文 z、一个问题 q 和一组候选答案 ${a_1, a_2, ..., a_k}$，从中选出正确答案）等。

由于gpt-1为自回归模型，是在连续句子上训练得到，对于某些需要句子对或者三元组的任务而言理解不足，gpt-1通过引入起止符和分割符将这种不连续的关系表示为了连续关系。对于具体如下图演示：

<div align=center>
	<img src="img_7.png"/>
</div>

微调大部分超参数设置与预训练时一致，此外，gpt-1在classifier上添加了概率为0.1的dropout，且对于大多数任务，使用 6.25e-5的lr和32的batch size，在实际微调中，一般收敛较快，3个epoch已经足够；采用线性学习率衰减策略，并在前0.2%的update中进行Warmup；辅助语言建模损失的权重设置为0.5，尽量保留它从预训练中学到的通用语言知识。

<div align=center>
	<img src="img_8.png"/>
</div>


<br>
<br>
<br>

## 参考资料
- [《Improving Language Understanding by Generative Pre-Training》](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)