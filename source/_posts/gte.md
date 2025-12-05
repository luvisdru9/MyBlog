---
title: 文本嵌入模型-GTE
author: luvisdru9
date: 2025-07-09 00:00:00
updated: 2025-07-09 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: GTE论文阅读笔记
keywords:
  - NLP
  - LLM
  - DL
  - ML
#top_img:
#comments:
cover: img_5.png
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

## 提出背景

- 现在多数文本嵌入模型都是针对特定任务进行训练（如STS和检索任务），泛化性不足
- 大规模预训练依赖于私有数据，开源性和可复现性差
- 模型参数量大，计算复杂度高

<br>
<br>
<br>

## 模型结构

GTE使用与SBert相似的基于transformer的双塔结构（dual-encoder）：

<div align=center>
	<img src="img.png"/>
</div>

输入给定文本``x``，经过语言模型的处理后输出上下文表示向量，而后对这些输出的向量进行平均池化得到文本的最终表示向量：

<div align=center>
	<img src="img_1.png"/>
</div>

<br>
<br>
<br>

## 损失函数

文章中使用余弦相似度衡量两个文本向量的相似度，同时使用一个温度系数`τ`对相似度的尖锐程度进行调整：

<div align=center>
	<img src="img_2.png"/>
</div>

<div align=center>
	<img src="img_3.png"/>
</div>

文章还对传统的InfoNCE损失函数进行了优化：
- 传统的InfoNCE函数只考虑了对比正样本和同一批次内的负样本，拉近正样本对的向量距离，推开负样本对。此时负样本仅来自同一批次的文档，数量有限；同时同一批次样本的其他查询也可以利用作为负样本。
- 优化后的InfoNCE函数扩展负样本池，引入同一批次内的所有查询和文档作为负样本，提升训练效率。批次数据大小为``N``，负样本数量从``N-1``扩大到``2N-1``。
- 优化后的损失函数既增加了负样本的数量，也更好地学习到了文本之间的语义不相关性

<div align=center>
	<img src="img_4.png"/>
</div>

<br>
<br>
<br>

## 模型训练

训练分为两个阶段进行，两个阶段均采用对比学习，使用改善后的InfoNCE损失函数进行训练
预训练：
- 采用无监督预训练的方式，使用各种领域的与**文本相关性**有关的开源数据
- 由于不同来源的数据量显著不同，文章使用多项式分布对不同数据源的数据进行采用，同时保证同个batch里的数据都是来源于同个任务，从而防止模型通过学习到不同任务特性来判断这些数据。同时，由于缺乏**hard negative**信息，为了保证模型能够学习到正负样本的边界，需要使用**较大的batch size**

微调：
- 以少量的人工标注的数据集为基础，利用额外的检索器获得相应的hard negative数据，从而构造出相应的文本相关性三元组（query, positive document, negative document）数据
- 在finetune阶段，由于数据集原本的强监督信息跟hard negative，batch size不必设置的特别大

<br>
<br>
<br>

## PS：

- 对于句向量而言，使用预训练跟微调两阶段训练得到的模型，相较于传统的只进行微调的方法明显在性能有很好的提升
- **个人理解：**当模型达到一定程度上时，在训练方法还是采用对比学习时，已经很难在微调阶段取得更大突破，这也许是因为基底模型本身能力所限制，因为很多基底模型本身就是针对基础语言任务训练的，跟基于句子级别的表征生成任务本身是有区别的，所以如果能针对句子表征生成任务做针对性的预训练，就能进一步提高基底模型的上限，从而提升模型整体表现

<br>
<br>
<br>

## 参考资料
- 《Towards General Text Embeddings with Multi-stage Contrastive Learning》
- GTE(MTEB Top2)的成功秘诀--大力出奇迹 - 泽龙的文章 - 知乎
https://zhuanlan.zhihu.com/p/652472924