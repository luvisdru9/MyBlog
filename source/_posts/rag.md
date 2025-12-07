---
title: RAG -《Retrieval-Augmented Generation for  Knowledge-Intensive NLP Tasks》论文阅读笔记
author: luvisdru9
date: 2025-09-26 00:00:00
updated: 2025-09-26 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: RAG -《Retrieval-Augmented Generation for  Knowledge-Intensive NLP Tasks》论文阅读笔记
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

传统的参数化模型如GPT、BERT，这些模型训练完成后，知识就锁死在模型参数中，难以更新。且由于模型的黑箱性质，无法提供预测的依据或来源，并且非常容易产生幻觉，容易编造看似合理但虚假的信息。


Facebook为了解决上述提到的问题，提出了Retrieval-Augmented Generation (RAG)，引入了非参数化记忆，也可以称为“外部知识库”，构建了一种混合范式。好处为：外部知识库可以随时拓展、可以“检查”模型到底参考了哪些源文档以及模型可以基于事实依据生成，减少幻觉。

与之前的REALM, ORQA等“抽取式问答”相比，RAG则将检索增强应用到了nlp的seq2seq任务中。


RAG的Workflow大致可以简述为下：给定一个输入序列$x$，并且检索文本文档$z$，并在生成目标序列$y$时将这些检索出来内容作为附加文本进行使用。 整个架构包含两个关键组件：

1. 一个参数为$\eta$的检索器$p_{\eta}(z|x)$：给定一个query $x$，这个检索器能够返回所有可能文档的概率（实际应用中可能返回Top-K个） ；
2. 一个参数为$\theta$的生成器$p_{\theta}(y_i|x,z,y_{1:i-1})$：给定初始的输入$x$、前$i-1$个token以及检索到的内容$z$，以生成一个当前的token。具体如下图所示：

<div align=center>
	<img src="img_1.png"/>
</div>

<br>
<br>
<br>

## Models

文中使用seq2seq的方案对检索器和生成器进行联合训练，并针对RAG的具体模型架构提出了两种方案，分别在sequence层级和token层级上利用检索：

### RAG-Sequence Model

让检索器先根据初始输入$x$检索得到Top-K个文档检索内容$z$，生成器根据这$K$个不同的文档内容生成$K$个不同的连续序列概率，将这些序列概率求和后得到总目标生成连续序列的概率，表达式如下：

<div align=center>
	<img src="img_2.png"/>
</div>

<br>

### RAG-Token Model

对于当前生成过程中的token $y_i$，我们先得到Top-K个不同的文档内容下输出该token的概率，然后求和得到总的该token的概率，最后按照这个过程重复生成token得到最后的生成序列$y$，表达式如下：

<div align=center>
	<img src="img_3.png"/>
</div>

<br>

文中还额外提到，如果将RAG用于序列分类的话，则最后都是依赖于某个token，此时两种方案退化为同一层级，使用哪一种效果都相同。


<br>

### 检索器

使用了DPR，DPR为双编码架构，分为文档编码器和查询编码器：

$$d(z)=BERT_d(z)$$

$$q(x)=BERT_q(x)$$

$$p_{\eta}(z|x) \propto exp(q(x)^Td(z))$$

计算Top-K$(p_{\eta}(\cdot | x))$是一个最大内积搜索问题（Maximum Inner Product Search, MIPS)，参考这篇文章：[《Billion-scale similarity search with gpus》](https://arxiv.org/abs/1702.08734)，FaceBook团队使用了FAISS（Facebook AI Similarity Search），这会将整个知识库的文档通过DPR的文档编码器转换为向量后，预先构建一个FAISS索引，而后使用优化后的倒排文件索引（IVF）进行近似搜索，实现亚线性时间的检索。



### 生成器

使用BART-large作为生成器，参数量为0.4B，文中将检索结果$z$和原始输入$x$简单进行拼接。

<br>
<br>
<br>


## Training

对检索器和生成器进行联合微调训练，语料数据格式只给出了输入和输出：$(x_i,y_i)$，而中间过程需要检索哪些内容没有进行规定，这将迫使模型进行无监督检索的学习。

更具体而言，基于Adam优化器和随机梯度下降对负边际对数似然进行优化：$\sum_j -log(y_j|x_j)$。

文中提到，在训练期间更新文档编码器的参数$BERT_d$计算开销很大，需要定期更新知识库文档索引，因此不对文档编码器进行更新，只更新查询编码器$BERT_q$和生成器$BART$。

<br>
<br>
<br>

## Decoding

在decode时，对于RAG-Token而言，它是一个标准的自回归seq2seq生成问题，且每一步选出token都需要给定这一步的Top-K个检索结果，根据这种特性，我们可以将其带入到一个标准的Beam Search中（非常自然的过程，每生成一个token都需要保留K个最高的概率结果，标准的Beam Search过程）。

而对于RAG-Sequence而言，整个过程无法自然地代入一个标准的Beam Search。如果需要彻底解码，则需要进行如下过程：

1. Top-K检索到$K$个文档结果，对于每一个文档结果，都进行一次Beam Search，得到$N$个在Beam Search策略下最有可能的答案序列；

2. 将$K$个检索文档结果的所有答案序列进行合并，得到全局候选池$Y$，而后则需要计算$Y$中每个答案序列$y$的“全局累积概率”$p(y|x)$（注意：这里全局比较时并不是直接拿基于每一个文档生成序列的局部累积概率来比较，局部累积概率只能用于同一个文档下生成的序列的比较）；

3. 根据公式：$\sum_{z \in TopK(p(\cdot | x))}p_{\eta}(z|x) \cdot p_{\theta}(y | x, z)$，我们可以计算全局累积概率，这要求对所有Top-K的$z$进行求和，但是这会出现一个问题：如果一个候选答案序列$y$没有出现在某个文档$z'$Beam Search的$N$个答案序列中，也就是我们缺少了这一部分的$p_{\theta}(y | x, z')$，而为了精确计算全局累积概率，需要将这一部分补上，则需要给定这个文档$z'$、原始输入$x$和已知的答案序列$y$，重新进行一次前向推理，得到$p_{\theta}(y | x, z')$。


这种彻底解码有可能会非常耗时，首先初始就需要进行$K \cdot N$次Beam Search，而最坏的情况，全局候选池$Y$中每一个$y$都在唯一的文档$z$下生成，那么这时候需要进行额外的$(K-1) \cdot |Y|$次额外的前向传播，对于长文本而言，这显然不太现实。

文中提出的“快速解码”是一种为了实际应用而做出的近似策略，即如果某个$y$没有在$z'$下的生成结果中，则令$p_{\theta}(y | x, z')=0$。这个假设的合理性在于：如果连在只看文档$z'$的情况下，$y$都不是前$N$个最可能的答案之一，那么它对于最终全局累积概率的贡献很可能微乎其微，可以忽略不计。


<br>
<br>
<br>

## 参考资料

- [《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》](https://arxiv.org/abs/2005.11401)