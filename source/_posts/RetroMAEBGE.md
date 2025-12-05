---
title: RetroMAE & BGE
author: luvisdru9
date: 2025-07-22 00:00:00
updated: 2025-07-22 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: RetroMAE & BGE
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

NLP任务中，大部分预训练模型都是基于token级别任务进行训练的，例如Seq2Seq范式与MLM范式。但是密集检索任务（比如RAG）更倾向于句子级别的表示，需要捕捉句子的信息和之间的关系，一般主流的策略是**自我对比学习**和**基于自动编码**。
对比学习会受到增强的数据质量的限制，需要大量的负样本（hard negative samples)。 
基于自动编码的方法不受数据增强和负样本采样的限制，文章中认为下面两个因素会影响基于自动编码方法的性能：
- 重建任务必须对编码质量有足够的要求
- 预训练数据需要被充分利用

文章没有从数据增强（构造正样本/负样本）出发，而是通过重构sentence embedding来增强表征效果。

<br>
<br>
<br>


## Encoder & Decoder

<div align=center>
	<img src="img.png"/>
</div>

### Encoding

类似于Bert，给定一个句子输入X，随机mask掉其中一小部分token后得到Input。通常会采用中等的mask比例（15%～30%），比一般的MLM的任务稍高一些，从而能保留句子原本大部分的信息。然后利用一个编码器对输入进行编码，得到对应的的句向量。文章中采用了类似BERT的编码器，最终将``[CLS]``位置输出作为句子向量。

<br>

### Decoding

同样给定该句子输入X，随机mask掉其中一部分token得到输入。文章中此处采取比encoder部分更加激进的mask比例（50%～70%），利用mask后的文本以及encoder生成的句向量对文本进行重建。


因为Decoding环节mask掉了大部分的输入Token，这强迫重建任务要从Encoding得到的句向量中获取大量的信息进行学习，从而迫使Encoding学习得到一个高质量的句向量。


<br>
<br>
<br>

## 预训练流程

文章认为，上述的Decoding方法存在某些不足之处，主要分为两处：
- 解码过程中，交叉熵损失函数只能从被mask掉的token中得到
- 每次进行重建训练时，都是基于相同的上下文进行
针对上述现象，文章提出了以下两种策略进行优化：

<div align=center>
	<img src="img_1.png"/>
</div>

### 双流自我注意力
文中提出一种双流自我注意力，输入部分会生成两个流``H1``和``H2``，``H1``是sentence embedding+Position embedding，``H2``为sentence embedding和token embedding+position embedding，得到H1和H2后，利用H1作为Q，H2作为K、V，H1中的每一个embedding会去H2中寻找最重要的上下文。

<div align=center>
	<img src="img_2.png"/>
</div>

<br>

### 特定位置注意力掩码的增强解码

引入了特定位置注意力掩码的增强解码，这样能够随机规定哪些token能被第1个token看到(这里保证自身看不到自身)，引入了一定随机性，最后输出的结果与H1一起（残差连接）一起过了layernorm和FFN之后用于重建原始输入（这里是全部的输入）。

<div align=center>
	<img src="img_3.png"/>
</div>


交叉熵损失函数可写做：

<div align=center>
	<img src="img_4.png"/>
</div>

文章基于预训练的Bert参数进行训练，在retrieval任务上进行zero-shot测试，以及基于DPR、ANCE微调后的有监督性能测试。

<br>
<br>
<br>

## BGE


BGE为北京智源人工智能实验室(BAAI)提出的文本嵌入模型，采用RetroMAE的预训练方法，基于Wudao数据集进行预训练，收集大量中文pair形式的通用文本数据，利用text2vec-chinese计算文本相似度后，选取一定阈值以上的数据对，对模型进行对比学习微调。

在特定任务中，不同的任务可能导致pair的相似度变化较大，争对这个问题，提出了两点解决方案：
- Instruction tuning；2.进行难负例挖掘；最后再在进行改造后的数据集上进行finetune；
- 将unlabeled的pair与领域监督数据集结合，涵盖retrieval、ranking、similarity comparison等任务，构建了数据集C-MTP。

<br>
<br>
<br>



## 参考资料

- 《RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder》
- 《C-Pack: Packed Resources For General Chinese Embeddings》
- https://zhuanlan.zhihu.com/p/649049846