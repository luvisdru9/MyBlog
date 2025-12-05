---
title: LLM常见的位置编码
author: luvisdru9
date: 2025-07-25 17:00:00
updated: 2025-07-25 17:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: LLM常见的位置编码
keywords:
  - NLP
  - LLM
  - DL
  - ML
#top_img:
#comments:
cover: img_6.png
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

## 为什么需要位置编码？

大模型在处理序列数据时，其内部的注意力机制是内容驱动的，只关注token的相似度，无法感知到token的位置信息，因此需要引入额外的信息来表达token所在的空间信息。


<br>
<br>
<br>

## sinusoidal位置编码

<div align=center>
	<img src="img.png"/>
</div>

sinusoidal位置编码基于embedding的维度``d``和其维度中具体的索引``i``以及token的位置``pos``进行编码。具体的，同时使用sin和cos的值，这样能够避免周期导致的值相等混淆了模型对于位置的理解。编码反映了token之间的相对关系而不是绝对关系，编码的差值与token位置的差值有关，且相邻的token编码值也相差较小：

<div align=center>
	<img src="img_1.png"/>
</div>

这里由公式可以得知，对于不同索引的维度，维度索引越大，频率越大，周期越短。对于低频情况，周期较长，同一个句子中的首尾token位置编码可能在这种周期下呈现较小的位置差异，这能够体现偏全局的位置信息，如``这两个token处于同一个句子``的信息；而高频周期较短的情况下，变化更为剧烈，值变化相对较大，对于相邻位置之间的微小差异更为敏感非常适合捕捉“token紧邻的顺序信息”。

<br>
<br>
<br>

## 旋转位置编码（RoPE，Rotary Position Embedding）

sinusoidal编码对于相对位置的建模比较间接，而RoPE则是一种更为直接的相对位置建模方法。传统的方式是将编码直接加到token embedding上，这样在计算注意力分数时仍然是依赖于两个token各自的位置， 并不会引入两个位置的差值。

<div align=center>
	<img src="img_2.png"/>
</div>

而RoPE则是将位置信息编码进注意力机制中Query和Key向量的**角度**中，用复数旋转或2D旋转的方式，引入“相对位移”信息，最后的注意力分数与位置的差值相关：

<div align=center>
	<img src="img_3.png"/>
</div>

RoPE相比于传统的正余弦编码，具有更强的外推能力。传统的正余弦编码每个词都有一个基于「第几位」的编码，比如第 1 位、第 99999 位，如果模型只见过最多第 1000 位，那么遇到第 99999 位时就会很懵逼，注意力分数变得不稳定，其他token“不知道”是不是该关注这个位置的内容。而RoPE只关注位置差值，一定程度上增加了模型的外推泛化能力。

PS：个人疑问，如果说正余弦编码中，模型是因为无法看到没有出现的位置的token而导致注意力分数计算出现偏移，那使用RoPE也会出现差值范围的不一致，是否也会导致模型出现一定的分数计算偏移呢？（当然整体来说RoPE外推能力肯定更强）

<br>
<br>
<br>

## ALiBi（Attention with Linear Biases）

上述内容提到，正余弦编码方式下外推能力比较弱。即使旋转位置编码比正弦方法有所改进，但仍未达到令人满意的结果。

ALiBi不显式修改embedding，而是直接在Attention Score上加「线性偏置」，让模型天然关注距离近的token，同时根据m每一个注意力的head会乘上一个预设好的斜率项(Slope)：

<div align=center>
	<img src="img_4.png"/>
</div>

<div align=center>
	<img src="img_5.png"/>
</div>

也就是说这种编码方法不像之前两种方法一样涉及模型的理解问题，而是简单在已有的分数上进行惩罚。

<br>
<br>
<br>

## 可学习位置编码

直接将不同token的不同维度的enbedding的位置编码当成可学习的参数（大力出奇迹）。