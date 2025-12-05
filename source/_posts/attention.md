---
title: Self-attetion & Cross-attetion
author: luvisdru9
date: 2025-07-28 23:00:00
updated: 2025-07-28 23:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML

categories: NLP
description: Self-attetion & Cross-attetion
keywords:
  - NLP
  - LLM
  - DL
  - ML
#top_img:
#comments:
cover: img_3.png
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

下面简单记录一下Self-attention和Cross-attention。

## Self-Attention

Scaled Dot-Product Attention（缩放点积注意力）演示图如下：

<div align=center>
	<img src="img.png"/>
</div>

而Self-Attention允许模型在处理一个输入序列时，关注序列内部的每个元素之间的关系。每个元素既作为查询（Query），又作为键（Key）和值（Value），通过计算自身与其他元素的相关性来更新表示：

<div align=center>
	<img src="img_1.png"/>
</div>


<br>
<br>
<br>

## Cross-Attention
Cross-Attention用于建模两个不同序列之间的关系。一个序列提供查询（Query），另一个序列提供键（Key）和值（Value），它通常用于需要融合来自不同数据源或模态的信息的任务。

在 Transformer Decoder中，查询来自目标语言序列，键-值来自源语言序列（如将“Je t’aime”翻译为“I love you”时对齐“aime”和“love”）。

<div align=center>
	<img src="img_2.png"/>
</div>

<br>
<br>
<br>

## 参考资料

- https://blog.csdn.net/qq_41990294/article/details/147746522