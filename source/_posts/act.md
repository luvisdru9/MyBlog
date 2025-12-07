---
title: LLM中的激活函数
author: luvisdru9
date: 2025-09-11 00:00:00
updated: 2025-09-11 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: LLM中的激活函数
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

## ReLU(Rectified Linear Unit)

<div align=center>
	<img src="img_1.png"/>
</div>

表达式：$ReLU(x)=max(0,x)$。

在最早期的一些神经网络和小型NLP模型中，ReLU是出现频次最高的激活函数，但是在Transformer出现后，ReLU的一些问题也随之被放大，在$x<0$时，ReLU将梯度完全截断，在大规模的神经网络中，这很可能导致一些神经元长期得不到更新，出现“dead neuron”的风险。


<br>
<br>
<br>

## GELU(Gaussian Error Linear Unit)

<div align=center>
	<img src="img_2.png"/>
</div>


Hendrycks等人在2016年提出了一种名为GELU的激活函数，其表达式为：$GELU(x)=x \cdot \phi(x)$，其中$\phi(x)$为标准正态分布的CDF(cumulative distribution function)，可以写为$\phi(x)= \frac{1}{2 \pi} \int_{- \infty}^{x} e^{- \frac{t^2}{2}}dt$。

在实际应用中，为了便于实现，PyTorch、Tensorflow等框架都使用了近似计算来表示GELU，原文中也有相关叙述：$GELU(x)=0.5x(1+tanh[\sqrt{2/\pi(x+0.044715x^3)}])$。

GELU可以理解为带有概率的“Soft Gate”ReLU，输入$x$被保留的概率为$\phi(x)$，是一个平滑的激活过程：

- $x=0$时，$\phi(x)=0.5$，激活结果仍然是0，这一点似乎与ReLU结果相等，但是$\phi(x)=0.5$传达了一个信号：GELU是平滑的，他保持一个“确信度”，在零点处，这个确信度为0.5；

- $x \gg 0$时，$\phi(x) \to 1$，对于较大的正数而言，GELU“确信度”较高，几乎保留了整个原始的x，在x近乎于无穷大时，GELU的激活值接近于ReLU的激活值；

- $x \ll 0$时，$\phi(x) \to 0$，对于较大的负数而言，GELU“确信度”较低，非常小的保留概率使得激活值接近于0，但是与ReLU不同的是，任何负数在GELU的激活后都会保留一些负信号，而不是直接清空。

GELU这种平滑且连续可导的激活函数在实际应用中优化更稳定，且能很好地避免“dead neuron”的问题，BERT、GPT-2等模型中便使用了GELU作为激活函数方案。



<br>
<br>
<br>

## GLU(Gated Linear Unit)

Dauphin等人在2017年提出了GLU，GLU利用可学习的门控机制进行信息控制，其表达式为：$GLU(x)=(xW+b) \otimes \sigma(xV+c)$，其中$W$和$V$都是可训练的权重，$\otimes$为逐元素乘法。

根据公式分析，GLU相当于就是对输入作线性变化，而后使用sigmoid函数作为门控控制输出，通过可学习的权重与模型一起学习如何对信息进行更优的控制。

GLU最早用于CNN，而后以一些变体的形式出现在一些大模型当中，下面介绍主要的三个变体：GeGLU、ReGLU和SwiGLU，这些变体在Switch Transformer、T5、PaLM、LLaMA等模型架构中都展现了非常好的性能。


<br>
<br>
<br>

## ReGLU(Gated GELU)

ReGLU在GLU的基础上，将sigmoid函数换成了ReLU函数，其表达式为：$ReGLU(x)=(xW+b) \otimes ReLU(xV+c)$。

sigmoid函数作为门控而言，容易进入梯度饱和区而导致出现梯度消失的情况；而ReLU虽然会产生“dead neuron”情况，但是部分神经元仍旧梯度稳定，相比于sigmoid更适用于深层网络，且这种强制截断导向的“稀疏性”结合门控机制也许能够带来更好的表达与正则化。



<br>
<br>
<br>

## GeGLU(Gated GELU)

GeGLU类似于ReGLU，但是用GELU代替了ReLU，其表达式为：$GeGLU(x)=(xW+b) \otimes GELU(xV+c)$。

结合上述GELU的特性来看，GELU相当于结合了sigmoid和ReLU的优点，在保持平滑性的基础上，软约束地引入了稀疏性。

<br>
<br>
<br>

## SwiGLU(Swish GLU)

22年，Google的PaLM中首次使用了SwiGLU，这个激活函数结合了GLU以及Google之前的工作Swish，其表达式为：$SwiGLU(x)=(xW+b) \otimes Swish(xV+c)$，下面先来介绍一下Swish函数。

Swish函数的表达式为：$Swish(x)=x \cdot \sigma(\beta x)$，当$\beta$取不同值的时候，Swish的特性曲线也不相同：

<div align=center>
	<img src="img_3.png"/>
</div>

可见Swish函数是平滑的，$\beta \to 0$时，Swish函数接近于线性函数$f(x)=\frac{x}{2}$；当$\beta \to \infty$时，Swish函数几乎变为了ReLU。当$\beta =1$时，$Swish(x)=x \cdot \sigma(x)$，这时候我们将Swish函数称为SiLU(Sigmoid Linear Unit)函数，非常直观，就是一个Linear乘上一个Sigmoid。

SwiGLU在负区间有小梯度，不会像ReLU一样出现“dead neuron”；而对比GeGLU，SwiGLU中的Swish函数计算效率相比于GELU中的$\phi(x)$而言更高，更适合大规模训练。
