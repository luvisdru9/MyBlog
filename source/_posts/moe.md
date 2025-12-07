---
title: MoE —《OUTRAGEOUSLY LARGE NEURAL NETWORKS:THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER》论文阅读笔记
author: luvisdru9
date: 2025-09-04 00:00:00
updated: 2025-09-04 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: MoE —《OUTRAGEOUSLY LARGE NEURAL NETWORKS:THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER》论文阅读笔记
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

深度学习的成功依赖于两个因素：更大的模型与更多的数据。但是对于典型的深度学习模型而言，每一个训练样本都会激活整个模型参数，随着模型规模和数据规模的增大，这些计算量的增长是爆炸式的，看起来似乎令人不太可接受。

为了应对这种问题，研究人员提出了各类形式的“条件计算”，即让每个样本只激活模型的一部分参数，哪些部分被激活由门控机制决定。而目前为止，却还没有人在这些方法上面得到模型容量、训练时间或模型质量等层面的巨大提升，这主要涉及以下几个原因：

- GPU擅长于矩阵运算，不擅长“分支逻辑”，这类条件计算方案对于GPU的利用率大大降低；
- 条件计算方案使得不同样本的训练路径不同，没有办法进行大批量训练，会影响模型的性能与训练的成本；
- GPU的算力增长速度很快，由显卡型号的迭代可以看出这一点，但是GPU之间的通信带宽却增长十分缓慢，这会影响多卡场景下的模型训练。比如：当token进行embedding时，它放置在GPU0上进行forward，但是如果整个词表非常大的话，embedding权重可能会被切分至不同的GPU进行保存，那么该token明明只需要进行一次很简单的计算，却可能需要跨卡间进行通信调取对应位置的权重，导致模型计算效率大打折扣；
- 在条件计算里，并不是天然就能保证“每次只激活少量子网络，而且还分布均匀”，如果不控制，门控机制可能会出现极端情况：如某些门控被几乎所有输入都选中，导致计算压力集中在少数子网络上义，这类问题无疑会影响模型的质量。所以训练时需要在损失函数里加入额外的正则化项，强制约束门控的分布，让它每个输入只激活很少几个子网络，且不同子网络都能分到差不多的工作量；
- 对于一些亿级参数的模型而言，之前的工作都是在一些几十万的小数据集上进行训练，似乎并不太支持将这样的大模型训练好。

这篇文章提出一种稀疏门控混合专家层（Sparsely-Gated Mixture-of-Experts Layer, MoE），MoE由多个“Expert”组成，每一个Expert都是一个简单的FFN，同时有一个可训练的门控网络，用来选择稀疏的Expert组合来处理每一个输入，网络的所有部分都能够通过BP进行训练。文章中重点关注语言建模和机器翻译任务，这些任务更倾向于从一些超大规模模型上得到更好的表现。

<div align="center">
    <img src="img_1.png"/>
</div>

由上图可知，MoE层由一组$n$个Expert网络$E_1...E_n$以及一个门控网络$G$组成，门控网络输出为一个稀疏的$n$维向量，那么对于一个输入$x$而言，MoE层的输出可以表示为：$y=\sum_{i=1}^nG(x)_iE_i(x)$。

当$G(x)_i=0$时，就不需要计算$E_i(x)$。在文章实验中，Expert数量可以达到上千个，但是每个样本只需要分配少量的Expert，如果Expert数量实在非常庞大，我们也可以通过两层MoE进行降低分支参数（branching factor），即选择一个Expert组合，其中每个Expert也是一个MoE。



<br>
<br>
<br>

## 一些门控机制

### Softmax Gating

最基础的一种MoE Gating方法，$x$经过线性层后直接进行Softmax，这样会为每个Expert都分配一个权重，但是是非稀疏的，没有节省计算量：$G(x)=Softmax(xW_g)$。



<br>

### Noisy Top-K Gating

文章在Softmax Gating中引入了两个点：稀疏性和噪声。首先，在应用Softmax之前，加上可学习的高斯噪声：$H(x)_i=(xW_g)_i+StandardNormal()\cdot Softplus((xW_{noise})_i)$，其中$Softplus(x)=log(1+e^x)$，这个函数类似于一个平滑版本的ReLU，呈现单调递增且永远为正的特性，可以保证训练出来的噪声方差始终为非负，用于控制噪声的“强度”。

而后，选择Top-K个Expert，其他的设置为$- \infty$：$KeepTopK(v,k)_i=\left\{\begin{matrix}v_i \\- \infty \end{matrix}\right.$。最后在Softmax后，选择部分成功激活，而其他部分则为0：$G(x)=Softmax(KeepTopK(H(x),k)$。


<br>
<br>
<br>

## Training

这里与其他部分一起，使用BP来训练门控网络，在BP时少量Expert的梯度非零，换个说法，模型梯度在大部分都不是很敏感。这种机制与带噪声的ReLU函数相似（Noisy ReLU），ReLU小于等于0时不激活，大于0时则激活，Noisy ReLU通过在小于等于0部分引入噪声，使得这一部分也有机会进行梯度回流，与此处门控网络加入噪声，使得部分Expert有机会跻身于Top-K行列，引入一些随机性。


### Shrinking Batch

前文提到过，现代的GPU，使用large batch可以平摊掉参数加载和更新的计算时间开销。对于MoE而言，如果每个样本从$n$个Expert中选择$k$个，那么每一个Expert大约只会接收到$\frac{kb}{n}$个样本，也就是说，每一个Expert拿到的batch size实际上很小，文章中提出了以下几种解决方法：

- 引入模型并行与数据并行混合机制，假设有$d$台设备，每台设备拷贝一份相同的模型，且将一个batch size为$b$的数据集均匀划分到不同设备进行训练（每台设备的数据集不相同），Expert总数目为$n$，每个样本被分到$k$个Expert，这样平均每个Expert被分到的样本数为$\frac{k\cdot b\cdot d}{n}$，比单台设备时多了$d$倍（相当于对于每一个样本而言，还是那个模型，还是那么些个Expert，只不过有还有多个设备在同步跑其他几个样本，整体速度加快了，意味着单位时间每一个Expert平均的吞吐量也提高了）；

- 分层MoE：每个设备都有一份“主Gating Network”和一份自己独有的“次级MoE”，每个设备都有完整的大小为$b$的数据集，那么这么看的话，整体的total batch size其实为$b \times d$，是随着设备数的增大而增大，因此每一个Expert单位时间吞吐的样本数不变，还是$\frac{kb}{n}$，但是可以增加Expert的数量，提高模型性能；

- 在语言模型中，如果有一个句子序列，对每一个时间步的token都进行MoE的话，利用率很低，我们可以利用CNN的权重共享性质，直接一次性将所有token输入MoE中，也就是相当于原本为batch size=1调用n次，现在是batch size为n调用1次；

- 部分研究人员把MoE嵌入到循环结构里面，这样每个时间步的MoE输入不止依赖于当前输入，也依赖于上一步MoE的输出，这样可以在每个时间步选择不同的Expert且考虑历史信息，但是这样必须依赖上个时间步的信息，不能像第三种方法一样一次性处理所有数据了。在这个方法上，我们想要提高batch size，会受到类RNN结构的限制，BPTT（Backprop Through Time）在大batch size、长时间序列的展开上可能会出现存储激活值多，内存爆炸现象。Gruslys等人则提出了一个方案，不存储每个时间步的激活，只存储关键时间步的激活，在反向传播时，如果需要某个未存储的激活，就重新前向计算得到它，从而用一些重复的计算来节省显存。



<br>
<br>
<br>

## 分布式计算

在分布式计算中，Expert本身是“固定的”，只是其输入和输出需要再多设备之间通信，这会导致一个问题：如果每个Expert的计算量很小，但是输入输出很大，也就是说单个GPU的计算很快，大部分时间都在等待卡间数据传输，造成效率低下。如果要保持计算效率，则``每个Expert的计算量/其输入输出数据的大小``要大于``计算设备的计算能力/网络带宽``，这样每一个设备都是满负荷充分利用。在文中使用的MoE中，每一个Expert都只有一层hidden layer，那么我们可以单纯地增大hidden layer或者添加多层hidden layer。



<br>
<br>
<br>

## Expert的重要性偏好

作者观察到，门控网络很容易收敛到一种状态：给少数几个Expert分配很大的权重。这是模型整体学习得到的自我强化，因为被频繁选中的Expert被训练得越来越强，导致门控网络进一步依赖这些Expert，进入一个恶性循环。Eigen等人使用$硬约束$方法，强行在训练初期使用每一个Expert；Bengio等人则使用$软约束$，在损失函数中添加正则化，鼓励门控网络输出更均衡的门控值。

在本文中，作者也使用了一种软约束的方法。首先需要设置一个定义：一个Expert的$重要性$，等于这个Expert在一个batch中所有门控值的总和：$Importance(E_i)=\sum_{batch}G(x)_i$，最后添加一个关于Expert重要性的损失项：$L_{importance}=w_{importance} \cdot CV(Importance(E_1), ..., Importance(E_i))^2$，其中$CV$为变异系数，其值为``标准差/平均值``，$CV$越大则Expert之间差异越大。最小化这个损失项则可以迫使模型中不同Expert的重要性趋于一致。


<br>
<br>
<br>

## 一些实验

作者也开展了相关实验，使用的baseline为Google在16年提出的GNMT，作者减少了原始模型的LSTM层数，同时在encoder之间插入了一些MoE层。


### 实验一

在“1 BILLION WORD LANGUAGE MODELING BENCHMARK”上进行对比实验，MoE架构通过引入E多个Expert，提高了模型规模，且在perplexity上远低于baseline。而在相同计算量下，MoE的perplexity也远低于baseline

<div align="center">
    <img src="img_2.png"/>
</div>

<br>

### 实验二

在 1B 数据集上，当MoE参数超过大约1B后，提升速率开始变小。而在 100B 数据集上时，perplexity的优化在模型参数量68B之前还在显著改善，但是在137B时却反而恶化了，这可能是过度的稀疏导致的。不管怎么样，下图中两条曲线逐渐增大的差距验证了一个观点：在更大的数据集上，更大的模型规模能够更好地进行学习（无处不在的Scaling Law）

<div align="center">
    <img src="img_2.png"/>
</div>

<br>
<br>
<br>

## 参考资料

- [《OUTRAGEOUSLY LARGE NEURAL NETWORKS:THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER》](https://arxiv.org/abs/1701.06538)