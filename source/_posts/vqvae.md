---
title: VQ-VAE -《Neural Discrete Representation Learning》
author: luvisdru9
#date: 2025-07-25 00:00:00
#updated: 2025-07-25 00:00:00
tags: 
  - CV
  - DL
  - ML
categories: CV
description: 《Neural Discrete Representation Learning》阅读笔记
keywords:
  - CV
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

## 传统连续潜变量VAE的后验坍塌

在传统的连续潜变量VAE模型中，其ELBO(the Evidence Lower Bound)可以大致建模为：

$$\begin{aligned}
\log p(x) &= \log \int_{z} p(x)q(z|x)dz = \mathbb{E}{q(z|x)}[\log p(x)] \\
&= \mathbb{E}{q(z|x)}\left[\log \frac{p(x, z)}{p(z|x)}\right] = \mathbb{E}{q(z|x)}\left[\log \frac{ {\color{red}q}(z|x)p(x, z)}{p(z|x){\color{red}q}(z|x)}\right] \\
&= \mathbb{E}{q(z|x)}[\log p(x, z) - \log q(z|x)] + D_{KL}(q(z|x)||p(z|x)) \\
&\ge \mathbb{E}{q(z|x)}[\log p(x, z) - \log q(z|x)] \\
&:= ELBO \\
&= \mathbb{E}{q(z|x)}[\log p(z) + \log p(x|z) - \log q(z|x)] \\
&= \underbrace{\mathbb{E}{q(z|x)}[\log p(x|z)]}{\text{Reconstruct term } L_{Rec} } - \underbrace{D_{KL}(q(z|x)||p(z))}{\text{KL term } L{KL}}
\end{aligned}$$

这类VAE与一些强大的自回归解码器结合使用时，在训练过程中，这个能力强大的解码器很有可能忽略编码器产生的后验分布$q_\theta(z|x)$，导致$q_\theta(z|x)$成为了无意义的噪声，我们称为“后验坍塌”(Posterior Collapse)，也称为KL散度消失(KL-vanishing)。这辉导致整个生成过程不再包含任何来自源数据$x$分布的信息，于是VAE自此失效。

<br>
<br>
<br>

## 传统离散VAE的“梯度方差过大”

对于离散的VAE而言，其训练过程会遇到**梯度方差过大**的问题。这是因为我们需要从一个概率分布中**抽取**一个离散的类别，抽样这个动作，以及离散的阶跃变化，在数学上是不可导的。

为了对离散VAE进行训练，许多研究者提出了不同的方法，文中提到[NVIL 估计器](https://arxiv.org/abs/1402.0030)、[VIMCO](https://arxiv.org/abs/1602.06725)等不同的方法，但是这些方法都没能弥合与连续潜变量VAE的性能差距，因为连续 VAE 可以使用高斯重参数化技巧，其梯度方差要低得多。

VQ-VAE采用了一种新的训练方式，后验分布和先验分布都是categorical，从这些分布中抽取的样本将作为索引去检索一个嵌入表，然后，这些嵌入向量被用作解码器网络的输入。
<br>
<br>
<br>

## VQ-VAE

### Discrete Latent variables

<div align=center>
	<img src="img_1.png"/>
</div>

VQ-VAE定义了一个潜在嵌入空间(这里我们又称为字典)$e \in R^{K \times D}$，其中 $K$ 是离散潜变量空间的大小（即 $K$ 维分类），$D$ 是每个潜变量嵌入向量 $e_i$ 的维度。模型接收输入 $x$，通过编码器产生连续输出 $z_e(x)$。然后，通过在共享的嵌入空间 $e$ 中进行最近邻查找（如公式 1 所示）来计算离散潜变量 $z$。解码器的输入就是对应的嵌入向量 $e_k$（如公式 2 所示）。

$$q(z = k|x) = \begin{cases} 1 & \text{for } k = \text{argmin}_j ||z_e(x) - e_j||_2, \\ 0 & \text{otherwise} \end{cases} \quad (1)$$

$$z_q(x) = e_k, \quad \text{where} \quad k = \text{argmin}_j \|z_e(x) - e_j\|_2 \quad (2)$$

作者在文中提到，建议分布 $q(z=k|x)$ 是具有确定性的，并且将 $z$ 定义一个简单的均匀分布，从而得到了一个恒等于 $\log K$ 的 KL 散度常数，通过这种方式，将KL散度的值固定下来，解决了前文出现的“KL-vanishing”问题。

<br>

### Learning

可以注意到，公式2也像之前提到的离散VAE的抽取方式一样，是没有梯度的。VQ-VAE中采用了直通估计的思想(straight-through estimator, STE)，直接将解码器输入 $z_q(x)$ 的梯度复制给编码器输出 $z_e(x)$。

在前向传播过程中，将nearest embedding $z_q(x)$传给解码器；而在反向传播过程中，梯$\nabla_z L$保持不变地传给编码器，尽管是近似，由于它们共享同一个 $D$ 维空间，这个梯度包含了指导编码器降低重构损失的有用信息。这个梯度能够使得输入$x$在下一次前向传播进入编码器后以不同的方式被离散化，因为公式1此时的赋值会更新导致不同。

VQ-VAE整体的训练函数如下所示，总共包含三项：

$$L = \log p(x|z_q(x)) + ||\text{sg}[z_e(x)] - e||_2^2 + \beta ||z_e(x) - \text{sg}[e]||_2^2 \quad (3)$$

#### 重构损失 $\log p(x|z_q(x))$

这一项通过将$z_q(x)$的梯度直接复制给$z_e(x)$，跳过中间的索引操作，利用这个梯度对编码器和解码器进行训练。

#### 字典更新损失(VO loss) $||\text{sg}[z_e(x)] - e||_2^2$

因为在重构损失的计算中，跳过了字典索引，因此字典接收不到这部分损失，那我们如何更新字典呢？

这里作者使用了一种类似于k-means的思想，我们将编码器提取出来的连续特征$z_e(x)$看作散落在空间中的数据点，而字典中的离散向量$e_i$则是空间中的聚类中心。对于这个公式而言，$sg$代表stop gradient，即禁止这一项loss的梯度流向$z_e(x)$，只流向$e$。那么其含义就是：通过更新这个字典向量$e$，使其作为一个聚类中心能够更好地表示编码器经常输出特征的那些高密度的“特征簇”。

#### 承诺损失(Commitment Loss) $\beta ||z_e(x) - \text{sg}[e]||_2^2$

由前两项损失可以看出，编码器根据重构损失，为了于解码器一起把图像重建地更加好而不断地更新自身。而对于字典而言，他则需要根据编码器给出的嵌入表示进行更新。如果字典的训练速度跟不上编码器的话，可能编码器的嵌入空间更新后离开上一次的$e$非常远，这导致字典的最近邻查找完全失效，整个表征体系崩溃。

因此，为了解决这个问题，作者加入了Comitment Loss，形象地看，就是利用利用$sg$将之前的$e$作为一个“固定坐标”，告诉编码器，可以进行更新，但是要对原有的字典有一个commitment，更新范围基于$\beta ||z_e(x) - \text{sg}[e]||_2^2$进行限制，通过这种方法使得离散特征空间也因此得以保持稳定和紧凑。

<br>

### 生成

字典训练完毕后并且冻结，现在我们需要进行图像生成，但如果把这些离散编码随机拼凑喂给解码器，生成的只会是毫无意义的雪花噪点。我们需要学习这些离散向量的排列规律。对于图像生成来说，作者引入了一个PixelCNN，用大量真实的图像数据通过VQ-VAE进行压缩得到其离散向量集，根据对应的字典索引得到离散索引集$I$，在这个离散索引集$I$的基础上用自回归的方式训练pixelnet，使其拥有“根据上一个位置离散索引预测下一个位置离散索引”的能力。

让训练好的 PixelCNN 从左上角第一个位置开始，根据学到的规律，自回归地逐个预测下一个离散索引，直到填满整个特征图。假设这个特征图大小为$32 \times 32$，那么就有$32 \times 32$个预测得到的离散索引(除了第一个)。将这$32 \times 32$个预测得到的离散索引对应字典中的向量取出，然后通过解码器进行解码，最终得到生成的图片。

<br>
<br>
<br>

## 参考资料
- [《Neural Discrete Representation Learning》](https://arxiv.org/abs/1711.00937)


