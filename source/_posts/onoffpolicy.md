---
title: On Policy & Off Policy
author: luvisdru9
#date: 2025-10-22 00:00:00
#updated: 2025-10-22 00:00:00
tags: 
  - RL
  - DL
  - ML
categories: RL
description: On Policy & Off Policy
keywords:
  - RL
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

## On Policy

首先介绍On Policy的特性，这里以Policy Gradient为例，这是一种典型的On Policy方法，我们拿到Policy Gradient的梯度更新项：

$$\nabla \bar{R}_{\theta} = E_{\tau \sim p_{\theta}(\tau)} [R(\tau) \nabla \log p_{\theta}(\tau)]$$

我们需要在Policy $\pi_{\theta}$下采集数据（多条Trajectory $\tau$），然后对参数$\theta$进行更新。我们可以从梯度项中看到$\tau \sim p_{\theta}(\tau)$，也就是说这些Trajectory $\tau$都只能采样于Policy $\pi_{\theta}$中，当$\theta_t$被更新为$\theta_{t+1}$，就需要进行下一次的数据采集，旧的Policy下的数据就无法使用了。

这会导致使用Policy Gradient进行On Policy RL时，在某个Policy下收集大量的数据，更新一次后又需要继续收集，导致大部分时间都在采集数据而不是参数学习，这样的学习效率是相对较低的。


<br>
<br>
<br>

## Off Policy

清楚了On Policy的不足，我们希望从On Policy转为Off Policy能够实现如下目标：使用一个参数固定的Policy $\pi_{\theta'}$进行数据采集，将这些数据用于更新参数$\theta$，这样Policy $\pi_{\theta}$的更新就可以重复使用这些数据了。

那具体怎么做呢？首先先介绍一个概念：**Importance Sampling**。

### Importance Sampling

假设我们现在有一个函数$f(x)$，需要从分布$p$中采样$x$，然后求$f(x)$的期望$E_{x \sim p}[f(x)]$。如果我们没有办法对分布$p$计算期望的话，我们可以从$p$中采样一些样本$x_i$，然后近似计算得到：

$$E_{x \sim p}[f(x)] \approx \frac{1}{N} \sum_{i=1}^{N} f(x^i)$$

现在我们进一步考虑一个问题：如果我们没有办法从分布$p$中采样$x_i$，而是只能从另一个分布$q$采样$x_i$呢？这样我们就需要换一种方式计算这个期望，这里可以将原式展开推导：

$$E_{x \sim p}[f(x)] = \int f(x)p(x)dx = \int f(x) \frac{p(x)}{q(x)} q(x)dx = E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]$$

这样我们就可以将从$p$中采样$x$转为从$q$中采样$x$。将$E_{x \sim p}[f(x)]$与$E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]$对比我们可以发现，从$q$中采样计算的期望相比于从$p$中采样得到的期望而言乘上了一个权重$\frac{p(x)}{q(x)}$，用于修正两个分布之间的差异。

<br>

### Issue Of Importance Sampling
这个式子$E_{x \sim p}[f(x)] = E_{x \sim q} \left[ f(x) \frac{p(x)}{q(x)} \right]$看起来似乎对任意的$q$分布都成立，但是实际操作上，$p$和$q$最好不要差太多。

我们可以计算两者的方差：

$$Var_{x \sim p}[f(x)] = E_{x \sim p}[f(x)^2] - (E_{x \sim p}[f(x)])^2 \tag{1}$$

$$Var_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right] = E_{x \sim q}\left[\left(f(x) \frac{p(x)}{q(x)}\right)^2\right] - \left(E_{x \sim q}\left[f(x) \frac{p(x)}{q(x)}\right]\right)^2$$
$$= E_{x \sim p}\left[f(x)^2 \frac{p(x)}{q(x)}\right] - (E_{x \sim p}[f(x)])^2 \tag{2}$$

可以发现，后者的方差在$p$和$q$分布非常不一致时也会相差非常大，这就导致在**采样数量比较少**时，方差相差巨大会导致你根据采样数据近似的期望也会相差比较大，更形象的解释如下方所示：

下图是当$p$和$q$分布相差较大时，采样点较少的情况，可以看到如果基于$q$进行采样时，很容易全部采样在$q$概率较大的部分，比如绿色的点，这时候$f(x)$为正，算出的期望也是正；反之如果基于$p$进行采样，则很容易全部采样在$p$概率较大的部分，这时候$f(x)$为负，算出的期望也是负。两者算出的采样期望相差很大。

<div align="center">
    <img src="img_2.png"/>
</div>

当然，如果采样数量足够多的话也能够解决这种问题，如下图所示，基于$q$进行采样，采样数量足够多时在$q$概率较小的地方也能有一个采样点，此时$\frac{p(x)}{q(x)}$值非常大，会给这个点加上一个很大的权重，从而进行平衡。

<div align="center">
    <img src="img_1.png"/>
</div>

<br>

### On Policy -> Off Policy

根据Importance Sampling，我们可以将On Policy的期望计算：

$$\nabla \bar{R}_{\theta} = E_{\tau \sim p_{\theta}(\tau)} [R(\tau) \nabla \log p_{\theta}(\tau)]$$

变为符合Off Policy **“从另一个Policy分布采集数据”** 的期望计算形式：

$$\nabla \bar{R}_{\theta} = E_{\tau \sim p_{\theta'}(\tau)} [\frac{p_{\theta}(\tau)}{p_{\theta'}(\tau)} R(\tau) \nabla \log p_{\theta}(\tau)]$$

在上次Policy Gradient笔记的最后提到了，由于每个$(s_t,a_t)$我们会根据后续的情况赋上不同的奖励权重，将这种情况建模为一个优势函数$A^{\theta}$，真实的期望表达应该是：

$$E_{(s_t, a_t) \sim \pi_\theta} [A^\theta(s_t, a_t) \nabla \log p_\theta(a_t | s_t)]$$

一样的，我们也可以将其转为Off Policy的形式：

$$E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{P_\theta(s_t, a_t)}{P_{\theta'}(s_t, a_t)} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t | s_t) \right]$$
$$= E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_\theta(a_t | s_t)}{p_{\theta'}(a_t | s_t)} \frac{p_\theta(s_t)}{p_{\theta'}(s_t)} A^\theta(s_t, a_t) \nabla \log p_\theta(a_t | s_t) \right]$$

这里的$A^\theta(s_t, a_t)$代表着policy $\pi_\theta$和环境进行交互，但是现在换成了$\pi_{\theta'}$和环境做交互，所以应该换成$A^{\theta'}(s_t, a_t)$

**(存疑点)** 这里，对于$\frac{p_\theta(s_t)}{p_{\theta'}(s_t)}$这一项而言，其中的$p_{\theta}(s_t)$不太好计算，我们没有在policy$\pi_{\theta}中进行采样$。且前面也提及了$p$和$q$最好不要差太多，那么这里近似认为$\frac{p_\theta(s_t)}{p_{\theta'}(s_t)} = 1$，可以不进行考虑。

最后，我们参考公式$\nabla f(x) = f(x) \nabla \log f(x)$，可转化得到我们的目标函数：

$$J^{\theta'}(\theta) = E_{(s_t, a_t) \sim \pi_{\theta'}} \left[ \frac{p_\theta(a_t|s_t)}{p_{\theta'}(a_t|s_t)} A^{\theta'}(s_t, a_t) \right]$$

其中的$p_\theta(a_t|s_t)$和$p_{\theta'}(a_t|s_t)$可以很方便地进行计算，这就完成了On Policy到Off Policy的转换。

## 参考资料

- https://speech.ee.ntu.edu.tw/~tlkagk/courses_MLDS18.html