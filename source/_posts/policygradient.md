---
title: Policy Gradient 学习笔记
author: luvisdru9
#date: 2025-10-22 00:00:00
#updated: 2025-10-22 00:00:00
tags: 
  - RL
  - DL
  - ML
categories: NLP
description: Policy Gradient
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


## Basic Components In Policy Gradient

对于一个智能体，我们对其进行Policy Gradient的RL。

在介绍Policy Gradient之前，引出其中涉及的几个组成部分：

- 无法更改部分：
  - Environment：智能体与之交互的环境；
  - Reward Function：奖励函数决定了智能体采取动作所获得的反馈。

- 待训练部分：
  - Actor：进行决策的智能体。

我们将Actor的决策建模为一个**Policy Of Actor**，模型用$\pi$表示，其参数为$\theta$。这个Policy能够给定一个当前状态的观测值(state)，输出一个最有可能采取的动作(action)。例如下图中，给定一个游戏画面的观测值，输出下一步要进行的操作。

<div align="center">
    <img src="img_1.png"/>
</div>

<br>
<br>
<br>

## 问题阐述

那根据上述的组件，我们如何建模这个智能体在环境中的强化学习问题呢？

对于某个时刻$t$来说，Actor根据Environment当前的观测值 \ 状态 $s_t$，得到action空间中不同action的概率，根据这些概率采样一个action $a_t$执行，而后Environment进入下一个观测状态值，重复这个交互过程，直到达到轮次上限$T$。大致过程如下图所示：

<div align="center">
    <img src="img_2.png"/>
</div>

我们将这过程中产生的所有${s_t, a_t}$数据收集在一起，称为一个**Trajectory**：

$$Trajectory \tau = {s_1, a_1, s_2, a_2, ..., s_T, a_T}$$

给定一个初始观测状态$s_1$，一个Actor通过其Policy决策得到某个$\tau$的概率可建模为：

$$p_{\theta}(\tau) = p(s_1)p_{\theta}(a_1|s_1)p(s_2|s_1, a_1)p_{\theta}(a_2|s_2)p(s_3|s_2, a_2) \cdots = p(s_1) \prod_{t=1}^{T} p_{\theta}(a_t|s_t)p(s_{t+1}|s_t, a_t)$$

在Actor基于某个状态$s_t$采取了某个action $a_t$后，可以由reward function得到此时的即时奖励$r_t$，具体如下图所示，那么Trajectory $\tau$的总奖励可以计算为：$R(\tau) = \sum_{t=1}^{T} r_t$。

<div align="center">
    <img src="img_3.png"/>
</div>

由于action是根据概率采样得到的；而给定Environment一个action，其下一个state的产生可能也会产生随机性。因此对于一个参数为$\theta$的Policy而言，其产生的Trajectory是随机的，$R_{\theta}$其实也是一个随机变量。因此，我们只对其期望进行建模：

$$\bar{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta}(\tau) = E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]$$

上述的$\bar{R}_{\theta}$便是我们需要最大化的目标。

<br>
<br>
<br>

## Policy Gradient

得到了优化目标：$\bar{R}_{\theta} = \sum_{\tau} R(\tau) p_{\theta}(\tau)$后，我们如何进行参数更新呢？

一个直观的方法是进行梯度下降，因此我们需要计算优化目标的梯度：$\nabla \bar{R}_{\theta} = \sum_{\tau} R(\tau) \nabla p_{\theta}(\tau) = \sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}$。

可以看到，在训练过程中，$R(\tau)$与参数待更新$\theta$无关，我们不需要对其进行微分计算，因此$R(\tau)$的计算可以只是一个黑盒。

进一步推导，可以得到：

$$ = \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau)$$

实际训练中，一般都是根据训练样本进行期望计算，例如训练样本中，因此可以继续推导为：
$$ = E_{\tau \sim p_{\theta}(\tau)} [R(\tau) \nabla \log p_{\theta}(\tau)] \approx \frac{1}{N} \sum_{n=1}^{N} R(\tau^n) \nabla \log p_{\theta}(\tau^n)$$
$$ = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

其中，Trajectory $\tau^n$是训练样本中的第$n$条Trajectory，Policy$\theta$的大致更新流程如下图所示：

<div align="center">
    <img src="img_4.png"/>
</div>

最后，Policy Gradient方法的梯度更新则可以总结为：

$$\theta \leftarrow \theta + \eta \nabla \bar{R}_{\theta}$$

$$\nabla \bar{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

<br>
<br>
<br>

## 一些Trick

### Add A Baseline

前文可知，Policy Gradient的优化梯度为：

$$\nabla \bar{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} R(\tau^n) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

那么我们可以很直观地知道，这个优化梯度反映的是，当Policy基于state $s_t$选取某个action $a_t$时，如果$R(\tau)$是正的reward，那么则增大这一项的几率，反之则减小这一项的几率。

但是，在有一些情况下，$R(\tau)$也是有可能不是正负兼顾，而是一直为正，且量有大有小，比如打游戏，击败一个敌人和n个敌人都是正向的，只是数量不一样。在这种条件下，如果是理想情况，训练样本中能够采样到所有的action，那么也不会产生什么问题。例如一个action空间中有a、b、c三种action，如下图所示，由于三种action出现的概率相加为1，那么即使他们都被正值的$R(\tau)$加权，他们的概率也不会同时上升，而是$R(\tau)$相对值更大的那个action概率变大。相对值较小的则下降。

<div align="center">
    <img src="img_5.png"/>
</div>

但是，如果在训练样本中我们只采样了action b和 action c的样本会出现什么问题呢？

这时候，因为$R(\tau)$全部为正值，模型看到了action b和action c能够带来正收益，就会“把a的采样可能性向b和c转移”，那么此时action a出现的概率就会减小，而后则产生蝴蝶效应，采样到action a的情况就变得更为困难，如下图所示。如果action a对于模型来说是一个比较好的action，这种梯度计算方式显然是有问题的。

<div align="center">
    <img src="img_6.png"/>
</div>

如何解决这个问题呢？我们可以为$R(\tau)$引入一个“baseline” $b$：

$$\nabla \bar{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R(\tau^n) - b) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

这样即使是$R(\tau)$全部为正值的情况下，如果$R(\tau)$没有满足$b$的门槛，$R(\tau^n) - b$为负，证明这个action还不够好，将其概率降低，为其他好action的采样“让出道路”。

<br>

### Assign Suitable Credit

在梯度计算式中：

$$\nabla \bar{R}_{\theta} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} (R(\tau^n) - b) \nabla \log p_{\theta}(a_t^n | s_t^n)$$

我们可以看到，在某一个$s^n_t$给出一个$a^n_t$这个行为是被$R(\tau^n) - b$加权的，也就是说处于一个Trajectory中的数据都是被同一个值加权的，