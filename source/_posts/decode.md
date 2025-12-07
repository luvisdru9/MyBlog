---
title: LLM解码策略
author: luvisdru9
date: 2025-08-06 00:00:00
updated: 2025-08-06 00:00:00
tags: 
  - NLP
  - LLM
  - DL
  - ML
categories: NLP
description: LLM解码策略
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

**模型解码（decoding）**指的是模型在生成文本时，模型会根据前面的内容预测下一个最可能出现的 token，直到满足终止条件（比如达到最大长度或遇到结束符 </s>）。

解码策略决定了模型如何从多个候选token中做出选择，不同策略在不同情况下带来的效果是不尽相同的。假设模型已经生成了前n-1个token：``x1、x2、...、xn-1``，概率分布：``P(x|x1、x2、...、xn-1)``描述了模型在已生成文本的基础上选择某个 token的可能性，解码策略的关键就在于**如何从这个概率分布中选择最合适的token**。


<br>
<br>
<br>

## Temperature

解码中通过引入一个大于0的温度参数，来控制概率分布的平滑程度。当温度等于1时，没有变化；当温度小于1时候，概率分布变得更激进，高概率和低概率变得更为两极分化；而当温度大于1时，概率分布则变得“温和”，对于低概率分布token更为包容，而对于高概率分布token也没有那么偏爱了。

$$ P_{\tau}(x|x_{< t}) = \frac{ P(x|x_{< t}) ^{1/\tau}}{\sum\limits_{x'}  P(x'|x_{< t}) ^{1/\tau}} $$

<br>
<br>
<br>

## Decoding Method

### Greedy Search

贪心解码的原理非常简单，即每一步都选择概率最高的token作为下一个生成的token：

$$ \hat{x}_t = argmax_x P(x|x_{< t}) $$

优点：生成的文本通常会较为确定，计算速度快，适合实时生成。

缺点：
- 容易陷入局部最优：只选每步概率最高的词，可能错过整体概率更大的序列；
- 生成结果单调、缺乏多样性：重复或缺少创造性表达；
- 忽略全局上下文，只关注当前一步最优。

<br>

### Random Sampling

随机采样则从当前给出的概率分布中随机选择一个token作为生成结果：

$$\hat{x}_t \sim argmax_x P(x|x_{\lt t})$$

优点：生成的文本更加随机，带来更多的想象力和创造性，可用于一些创造性文本的生成、开阔性思维的生成。
缺点：
	- 不稳定，易生成无意义或不连贯文本，因为可能采样到概率很低的词；
	- 生成质量波动大，可能出现语法错误或语义跳跃。

<br>

### Top-K Sampling

Top-K策略先选取概率分布最高的k个token，再在这k个token中随机选取一个作为最后的生成结果：

$$V^{(k)} = \{x | x \in TOPK\}$$

$$
P'(x \mid x_{< t}) = 
\begin{cases}
\displaystyle\frac{P(x \mid x_{< t})^{1/\tau}}{\sum_{x' \in V^{(k)}} P(x' \mid x_{< t})^{1/\tau}}, & \text{if } x \in V^{(k)} \\
0, & \text{otherwise}
\end{cases}
$$

$$ \hat{x}_t \sim argmax_x P'(x|x_{< t}) $$


优点：结合了概率分布的可靠性和随机采样的随机性，能够保证在一定合理范围内增加生成的多样性。
缺点：
- 对k的设置很敏感， k过大过小都不合适;
- K值是固定的。有时候概率分布很平坦，需要一个很大的K值才能包含所有合理的选项；有时候概率分布很集中，可能前2个词就占了99%的概率，此时一个大的K值反而会纳入不必要的词。

<br>

### Top-P Sampling

按照概率分布从大到小进行排序，找到最小集合``V(p)``，使得其中的token的概率和大于等于某个阈值：

$$\sum_{x \in V^{(p)}} P(x|x_{\lt t})\ge p$$

对``V(p)``中的概率进行重新归一化：

$$
P'(x \mid x_{< t}) = 
\begin{cases}
\displaystyle\frac{P(x \mid x_{< t})^{1/\tau}}{\sum_{x' \in V^{(p)}} P(x' \mid x_{< t})^{1/\tau}}, & \text{if } x \in V^{(p)} \\
0, & \text{otherwise}
\end{cases}
$$

从归一化后的token中随机采样：

$$\hat{x}_t \sim argmax_x P'(x|x_{< t})$$

优点：相较于Top-K更为灵活，能够自动根据概率分布调整候选集大小。
缺点：
- 计算复杂度较高，需要排序和累积概率计算；
- 阈值仍然选择敏感，过小丢失多样性，过大可能导致生成质量下降；
- 仍可能生成不连贯内容，尤其在模型概率分布不准确时。

<br>

### Beam Search

束搜索每一步保留 top-k 个候选序列：

$$
P_{\text{beam}}(y_t \mid y_{<t}, x) = 
\begin{cases}
\displaystyle\frac{\exp(\log P(y_t \mid y_{<t}, x) / \tau)}
{\sum_{y' \in \mathcal{B}_t} \exp(\log P(y' \mid y_{<t}, x) / \tau)}, 
& y_t \in \mathcal{B}_t \\
0, & y_t \notin \mathcal{B}_t
\end{cases}
$$

其中 $\mathcal{B}_t$ 是在第 $t$ 步的候选序列， $\mathcal{S}$：
$$
\mathcal{B}_t = \underset{\substack{\mathcal{S} \subseteq \mathcal{V} \\ |\mathcal{S}| = k}}{\arg\max} 
\sum_{y \in \mathcal{S}} \log P(y \mid y_{<t}, x)
$$

下一步则在候选序列的每一个选择基础上继续保留 top-k 个候选序列，最后达到终止符或者最大长度会形成若干条路径，选择得分最高的路径。

优点：考虑到全局的信息，能够生成更流畅和高质量的内容，适用于机器翻译等任务。
缺点：
- 计算资源消耗大，尤其束宽度大时搜索空间大；
- 容易生成缺乏多样性、重复内容，因为只选概率最高路径，导致结果单一；
- 束宽度过小效果类似贪心解码，过大则增加计算且未必提升质量；
 - 偏向短序列，因总概率乘积或累积对长度敏感。

<br>

### Speculative Decoding

也称投机解码，是Google、DeepMind在2022年发现的大模型推理加速方法，这种方案需要两个模型一个是主模型(Target model)， 一个是轻量模型（Draft Model），这里的轻量模型可以用蒸馏/量化的方式得到与主模型相似的输出分布。

轻量模型体量小，生成token耗时少，主模型只需负责验证轻量模型的输出即可，避免大模型做多轮预测输出，导致大量耗时，生成主要流程如下：
- 由轻量模型在前文``T_pre``的基础上生成n个候选token；
- 将这n个候选token合并在前文后面，成为一个新的输入``T_new``；
- 将``T_new``输入到主模型进行一次forward，得到n个候选位置处的概率（这里算的是一次forward的hidden state，不要与generate后文n次混淆），此处为并行计算，耗时能够减少1/2左右。

**投机解码接收逻辑：**对于某个token而言，令Q为轻量模型得到的概率，P为主模型forward后输出的概率，生成一个随机的阈值，如果P/Q大于某个阈值，说明主模型对于轻量模型生成的结果甚至更有自信，说明无需更改即可被接受。

**投机解码拒绝逻辑：**如果验证当前token时，P/Q没有超过阈值，说明主模型对这个token没那么自信，则主模型以P/Q的概率接收当前token，1-P/Q的概率拒绝这个token。

而后如果轻量模型生成的k个结果都满意的话，则使用主模型采样下一个token，结合这k个token一起作为结果输出；如果对于第n+1个token不满意，这时候需要创造一个新的分布``p'(x)=norm(max(0, pn+1(x)-qn+1(x)))``，这里先计算差分，结合max函数，将那些轻量模型比主模型更为自信的token所在的概率变为0，而主模型更为自信的地方则是正数，使得轻量模型过于自信的token被削减，最后归一化后进行第n+1个token的重采样，后续token全部丢弃，下一轮继续由轻量模型生成新的候选序列。


#### 投机解码优化措施

在投机解码中，轻量模型生成的token接受率深刻受到了该模型与主模型分布一致性的影响，优化轻量模型的分布是投机解码进行优化的核心问题之一：

- DistillSpec（Distilled Speculative Decoding）：基于蒸馏学习从主模型中蒸馏出轻量模型
- SSD（Self-Speculative Decoding）：自动选择主模型中的部分层（如前几层）作为轻量模型，无需重新训练
- OSD（Online Speculative Decoding）：长期使用后，可能用户的需求不再和轻量模型的分布匹配，OSD能够在线动态调整轻量模型，具体的，线上运行时记录哪些 token 被主模型拒绝，用这些数据对草稿模型做在线蒸馏，使其逐渐适应新数据分布
- PaSS（Parallel Speculative Sampling）：让主模型自己充当轻量模型的角色，自己生成草稿。具体的，在输入序列的基础上构造训练是的鳄lookahead token序列，如``"The capital of France is [MASK] [MASK] [MASK]"``，进行一次forward后预测得到三个token，剩下的步骤与常规投机解码大体一致
- REST（Retrieval-Enhanced Speculative Decoding）：事先准备了大量高质量的对话片段和相关上下文对，向量化存储在检索库中，系统将当前用户上下文转成向量，去检索库中找最相似的上下文，直接把这段“回答”作为草稿 token 序列，主模型对检索得到的草稿 token 进行验证
- SpecInfer：可以使用一个或者多个SSM(Small Speculative Model)生成不同的候选序列，而后将这些序列构成一棵树，每层代表同一个位置的多个可能 toke主hu模型用专门设计的“树形注意力机制”同时对树中所有路径的 token 进行概率计算（前向传播一次完成多条序列验证），而不是一条条串行验证。对所有路径中符合大模型预测概率较高的分支，批量接受对应的 token，提升推理吞吐；对概率较低或大模型不认可的分支进行剪枝，减少无效计算
- Medusa：在大模型基础上添加多个微调头，每个头专门生成一个未来token，利用这些头并行生成多个草稿序列进行验证
- Eagle：轻量模型是一个同构的轻量LLM，其中Embedding层和LM Head层均复用原始LLM模型，中间的One Auto-regression Head(简称 AR Head)为由一层FC层以及一层Transformer Layer组成。AR Head是唯一需要微调的网络层，训练成本也是极低。使用 轻量模型通过自回归采样方式生成草稿，并且为了提升接收率，在运行AR Head前会融合上一个Token的隐状态与当前Token的 Embedding层，并通过AR Head的FC层融合：``[seq_length, hidden_size*2]->[seq_length, hidden_size]``，轻量模型基于草稿树进行自回归生成

另一方面研究侧重于设计更有效的草稿构建策略。传统的方法通常产生单一的草稿token序列，这对通过验证提出了挑战：

- Spectr：不只生成一条草稿，而是同时生成多条草稿序列（k 条），然后使用k-sequential并行交给大模型验证（多个候选序列作为一个batch输入）。
- SpecInfer：SpecInfer 通过 token tree（令牌树）结构来组织多个草稿序列，并引入 Tree Attention（树形注意力）验证机制，让大模型可以高效并行地验证多条候选路径（如果我们有多个草稿序列（多条分支），直接把它们拼接成一个长序列做普通自注意力，会浪费大量计算：前缀共享的部分会重复计算；还可能造成不必要的信息交叉（不同分支不应互相影响），每个token只与其所在分支的祖先节点或共享前缀节点计算注意力）

<br>
<br>
<br>

## 参考资料
- https://zhuanlan.zhihu.com/p/716344354
- https://www.cnblogs.com/rossiXYZ/p/18837229#534-%E4%BC%98%E5%8C%96