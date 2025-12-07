---
title: CoT - 《Chain-of-Thought Prompting Elicits Reasoning》
 in Large Language Models
author: luvisdru9
#date: 2025-06-30 00:00:00
#updated: 2025-06-30 00:00:00
tags: 
  - NLP
  - LLM

categories: NLP
description: CoT论文阅读笔记
keywords:
  - NLP
  - LLM

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

## Abstract & Introduction

近年来，扩大LLM的规模已被证明带来了一系列好处，例如性能提升和样本效率提高。然而，仅仅扩大模型规模还不足以在算术、常识和符号推理等具有挑战性的任务上取得高性能。

通过生成通向最终答案的自然语言有益于提升这种推理能力，意味着模型在得出答案前需要输出中间的推理步骤，例如Q：``1 + 1 = ?``，模型并不直接输出`2`，而是进行思考：

- 因为`1 + 1 = 2`；
- 所以最终答案为`2`。

在这之前，相关的研究主要分为两类：

1. **提供中间推理步骤的训练**，这类方法需要大量人工标注推理步骤，成本极高；
2. **few-shot prompting**，类似GPT-3这种few-shot prompting的方法，面对需要推理的任务表现很差，且随着模型规模增加也不会明显变好。

文章结合两种方案的优势，引入模型的**CoT**，Chain Of Thought(思维链)，具体演示如下图所示，即在推理时给定一个``(input, CoT, output)``三元组，使用纯prompting的方法增强模型的推理能力。

<div align=center>
	<img src="img_1.png"/>
</div>

这种纯prompting的方案既没有高昂的训练成本，也不会因为训练导致模型损失任务性能。当模型规模足够大（例如百亿至千亿参数时），只需要给几个CoT示例，它就能模仿这种推理方式，并表现出明显更高级的推理能力，相比于一些专门进行微调的模型在推理任务上展现出了优异的性能表现。

<br>
<br>
<br>

## Chain-of-Thought Prompting

作为一种提升模型推理能力的方法，CoT有以下几个优势：

### 多步骤问题的拆解

不使用CoT的情况下，模型基于这种“给出答案”的任务要求，对于单步骤推理任务和复杂的多步骤推理任务都是“一步到位”。

而CoT则能够对于多步骤的复杂推理任务进行步骤拆分，直观来看增长了这类任务的上下文长度和推理时长，实现了一种根据“推理任务的复杂程度”来适应地分配推理资源的功能。

<br>

### 可解释性

CoT下的模型输出提供了一个可解释的中间过程，虽然这并不能完全揭示模型内部的机理，但是能显示模型是如何得出某个答案的，并为定位推理路径中的错误提供机会。

<br>

### 通用性

根据CoT这种“要求模型先推理后得到答案”的任务特性，能很好地用于数学应用题、常识推理、符号操作等任务，并且理论上可能适用于任何人类能够通过语言推理解决的任务。

<br>

### 便捷性

在足够大的现成语言模型中，只需使用few-shot prompting的方式在模型输入中加入少量的CoT示例，就可以增强模型的推理能力。

<br>
<br>
<br>

## Arithmetic Reasoning

### 实验设置

作者在数学推理任务（Arithmetic Reasoning）上对CoT的效果进行测试，选取了5个相关的数据集：

- GSM8K 基准（数学应用题）；
- SVAMP 数据集（结构多样的应用题）；
- ASDiv 数据集（多样性数学应用题）；
- AQuA 数据集（代数应用题）；
- MAWPS 基准。

<div align=center>
	<img src="img_2.png"/>
</div>

作者为这5个数学相关任务生成了8个CoT模板案例，具体如下图所示：

<div align=center>
	<img src="img_3.png"/>
</div>

将CoT prompting方案与传统的few-shot prompting方案（`Q -> A`）进行对比，选取了5种不同的模型作为载体：

- GPT-3（选择Instruct微调后的模型：ada 350M, babbage 1.3B, curie 6.7B, davinci 175B）；
- LaMDA（422M, 2B, 8B, 68B, 137B）；
- PaLM（8B, 62B, 540B）；
- UL2 20B；
- Codex（代码任务优化后模型：code-davinci-002）。

<br>

### 实验结果

下图为相关实验结果：

<div align=center>
	<img src="img_4.png"/>
</div>

根据实验结果能够得到三个主要结论：

#### 结论1

结果揭示了：CoT Prompt是一种由模型规模所诱发的涌现能力。

对小模型而言，CoT Prompt不会带来性能提升 只有当模型规模达到约 100B 参数级别 时，才会出现性能增益。

作者定性地观察到，小规模模型生成的CoT虽然流畅，但在逻辑上不正确，导致其性能反而比标准Prompt更低。

#### 结论2

CoT Prompt在更复杂的问题上带来的性能提升更大。

例如对于 GSM8K（其基线性能最低的数据集），最大规模的 GPT 和 PaLM 模型的性能在CoT Prompt下提升了超过一倍，而对于 MAWPS 中“SingleOp”这种只需一步推理的简单问题集，性能提升很小甚至为负。

#### 结论3

CoT Prompt下的 GPT-3 175B 和 PaLM 540B 的表现可以和以往最先进方法相比，甚至更优。而以往方法一般需要对模型进行任务特定微调并使用标注训练数据。

图中显示，PaLM 540B + CoT Prompt 在 GSM8K、SVAMP 和 MAWPS 上达到新的 SOTA（注意标准 prompting 在 SVAMP 上已经超过了之前的最优方案），在 AQuA 和 ASDiv 上，PaLM 的表现比 SOTA 仅低 2%。


#### 为何CoT有效

##### 一、CoT反映了模型真实的推理过程，而非表层模仿

作者对 LaMDA 137B 在GSM8K上生成的CoT进行分析，发现了如下现象：

- 在答对的问题中，几乎所有CoT都逻辑严谨且数学正确 -> 说明模型在按CoT逐步推理，而不是“瞎编理由”；

- 在答错的问题中，有 46% 的CoT只包含轻微错误（如计算细节、符号映射或少一步推理）-> 表明模型的推理结构是对的，只是执行中出错；

- 剩余 54% 的错误CoT呈现语义理解偏差或逻辑断裂，这些是当前**模型本身推理能力**的瓶颈所在。

##### 二、模型规模进一步增强CoT的质量与稳定性

对比 PaLM 62B 和 PaLM 540B 的错误模式发现，规模更大的模型能显著减少如下现象：

- 缺少关键步骤的推理
- 数量关系误解
- 语义理解错误
- 

模型规模越大，其CoT越稳定、越完整、越符合逻辑。

<br>

### Ablation Study

文中用三个消融实验说明了CoT对增强推理能力的重要性。

#### Equation Only

在给定答案前不使用CoT，而是直接写出要计算的数学方程，然后再给答案。

结果显示，在 GSM8K 这类复杂语义题上，模型的推理能力几乎没有提升，只能在只需 1–2 步的简单数据集上有一点提升。

这证明了自然语言推理链的语义拆解不可替代。复杂题目不能直接从题目语义映射到数学公式，必须借助自然语言一步步理解题意。

####  Variable compute only

前文提到，CoT间接提升了一些多步骤问题的推理时间，那是否不管什么形式只需要增加推理时间也能达到这种效果呢？

在给定答案前，让模型输出一串dot tokens``......``，数量与解方程所需字符数相同，让模型花更多 token，模拟“更多推理”这个步骤。

结果显示，这样做的效果与 baseline（普通 few-shot）几乎一样。这说明仅增加 token 和计算量并不能提升推理能力，模型需要的是“结构化的推理语义内容”，而不是单纯输出更多字符。

####  Chain of Thought After Answer

将`CoT + 答案`的形式反转为`答案 + CoT`的形式，目的是测试CoT是否只是“激活知识”，与推理顺序无关？ 

结果显示效果与 baseline 基本一样，也起到了增强推理的作用。这其实听上去非常make sense，对于人类而言，不论是先给答案还是先给解题步骤，只要是两个信息都给了，人类就能够将其关联在一起培养推理能力。

<div align=center>
	<img src="img_5.png"/>
</div>

<br>

###  Robustness of Chain of Thought

为了验证CoT是否具有鲁棒性，而不是随着不同人不同风格的编写导致性能明显下降，实验找了三个人`A`、`B`、`C`，编写了三种不同风格的$CoT_A$、$CoT_B$、$CoT_C$，其中`A`还编写了一份简洁风格的`CoT_{A, concise}`，加起来一共四种不同的CoT模板。此外，GSM8K中也会含有带推理过程的数据，将其推理过程抽取出来作为CoT与答案进行拼接也作为一种不同风格的CoT。

实验结果如下图所示，可以看见，不管是什么风格的CoT，都比标准的Prompt要效果更好。

<div align=center>
	<img src="img_6.png"/>
</div>

<br>
<br>
<br>

## Commonsense Reasoning

作者测试了CoT在一系列常识推理数据集上的表现效果，实验结果如下图所示，证明CoT也可以提高在需要各种常识推理能力的任务上的性能，且随着模型规模的增大，这种能力会越发强大。


<div align=center>
	<img src="img_7.png"/>
</div>

<br>
<br>
<br>

## Symbolic Reasoning

使用两种“toy task”来测试CoT在符号推理任务上的性能表现：

- 最后字母拼接：取姓名中每个单词的最后一个字母进行拼接；
- 硬币翻转：给定多次“flip / not flip”步骤后，判断硬币最终正反。

具体的实验结果如下图所示。

得到结论如下：

### 标准Prompt vs CoT Prompt

在标准Prompt下，大模型难以完成这些任务，尤其是多步推理；但使用CoT Prompt后，PaLM 540B 几乎达到了 100% 的正确率，说明CoT 让模型能够执行形式化、逐步、可编程的符号操作，不是浅层匹配；

<br>

### In-domain & Out-of-domain

CoT可分为In-domain和Out-of-domain两种情况，前者即给定的**CoT Prompt模板的数据处理步骤**和给定**待输出结果的数据处理步骤**一致，后者则不一致。例如在最后字母拼接任务中，给定的CoT Prompt中都是关于两个单词姓名的`Amy Brown -> yn`，但是最后要输出推理结果的数据则是三个单词的。

实验结果如下图所示，可以发现，标准Prompt完全失败，这样的Prompt提示下，模型并不会自动扩展规则。而CoT Prompt则可泛化到更长的CoT，且随模型规模增大，OOD 准确率显著提升。但是，100B 规模以下的模型难以对抽象符号进行泛化操作，即便CoT结构非常简单、规则明确。说明CoT 的效果依赖模型规模。


<div align=center>
	<img src="img_8.png"/>
</div>

<br>

<br>
<br>
<br>

## 参考资料

- [《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》](https://arxiv.org/pdf/2201.11903)