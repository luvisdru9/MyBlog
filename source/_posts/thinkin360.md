---
title: 《Thinking in 360°:Humanoid Visual Search in the Wild》论文阅读笔记
author: luvisdru9
#date: 2025-07-28 23:00:00
#updated: 2025-07-28 23:00:00
tags: 

  - Robotics
  - DL
  - CV

#hidden: true
categories: Robotics
description: 《Thinking in 360°:Humanoid Visual Search in the Wild》论文阅读笔记
keywords:
  - Robotics
  - DL
  - CV
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

人类通过头部与眼睛的协同运动实现高效的 360° 主动视觉搜索，而传统方法仅依赖**静态图像**，忽略了具身性与三维交互，并不像在真实场景中寻找物品。

文中提出了**Humanoid Visual Search, HVS**，模仿人类这种搜索方式，主要有以下特点：

- HVS是交互式的，从某一个局部视角出发，在一个轻量级的360°场景中行动，每一次头部旋转都会改变局部视角，得到一个新的视觉输入，实现一个闭环的`感知-行动`流程；
- HVS是具身的，将**视觉推理**与**物理动作**紧密耦合，要求任务中的智能体将“头部运动”作为其思考的一部分，模拟了人类的搜索步骤。

同时，具身的搜索任务也被分为两种核心形式：

- humanoid object search (HOS)：类人的目标搜索，定位并将视线聚焦在目标物体上，一般作为操作任务的前置步骤；
- humanoid path search (HPS)：识别通往目的地的可导航路径，并调整身体方向，作为移动的前置步骤。

这种高级的**视觉–空间**推理能力非常适用于人造环境，人造环境中通常存在丰富的结构复杂性、语义复杂性以及体积复杂性，是此类推理最有价值的测试场景。

因此，HVS的关注点从简单受限的场景转向现实中的挑战，例如：
- 在地铁站多层的迷宫结构中导航以找到特定出口；
- 在大型购物中心中找到特定的商店；
- 在摆满货物的超市货架中找到某个特定产品。

但是，由于现有的Embodied AI平台存在**感知真实感不足**和**局限于简单家庭场景**的局限性，无法代表上述所属的复杂人造环境。

为解决这种局限性，文中提出 **H∗Bench**，一个包含多样现实场景、适用于 HVS 的全新benchmark，包括但不限于交通枢纽（机场和地铁站）、大型零售空间（超市和购物中心）以及公共机构（图书馆和博物馆），如下图所示。其中每张全景图都密集地标注了具身任务问题及对应的真实行动，包括**用于 HOS 的最佳头部朝向**和**用于 HPS 的地平面单位方向向量**。

<div align=center>
	<img src="img_1.png"/>
</div>

<br>
<br>
<br>

## Humanoid Visual Search

人类在真实空间中进行决策时，会受到一些**关键的决策点**的影响，在遇到这些关键点后，我们会停下来进行观察、推理并尝试理清一些歧义，而后才能自信地采取下一步行动。

HVS的任务也同样聚焦于这些关键点，将full-body motion抽象为头部旋转这一原子操作。

### 问题表述

想象一个具有局部FoV的智能体正在穿过地铁站中复杂的多走廊交叉口，其任务是尽快找到地铁站出口。

有限的FoV要求头部与眼睛运动之间紧密协调：即头部通过旋转定位到新的观察点来探索未知区域，而眼睛通过凝视看到任务相关细节进行利用。

具体形式上，将环境建模为一张360°全景图像，所有可能的观察集合定义为为$S_o = {o_{\phi, \gamma}}$，包含从该全景中采样的窄FoV图像，每个视角由其方位角$\phi$和俯仰角唯一确定。

现在HVS的目标是找到最优方向$(\phi^{*}, \gamma^{*})$，在给定语言指令$x$和视觉观察$o_{\phi, \gamma}$，使得任务成功的概率$r_s$最大化：

$$(\phi^{*}, \gamma^{*})=argmax_{\phi, \gamma}P(r_s | o_{\phi, \gamma}, x)$$

#### HOS

HOS解决的是在未知3D环境中主动搜索目标的问题，其方式是找到一个最终观察方向$(\phi^{*}, \gamma^{*})$，使得目标出现在视图中央的 **central foveal region**（中央凹区域）中。

#### HPS

HPS要求智能体搜索通向目标位置的可导航路径，这是移动之前的高层规划，目标是确定一个最终的朝向$\phi^{*}$。

<br>

### 利用MLLM进行HVS

文中将HVS建模为一个多模态推理任务，通过将**MLLM的工具使用**和**头部旋转**耦合实现。

具体方案采用了[《Thinking with Images for Multimodal Reasoning:
 Foundations, Methods, and Future Frontiers》](https://arxiv.org/pdf/2506.23918)这篇工作的工具增强型MLLM，将智能体策略定义为$\pi_{\theta}(y_t,a_t|o_t,x,H_t)$。

具体来说，在每一个时间步$t$，智能体会生成一个文本形式的CoT $y_t$和一个动作$a_t$，其条件为：

- 当前的观察 $o_t = o_{\phi, \gamma}$；
- 语言指令 $x$；
- 历史状态 $H_t = {(o_i,y_i,a_i)}_{i=1}^{t-1}$

每一个episode允许一系列旋转动作，并以提交动作$a_t$作为结束标志，最终把动作$a_t$提交作为最终输出。动作空间包含两个动作原语：

#### Rotate 

$Rotate a_t^rot = (\Delta \phi, \Delta \gamma)$调整观察方向（右/上方向为正，yaw循环），使得：
- $\phi_{t+1} = \phi_t + \Delta \phi$
- $\gamma_{t+1} = \gamma_t + \Delta \gamma$

#### Submit

$Submit a_t^{sub}$ 将当前方向作为最终估计的$(\hat{\phi}, \hat{\gamma})$并终止当前episode。

<br>

### MLLM Post-training

MLLM训练于静态、无embodied的互联网数据，并不具备空间常识和主动3D规划能力，即使最先进的GPT-4o成功率也仅约20%。

因此，通过下图所示的两阶段后训练流程将MLLMs适配为有效的视觉搜索智能体。

<div align=center>
	<img src="img_2.png"/>
</div>

#### SFT

在一个精心挑选的多轮数据集上进行SFT，为模型灌输基本的**任务导向推理能力**和**工具使用能力**。

这一步骤能够教会模型如何根据多模态输入生成**结构化的行动计划**，建立一个**强有力的行为先验**。

#### Multi-Turn RL

使用GRPO进行进一步的策略优化。这一步的RL鼓励长时间尺度的推理，对于发展稳健、可泛化的搜索策略至关重要。

<br>
<br>
<br>

## H*Bench

### 数据集概述

H*Bench包含大约3000个带有注释的任务实例，这些实例来自多种高分辨率的全景视频（最高可达$7680 × 3840$）。

通过为每个任务实例设置**四种不同的初始朝向**，获得了共12000个搜索episode。

H*Bench的数据来自全球各地区以及开放平台，具有广泛的地理覆盖范围和场景多样性，系统地划分为6大场景类别和18个细粒度场景类型。具体统计数据如下图所示：


<div align=center>
	<img src="img_3.png"/>
</div>

<br>

### Benchmark Construction

#### Task Annotation

对于一个全景图而言，在一个透视视角的界面中对其进行标注，根据已知视角角度$(\phi, \gamma)$从全景渲染该视角下的局部的 FoV 图像。

标注者可以自由旋转虚拟相机来检查场景，确定一个合适的具身搜索任务，而后编写自然语言指令，并通过绘制边界框来标记目标，以指明其最优方向。该边界框随后被反投影到全景图，其中心即为目标的最优方向$(\phi^*, \gamma^*)$；对于 HPS仅保留$\phi^*$，因为对于路径规划而言，可将行走平面近似为二维平面。


####  Cold-Start Data Curation

由前文可知，在一个搜索episode的训练中，搜索过程并不是一步到位的，而是给定了当前视角和最优目标视角的Ground Truth后，通过多轮“旋转 + 推理”才能最终提交。

为了构建用于 SFT 的高质量多轮决策数据，从任务实例中选取一个子集，并通过prompting一个强大的 MLLM（GPT-4o）来生成结构化的 CoT。

采用人类参与的构建流程，标注者会审查并修改生成的CoT推理，以消除幻觉，确保推理是基于可见的视觉线索、并保持风格一致。

最终的数据集包含 2000 条多轮次数据，包含视觉观察、已验证的 CoT 推理和动作，用于MLLM冷启动 SFT 训练。总共有 6 名标注者投入了 250 小时进行具身任务标注和 CoT 修订工作。

####  Difficulty Taxonomy

对于 HOS，根据目标物体在初始视角中的可见性来定义任务难度。

具体的，计算一个可见度比例$d$，它是**初始视角中目标可见区域面积**与**整个目标面积**的比值。可见度越高，感知线索越强，探索难度越低；反之则需要更多的视觉探索。因此，将 HOS 样本分为 **Easy**、**Medium** 和 **Hard** 三类。

<div align=center>
	<img src="img_4.png"/>
</div>

对于 HPS，难度取决于场景是否包含文本信息，以及视觉 / 文本线索是否与实际路径方向一致。这两个因素共同定义了四种难度级别。

<div align=center>
	<img src="img_5.png"/>
</div>

<div align=center>
	<img src="img_6.png"/>
</div>

<div align=center>
	<img src="img_7.png"/>
</div>

<div align=center>
	<img src="img_8.png"/>
</div>

<br>
<br>
<br>

## 实验

### 实验设置

#### 实现细节

在HOS和HPS的混合数据集上微调模型，SFT基于LLaMA-Factory实现，RL 训练在开源框架 VAGEN 中进行。

对 Qwen2.5-VL-3B-Instruct 进行全参数 SFT，共 3 个 epoch；得到的模型称为 **HVS-3B (w/ SFT only)**。

在 RL 阶段，训练了 70 步。得到的模型记为 **HVS-3B**。

实验中使用的 prompts 如下图所示：

<div align=center>
	<img src="img_9.png"/>
</div>

此外，还评估了支持多图像输入的**开源和商用模型**。

#### 评估指标

对于一次最终提交的方向$(\hat{\phi}, \hat{\gamma})$，以标注的Bounding Box的中心方向$(\phi^*, \gamma^*)$为参考，如果提交的方向落到了以该中心方向为中心的**可容忍区域** $[ \phi^* - \pi \phi, \phi^* + \pi \phi] × [\gamma^* - \pi \gamma, \gamma^* + \pi \gamma]$，则看作成功。

其中，$\tau_{\phi} = \max\left(\frac{w_{\phi}}{2}, \tau_{\phi}\right),  \tau_{\gamma} = \max\left(\frac{w_{\gamma}}{2}, \tau_{\gamma}\right)$，$w_{\phi}$和$w_{\gamma}$分别为Bounding Box的Angular Width和Angular Height。

根据前文所述，HOS任务需要评估$(\hat{\phi}, \hat{\gamma})$，而HPS任务只评估$(\hat{\phi})$。为了模拟人类的中央凹视野，HOS任务容忍范围设置为$\pi \phi = 30°, \pi \gamma = 20°$；而为了反映对精准运动方向的需求，HPS的容忍范围设置为$\pi \phi = 10°$。


<br>

### 探究 MLLM 的具身视觉搜索能力

#### 主要结果

下表显示商用模型与开源模型之间存在显著性能差距。

<div align=center>
	<img src="img_10.png"/>
</div>

在 HOS（31.96）和 HPS（33.00）任务中，**Gemini 2.5-Pro** 是整体表现最好的商用模型；开源模型中，Gemma-3 系列取得了最好结果。

有趣的是，更大的模型规模并不能保证更好的性能。对于 Gemma-3 系列和 Qwen2.5-VL 系列来说，小模型（4B/3B）在 HOS 中反而超过了更大的 12B/7B 模型，并且在 HPS 上表现相当。

#### 错误分析

在 HOS中，错误来自以下两个方面：

- **视觉落地能力不足**：模型难以在杂乱环境中稳定识别目标；
- **感知-行动脱节**：模型能看到目标，但无法进行精细的中心凹对准。


在 HPS中，错误主要有三类：

- **视觉-行动不匹配**：模型能看到路牌等视觉线索，但不能将其转化为行动；

- **缺乏物理常识**：行动违反 3D 约束（如试图穿墙）；

- **缺乏社会-空间常识**：模型忽视建筑环境中的**隐性规则**（如楼梯用途、警戒带、斑马线）。

对结果的分解具体如下图所示：

<div align=center>
	<img src="img_11.png"/>
</div>

以上的发现都证明：**“MLLM 能在被动的世界描述中构建语言基础的空间模型，但无法构建面向具身交互的物理落地模型”**


<br>

### 后训练的作用与局限性

#### SFT和RL的局限性

SFT 提供了大部分性能增益，在 HOS 上，SFT 使整体得分从 **14.83** 提升到 **40.83（↑26.00）**；在 HPS 上，从 **6.44** 提升到 **23.00（↑16.56）**。

随后的 RL 带来额外但较小的增益：**HOS ↑6.55**，**HPS ↑1.94**。

这表明SFT 构建了基本任务能力，而 RL 充当进一步优化的步骤。具体而言，后训练提升了以下关键能力：

- 对旋转角度的精确控制；

- 使用大角度转向来探索新区域；

- 根据方向性标志采取行动，具体如下图所示。

<div align=center>
	<img src="img_12.png"/>
</div>

此外，如果没有 SFT 而直接应用 RL，会削弱模型的指令跟随能力。

#### 任务相关的效果差异

后训练的收益因任务复杂度而异。对于简单的HOS，文中所训练的模型（47.38）超过最先进的 Gemini2.5-Pro（31.96）。但对于复杂的路径搜索，其模型得分（24.94）落后于最优模型（33.00）。

这一差距说明**后训练在提升高阶空间推理能力方面存在局限**（模型的基础能力很重要）。

#### RL在复杂任务中的负面效果

在 HPS 中，RL 将模型在中等难度任务的表现从 23.03 降到 20.18；在极端难度上从 14.81 降到 12.04。这些场景的特点是：**视觉线索与最优路径之间存在错位**，这在错误分类中被认为难度很高。

文中推测性能下降源于 **reward hacking**，模型学会**利用奖励信号**而非真正提升推理能力。这更证明了，设计能够在所有难度下与任务目标一致的奖励函数是非常困难的。

<br>

经过上述的分析，可以得到关键结论：`SFT+RL 在HOS上能够显著提升视觉落地与探索能力；但在HPS上难以传授物理、空间和社会常识，这些能力往往是隐性的、情境化的、过程性的`。

<br>

### HOS和HPS的进一步分析

<div align=center>
	<img src="img_13.png"/>
</div>

上图分析了MLLM Baseline、in-task、cross-task和HVS四种方案的性能，其中in-task即在同一个任务上训练+测试，比如在HOS上训练，在HPS上测试；而cross-task则是在一个任务上训练，在另一个任务上测试。

#### in-task的优越性

由上图可知，in-task训练一般带来最强性能，但有一个例外：一个**仅用 HOS 训练**的模型在简单 HPS 上达 37.8%，超过Baseline（7.0%）和 HPS 专用模型（33.8%）。

作者推测，这些简单的HOS任务等价于“简单物体搜索”，有清晰的视觉线索来引导路径，使 HOS 中的**强物体识别能力**可以很好迁移，更具有泛化性。


#### cross-task的泛化性

此外，可以由结果观察到明显的**双向增益**：

- 用HOS训练能将HPS从 6.4% → 20.7%；

- 用HPS训练能将HOS从 14.8% → 29.5%。

原因是HPS中学到的主动探索和路径推理能帮助HOS，HOS中学到的视觉落地能力也能帮助HPS，这些能力都是互相促进的。

#### 混合数据训练

使用混合的HOS + HPS数据可获得整体最佳表现。但存在如下挑战：性能提升分布不均衡，某些难度上提升会导致其他难度下降。因此在训练通用的 humanoid agent 时，如何平衡这一权衡非常关键。

<br>

### 消融实验

#### Reward Shaping

文中对HPS任务中的RL环节尝试三种奖励设计方案：

- format + correctness；

- format + correctness + distance-to-goal；

- format + distance-to-goal。

这些奖励变体只在简单难度提升性能，而在更难场景中往往性能下降，如下表所示。这更强调了HPS这个任务的难度，需要更先进的方案。

<div align=center>
	<img src="img_14.png"/>
</div>

#### Training Rollout and Context Length

采用短rollout的 GRPO 的模型在测试时进行 **test-time scaling** 即可达到长 rollout 的性能，同时收敛更快。因此短 rollout 可保持训练效率。

此外，对于 HVS 模型， 输入只包含最近 2 轮的历史短上下文对话就已足够。

#### 主动视觉 vs. 被动视觉

消融实验还比较了下述两种搜索方案：

- 主动视觉搜索：智能体旋转相机逐步获取信息；

- 被动视觉：直接分析完整全景图。

**主动方式**在以下两个方面上更优：

- 更符合高效的类人搜索策略；

- 避免全景图畸变与 MLLM 的图像处理先验冲突（MLLM训练分布中一般不具备这种全景相机的分布）。

实验也证明了这一点，使用 Gemma-3-4B-it 时，被动方式显著降低性能，如下图所示。

<div align=center>
	<img src="img_15.png"/>
</div>

#### 具身 vs. 非具身 Benchmark

<div align=center>
	<img src="img_16.png"/>
</div>

如上图所示，2D 模型 Mini-o3 与 Chain-of-Focus 在**非具身**的 V* Bench上达到近饱和性能（88.2%、88.0%），但在具身 H*Bench 上表现暴跌到 2.5% 与 11.6%。

说明从互联网被动数据学到的能力无法迁移到 3D 具身交互，且图中显示 HVS-3B 也只有 38.4% 成功率，显示该问题远未解决。

虽然3D具身交互仍然是个问题，但反过来，HVS-3B在 V* Bench 上仍保持 65.5% 成功率。这说明在具身benchmark上进一步训练的模型学习到了 3D 具身搜索，同时没有过多损害其 2D 搜索能力，展示出统一模型的潜力。


<br>
<br>
<br>

## Discussion and Future Work

这篇文章主要研究如何纠正当前视觉模型偏向 2D、passive 的问题，转向 embodied + 3D + active search。

post-training可提升一些**低层次的感知-运动能力**，例如：

- 视觉 grounding（识别物体位置）；
- 基础 exploration（旋转、查看死角）；
- 控制角度输出（如 10°、30° 转动）。


而对于一些高层次的**常识推理**而言，post-training展现出了一些瓶颈，例如：

- physical commonsense（不能穿墙、不能走悬空）；

- spatial commonsense（可通行的楼梯、道路、拐角...）；

- social commonsense（警戒线不能跨越、人应该走斑马线）。



此外，文章发现 RL 可以在简单任务上提升能力，但在复杂任务中会存在误导（reward misalignment）、奖励欺骗（reward hacking）、过拟合简单的 pattern等问题。

性能上表现为：

- 复杂 HPS 性能下降；
- 把不正确的策略误当成“高分”策略；
- 模型变得更僵化、规则化。

这是 RL 在大模型中的通病，reward的设计很难对齐高阶推理目标，一个好的奖励设计是非常困难的。