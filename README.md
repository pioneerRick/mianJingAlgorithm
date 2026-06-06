改项目用于存储大模型算法面经



2026年6月6日

从面经、技术报告、项目三个方面进行介绍

# 面经

1. Transformers 结构（Encoder、Decoder、KV Cache 等概念）
2. PE 算法原理（YARN、ROPE 等）
3. MOE（注意各种路由选择算法等）
4. 多机多卡训练经验（Scaling Law、训 7B模型要多少卡、DeepSpeed等）
5. MOE 和 Agentic 的训练细节（比如 loss 的 mask 细节、多轮对话怎么处理）
6. RL 训练策略公式（GRPO、DAPO、GSPO、DPO、PPO，注意一下 GRPO 和 DPO 和 PPO 区别，几家大厂都问我相同问题了）。目前还遇见过手写grpo损失公式。
7. VIT 和对齐原理（多模态方向）
8. 模型参数命名（比如 MOE 什么 A22 的意思、Qwen 的 VL 和普通版本区别、think 和 instruct 的区别，细节一定要分清）
   手撕：Encoder、PE、 交叉熵、 InfoceLoss（一般练 pytorch 版本即可，不放心自己还练一下 numpy 的版本）。还有手写MHA、旋转数组。



# **技术报告**

1. Qwen3.5（和 3、2.5 的区别，做多模态朋友要看 VL 的区别）。观察Qwen3.5中统一多模态的设计，未来估计多模态大模型是一个趋势。
2. KIMI2.0 和KIMI k2.5。观察Kimi 2.5中有关于agent能力的考查，自从Claude code出现之后，agent 编程、设计之类的方向因其盈利能力，成为大模型厂家的角逐热点。

# 项目

## 训练框架

1. Slime 
2. AReal

关注点：看看这两个框架里异步是怎么实现的，数据流怎么组织，rollout、reward、update是怎么串起来的

如何进行学习：这两个框架的GitHub主页里有很多Example，从Example进行学习



## Agent RL项目 

1. SimpleTIR
2. Search-R1
3. Retool

理解RL如何作用在agent任务上，不同任务定义和奖励设计有什么区别，verl框架的example里也有很多项目，可以挑自己感兴趣的看看。



## Deep Research

首先了解什么是Deep Research技术：[(41 封私信 / 15 条消息) 万字长文深度解析最新Deep Research技术：前沿架构、核心技术与未来展望 - 知乎](https://zhuanlan.zhihu.com/p/1972258410557862481)

可以查看 字节出品的 deerflow 项目



## Agent Memory 

这个是我一直研究的领域，所以可以多推荐一点

1. Memo0
2. Memory-R1
3. A-Mem
4. MemGPT
5. MemOS
