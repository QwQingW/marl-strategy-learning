# Multi-Agent Strategy Learning (LLM-Guided MARL)

本项目构建了一个基于多智能体强化学习（MARL）的轻量化沙盒环境，旨在研究大语言模型（LLM）在高层决策中对底层强化学习策略的影响。系统在 PettingZoo 的 AECEnv 框架下实现，结合了由 LLM 引导的高层语义理解与基于 PPO 的底层策略优化。

## 核心架构：分层控制结构

本项目采用**分层控制（Hierarchical Control）**架构，将智能体的决策过程分为两个层次：

1.  **高层决策模块 (LLM)**：利用大语言模型强大的理解能力，实时分析当前环境状态，动态生成高层行为目标（如：**生存、采集、战斗**）及其关联的**奖励权重**（Reward Weights）。
2.  **底层执行模块 (RL - PPO)**：基于 [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) 算法，智能体在 LLM 给定的奖励权重引导下，通过与环境的交互学习具体的动作执行策略。

这种“语言模型调控策略学习”的框架，实现了从高层语义指令到底层精密操作的统一转化。

## 关键特性

*   **对抗交互环境**：在轻量级沙盒中实现玩家与敌人的实时对抗，双方均由 PPO 策略控制，通过交替训练（Self-Play/Alternating Training）达到博弈平衡。
*   **多目标决策场景**：环境中包含**资源点**（提供能量/得分）与**治疗点**（恢复状态）等被动实体，为智能体提供了复杂的生存与竞争冲突点。
*   **动态奖励机制**：区别于传统固定参数的奖励函数，LLM 根据战场局势动态调整不同目标的权重，引导强化学习过程快速且合理地收敛。
*   **PettingZoo AEC 架构**：严格遵循 PettingZoo 的 Agent-Environment Cycle (AEC) 规范，确保多智能体间动作顺序与状态同步的准确性。

## 研究目标

*   **策略收敛速度**：验证 LLM 生成的动态奖励引导是否能显著减少传统强化学习所需的冷启动时间。
*   **行为合理性**：评估在 LLM 引导下，智能体是否能展现出更具逻辑性、符合人类预期语义的高级策略。
*   **决策稳定性**：探究在复杂对抗环境中，这种混合驱动范式对策略鲁棒性的提升效果。

## 项目结构

```text
.
├── agents/             # 智能体逻辑实现，包含 LLM 高层决策接口与 PPO 底层封装
├── algorithms/         # 强化学习算法核心（PPO 优化、模型构建）
├── config/             # 系统配置、超参数管理、LLM Prompt 模板
├── envs/               # 沙盒环境实现（资源点、治疗点逻辑，PettingZoo 适配）
├── evaluation/         # 多维评估模块（收敛曲线分析、胜率对比、行为合理性指标）
├── training/           # 训练循环、经验回放缓存、双端交替训练逻辑
├── utils/              # 通用工具类（日志、模型保存、数值处理）
├── main.py             # 项目统一启动入口
└── README.md           # 项目中文文档
```

## 快速开始

### 环境依赖
*   Python 3.8+
*   PyTorch
*   PettingZoo
*   OpenAI API / 其他 LLM 服务（用于高层决策）

### 安装
```bash
pip install torch numpy pettingzoo[butterfly] openai
```

### 运行实验
```bash
# 启动带有 LLM 引导的高层策略训练
python main.py --mode train --use_llm True
```

## 技术路线
1.  **环境建模**：基于 `pettingzoo.AECEnv` 构建包含资源/治疗点的二维网格沙盒。
2.  **分层接口**：建立环境状态到 LLM Prompt 的映射，以及 LLM 输出到 Reward Function 的解析逻辑。
3.  **对抗训练**：实现基于 PPO 的异步交替训练，作为系统的底层策略库。
4.  **评估分析**：对比纯 RL 策略与 LLM-Guided 策略在多项指标下的表现。
