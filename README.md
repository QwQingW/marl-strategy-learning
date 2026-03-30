# Multi-Agent Strategy Learning (MARL) Framework

## 项目背景
在多智能体动态环境中，个体决策受到其他智能体行为影响，环境具有高度不确定性。传统基于规则的方法难以建模复杂的博弈关系。本项目提供了一个基于强化学习的框架，旨在让多个智能体在交互过程中学习协同或对抗策略。

## 项目目标
*   构建一个通用的多智能体强化学习（MARL）训练框架。
*   支持智能体在动态环境中通过交互学习有效策略（如竞争、协作、资源分配）。
*   提供模块化设计，支持后续扩展（Reward 优化、策略对比、可视化等）。

## 方法概述
*   **环境**: 基于 [PettingZoo](https://pettingzoo.farama.org/) 提供了多智能体交互的基础。
*   **算法**: 基于 **PPO (Proximal Policy Optimization)**，支持参数共享或独立策略。
*   **架构**: 采用经典的 Actor-Critic 结构，适配多智能体环境的并行观测与动作空间。

## 项目结构
```text
.
├── agents/             # 智能体类实现，支持多 agent 管理
├── algorithms/         # 强化学习算法核心（如 PPO）
├── config/             # 项目配置文件与超参数管理
├── envs/               # 环境封装，适配 PettingZoo 并进行 Wrapper 处理
├── evaluation/         # 策略评估逻辑
├── training/           # 训练循环与经验采集逻辑
├── utils/              # 辅助工具（日志、数据保存等）
├── main.py             # 项目启动入口
└── README.md           # 项目文档说明
```

## 如何运行
### 安装依赖
```bash
pip install torch numpy pettingzoo[butterfly]
```

### 启动训练与评估
```bash
python main.py --total_steps 50000 --num_agents 2
```

## 后续扩展方向
1.  **Reward 设计**: 优化奖励函数以支持更复杂的协作/竞争场景（如 Reward Shaping）。
2.  **多智能体博弈**: 引入不同算法（如 MADDPG, QMIX）进行策略对抗实验。
3.  **可视化模块**: 集成 TensorBoard 或 WandB 进行训练过程监控。
4.  **策略优化**: 引入更复杂的神经网络架构以处理高维视觉观测。
