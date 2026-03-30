import numpy as np
from pettingzoo.butterfly import pistonball_v6

class EnvWrapper:
    """
    环境封装类，用于适配 PettingZoo 环境并提供统一接口。
    """
    def __init__(self, config):
        self.config = config
        # 这里以 pistonball 为例，实际可根据 config.env_name 动态创建
        self.env = pistonball_v6.parallel_env(render_mode=None)
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        
        # 获取观测空间和动作空间（假设同构环境）
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

    def reset(self):
        """重置环境"""
        observations, infos = self.env.reset()
        return observations

    def step(self, actions):
        """执行动作"""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        return observations, rewards, terminations, truncations, infos

    def close(self):
        """关闭环境"""
        self.env.close()

    @property
    def single_observation_space(self):
        return self.observation_spaces[self.agents[0]]

    @property
    def single_action_space(self):
        return self.action_spaces[self.agents[0]]
