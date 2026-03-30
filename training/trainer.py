import numpy as np
from agents.marl_agent import MARLAgent

class Trainer:
    """
    负责训练循环、经验采集和算法更新。
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # 初始化智能体（示例采用参数共享）
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.n
        
        self.agents = {
            agent_id: MARLAgent(agent_id, obs_dim, action_dim, config)
            for agent_id in env.agents
        }

    def train(self):
        print(f"Starting training for {self.config.total_steps} steps...")
        
        step = 0
        while step < self.config.total_steps:
            obs = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_episode_steps):
                actions = {}
                for agent_id, agent_obs in obs.items():
                    action, _ = self.agents[agent_id].select_action(agent_obs)
                    actions[agent_id] = action
                
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                
                # 记录奖励
                episode_reward += sum(rewards.values())
                obs = next_obs
                step += 1
                
                if any(terms.values()) or any(truncs.values()):
                    break
            
            print(f"Step: {step}, Episode Reward: {episode_reward}")
            
            # 策略更新（示例中调用算法更新接口）
            for agent in self.agents.values():
                agent.update(None)

    def save_model(self, path):
        """保存模型"""
        print(f"Saving model to {path}")
        pass
