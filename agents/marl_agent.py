from algorithms.ppo import PPO

class MARLAgent:
    """
    多智能体中的单个 Agent 实例。
    """
    def __init__(self, agent_id, obs_dim, action_dim, config, shared_algo=None):
        self.agent_id = agent_id
        self.config = config
        
        # 支持参数共享或独立策略
        if shared_algo:
            self.algo = shared_algo
        else:
            self.algo = PPO(obs_dim, action_dim, config)

    def select_action(self, obs):
        return self.algo.select_action(obs)

    def update(self, experiences):
        self.algo.update(experiences)
