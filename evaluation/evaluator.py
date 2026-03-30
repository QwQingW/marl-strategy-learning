import numpy as np

class Evaluator:
    """
    用于评估已训练策略的性能。
    """
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config

    def run_evaluation(self, num_episodes=5):
        print(f"Running evaluation for {num_episodes} episodes...")
        total_rewards = []
        
        for i in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    # 确定性策略评估
                    action, _ = self.agents[agent_id].select_action(agent_obs)
                    actions[agent_id] = action
                
                obs, rewards, terms, truncs, _ = self.env.step(actions)
                episode_reward += sum(rewards.values())
                done = any(terms.values()) or any(truncs.values())
            
            total_rewards.append(episode_reward)
            print(f"Eval Episode {i}: Reward {episode_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Evaluation Average Reward: {avg_reward}")
        return avg_reward
