import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPO:
    """
    PPO 算法核心类，包含 Policy 和 Value 网络及其更新逻辑。
    """
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.value = ValueNetwork(obs_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': config.lr},
            {'params': self.value.parameters(), 'lr': config.lr}
        ])

    def select_action(self, obs):
        """选择动作"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        probs = self.policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rollouts):
        """更新策略和价值网络参数"""
        # 简化版更新逻辑
        print("Updating PPO parameters...")
        pass
