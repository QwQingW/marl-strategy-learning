import argparse

def get_config():
    parser = argparse.ArgumentParser(description="MARL Framework Configuration")
    
    # Environment
    parser.add_argument("--env_name", type=str, default="pistonball_v6", help="PettingZoo environment name")
    
    # Training
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    
    # PPO Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--ppo_epoch", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval for logging")
    
    return parser.parse_args()
