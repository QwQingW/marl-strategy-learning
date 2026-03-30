from config.base_config import get_config
from envs.env_wrapper import EnvWrapper
from training.trainer import Trainer
from evaluation.evaluator import Evaluator

def main():
    # 1. 获取配置
    config = get_config()
    
    # 2. 初始化环境
    env = EnvWrapper(config)
    
    # 3. 训练阶段
    trainer = Trainer(env, config)
    trainer.train()
    
    # 4. 评估阶段
    evaluator = Evaluator(env, trainer.agents, config)
    evaluator.run_evaluation(num_episodes=config.num_agents)
    
    # 5. 关闭环境
    env.close()
    print("MARL Task Completed.")

if __name__ == "__main__":
    main()
