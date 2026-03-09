from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import TrafficIntersectionEnv

def train_agent():
    # 1. Initialize custom environment
    env = TrafficIntersectionEnv()
    
    # Validate environment compliance with Gym API
    check_env(env, warn=True)
    
    # 2. Initialize PPO Agent
    # Multi-Layer Perceptron policy, appropriate for flat vector states
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, tensorboard_log="./ppo_traffic_tensorboard/")
    
    print("Starting Training...")
    # 3. Train the model
    model.learn(total_timesteps=100_000, progress_bar=True)
    
    # 4. Save the policy
    model.save("models/ppo_traffic_optimizer")
    print("Model saved to models/ppo_traffic_optimizer.zip")

if __name__ == "__main__":
    train_agent()