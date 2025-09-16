import ray
from ray.rllib.algorithms.ppo import PPO
from train_ppo import create_original_flatland_env

def evaluate_model(checkpoint="ppo_flatland", episodes=5):
    algo = PPO.from_checkpoint(checkpoint)
    env = create_original_flatland_env({})
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 100:
            action = {}
            for agent in env.agents:
                action[agent] = algo.compute_single_action(obs[agent])
            obs, rewards, dones, info = env.step(action)
            total_reward += sum(rewards.values())
            done = all(dones.values())
            steps += 1
        print(f"Episode {ep+1}: total_reward={total_reward}, steps={steps}")
    env.close()

if __name__ == "__main__":
    evaluate_model()