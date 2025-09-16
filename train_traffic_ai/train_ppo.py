import os
import ray
import numpy as np
from collections import deque
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
#from flatland.envs.observations.tree_obs import Node

# ----------------------------------------------------
# 1. Flatten observation handling Node objects
# ----------------------------------------------------
def flatten_observation(node, max_depth: int, observation_radius: int) -> np.ndarray:
    flat_obs = []
    queue = deque([(node, 0)])
    visited_nodes = set()

    while queue:
        current_node, depth = queue.popleft()

        if current_node is None or depth > max_depth:
            continue

        node_id = id(current_node)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)

        flat_obs.extend([
            v if v is not None and v != np.inf else 0 for v in [
                getattr(current_node, 'dist_min_to_target', 0),
                getattr(current_node, 'dist_other_agent_encountered', 0),
                getattr(current_node, 'dist_other_target_encountered', 0),
                getattr(current_node, 'num_agents_same_direction', 0),
                getattr(current_node, 'num_agents_opposite_direction', 0),
                getattr(current_node, 'num_agents_malfunctioning', 0),
                getattr(current_node, 'speed_min_fractional', 0),
                getattr(current_node, 'num_agents_ready_to_depart', 0)
            ]
        ])

        for child in getattr(current_node, 'childs', {}).values():
            if hasattr(child, 'dist_min_to_target'):
                queue.append((child, depth + 1))

    expected_size = 252
    if len(flat_obs) < expected_size:
        flat_obs.extend([0] * (expected_size - len(flat_obs)))

    return np.array(flat_obs[:expected_size], dtype=np.float32)

# ----------------------------------------------------
# 2. Flatland environment factory
# ----------------------------------------------------
def create_original_flatland_env(env_config):
    obs_builder = TreeObsForRailEnv(max_depth=2)
    env = RailEnv(
        width=25,
        height=25,
        rail_generator=sparse_rail_generator(
            max_num_cities=3,
            seed=42,
            grid_mode=False,
            max_rails_between_cities=2
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=1,
        obs_builder_object=obs_builder
    )
    return env

# ----------------------------------------------------
# 3. Gymnasium wrapper
# ----------------------------------------------------
class FlatlandGymnasiumWrapper(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self._env = create_original_flatland_env(env_config)
        self.action_space = Discrete(5)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(252,), dtype=np.float32)
        self.max_depth = self._env.obs_builder.max_depth

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(regenerate_rail=True, regenerate_schedule=True)
        print("Observation type:", type(obs_dict[0]))  # Debug
        flat_obs = flatten_observation(obs_dict[0], self.max_depth, 10)
        return flat_obs, info_dict

    def step(self, action):
        obs_dict, reward_dict, done_dict, _, info_dict = self._env.step({0: action})
        flat_obs = flatten_observation(obs_dict[0], self.max_depth, 10)
        reward = reward_dict[0]
        terminated = done_dict['__all__']
        truncated = False
        return flat_obs, reward, terminated, truncated, info_dict

# ----------------------------------------------------
# 4. Register environment
# ----------------------------------------------------
register_env("flatland_gym_env", lambda config: FlatlandGymnasiumWrapper(config))

# ----------------------------------------------------
# 5. PPO training
# ----------------------------------------------------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment("flatland_gym_env")
        .env_runners(num_env_runners=1)
        .training(
            train_batch_size=2048,
            lr=5e-5
        )
        .resources(num_gpus=0)
        .framework("torch")
        .evaluation(evaluation_interval=None)
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            }
        )
    )

    algo = config.build_algo()

    print("\n--- Starting Training ---")
    for i in range(10):
        result = algo.train()
        print(
            f"Iter: {i+1}, "
            f"Mean Reward: {result.get('episode_reward_mean', float('nan')):.2f}, "
            f"Min Reward: {result.get('episode_reward_min', float('nan')):.2f}, "
            f"Max Reward: {result.get('episode_reward_max', float('nan')):.2f}, "
            f"Len: {result.get('episode_len_mean', float('nan')):.2f}"
        )

    # Save the model with an absolute path
    save_path = os.path.abspath("ppo_flatland_model")
    algo.save(save_path)
    print(f"\n✅ Model saved to {save_path}/")

    ray.shutdown()
