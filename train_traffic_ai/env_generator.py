import json
from flatland.envs.rail_env import RailEnv
# from flatland.envs.line_generators import sparse_rail_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.observations import TreeObsForRailEnv

def generate_env(num_agents=5, width=25, height=25, seed=42, save_path="env.json"):
    line_generator = sparse_rail_generator(max_rails_between_cities=2,
                                           seed=seed,
                                           grid_mode=False
                                           )

    obs_builder = TreeObsForRailEnv(max_depth=2)

    env = RailEnv(width=width,
                  height=height,
                  # rail_generator=None,
                  # line_generator=line_generator,
                  rail_generator=line_generator,
                  number_of_agents=num_agents,
                  obs_builder_object=obs_builder)

    env.reset()
    env_dict = {
        "width": width,
        "height": height,
        "num_agents": num_agents,
        "seed": seed
    }

    with open(save_path, "w") as f:
        json.dump(env_dict, f, indent=4)

    print(f"✅ Environment generated and saved to {save_path}")
    return env

if __name__ == "__main__":
    generate_env()