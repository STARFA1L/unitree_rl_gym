# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os
# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

# import numpy as np
# import torch


# def play(args):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     # override some parameters for testing
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
#     # env_cfg.env.num_envs = 1
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.mesh_type = 'trimesh'
#     # env_cfg.terrain.num_rows = 5
#     # env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.terrain.selected = True
#     env_cfg.terrain.dump=False
#     env_cfg.terrain.rebuild = False
#     env_cfg.terrain.terrain_config_path = None #"dump_2025-04-03_13-37-31.yaml"
#     # {'type':'random_uniform_terrain', 'min_height':-0.05, 'max_height':0.05, 'step':0.005, 'downsampled_scale':0.2}
#     # {'type':'sloped_terrain', 'slope':1}
#     # {'type':'pyramid_sloped_terrain', 'slope':0.2, 'platform_size':3.}
#     # {'type':'rough_sloped_terrain', 'slope':0.2, 'platform_size':3.}
#     # {'type':'discrete_obstacles_terrain', 'max_height':0.15, 'min_size':1., 'max_size':2., 'num_rects':20, 'platform_size':3.}
#     # {'type':'wave_terrain', 'num_waves':1, 'amplitude':1.}
#     # {'type':'stairs_terrain', 'step_width':0.5, 'step_height':0.05}
#     # { 'type': 'pyramid_stairs_terrain', 'step_width': 0.31, 'step_height': -0.01, 'platform_size': 3.}
#     # {'type':'stepping_stones_terrain', 'stone_size':0.825, 'stone_distance':0.1, 'max_height':0., 'platform_size':4., 'depth':-10}
#     # {'type':'gap_terrain', 'gap_size':0.5, 'platform_size':3.}
#     # {'type':'pit_terrain', 'depth':0.5, 'platform_size':4.}

#     # # extreme(0402 update):
#     # {'type':'pyramid_stairs_terrain', 'step_width': 0.31, 'step_height': -0.07, 'platform_size': 3.}
#     # {'type':'discrete_obstacles_terrain', 'max_height':0.07, 'min_size':1., 'max_size':2., 'num_rects':20, 'platform_size':3.}
#     # {'type':'rough_sloped_terrain', 'slope':0.5, 'platform_size':3.}
#     # #zqr
#     # {'type':'wave_terrain', 'num_waves':6, 'amplitude':0.15}
#     # {'type':'random_uniform_terrain', 'min_height':-0.15, 'max_height':0.15, 'step':0.01, 'downsampled_scale':0.2}

#     # #1000itr:
#     # {'type':'pyramid_sloped_terrain', 'slope':0.36, 'platform_size':3.}
#     # {'type':'rough_sloped_terrain', 'slope':0.36, 'platform_size':3.}
#     env_cfg.terrain.terrain_kwargs = {'type':'rough_sloped_terrain', 'slope':0.04, 'platform_size':3.}

#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     env_cfg.env.test = True

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     obs = env.get_observations()
#     # load policy
#     train_cfg.runner.resume = True
#     ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#     policy = ppo_runner.get_inference_policy(device=env.device)
    
#     # export policy as a jit module (used to run it from C++)
#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#         print('Exported policy as jit script to: ', path)

#     for i in range(10*int(env.max_episode_length)):
#         actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())

# if __name__ == '__main__':
#     EXPORT_POLICY = True
#     RECORD_FRAMES = False
#     MOVE_CAMERA = False
#     args = get_args()
#     play(args)


##0417 update: original
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)