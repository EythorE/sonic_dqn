"""
Train Sonic using DQN
"""
import argparse
import yaml
from pathlib import Path
import functools 

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import gymnasium as gym
import retro
from gymnasium.wrappers.time_limit import TimeLimit

from cnn_feature_extractor import SonicCNN
from environment import SlowResponse, SonicDiscretizer
from dqn import DqnNetwork
from replay_memory import ReplayMemory
from logger import Logger



def make_env():
    max_episode_steps = cfg['environment']['max_episode_steps']
    n_action_repeats = cfg['environment']['n_action_repeats']
    frame_diff_length = cfg['environment']['frame_diff_length']
    scenario = cfg['environment']['scenario']
    render_mode = cfg['environment']['render_mode']

    env = retro.make(
            game="SonicTheHedgehog2-Genesis", state=retro.State.DEFAULT,
            scenario=scenario, render_mode=render_mode, record=False
            )
    env = TimeLimit(env, max_episode_steps=max_episode_steps) # adds truncated to returns
    env = SlowResponse(env, n_action_repeats, frame_diff_length)
    env = SonicDiscretizer(env)
    return env


def make_dqn(env):
    device = cfg['dqn']['device']
    max_grad_norm = cfg['dqn']['max_grad_norm']
    discount_rate = cfg['dqn']['discount_rate']
    dqnetwork = DqnNetwork(
            SonicCNN,
            optim_params = cfg['dqn']['optimizer'],
            observation_space=env.observation_space,
            n_actions=env.action_space.n,
            discount_rate=discount_rate,
            loss_fn=F.mse_loss,
            max_grad_norm=max_grad_norm,
            device=device
            )
    return dqnetwork


def record_episode(path, env, dqn, action_fn, logger, global_step):
    env.unwrapped.record_movie(path.as_posix())
    obs, info = env.reset()
    video = [obs]
    q_values = []
    done = truncated = False
    rewards = 0
    while not (done or truncated):
        action = action_fn(obs)
        if dqn.last_q_values is not None:
            q_values.append(dqn.last_q_values[0])
        obs, reward, done, truncated, info = env.step(action)
        video.append(obs)
        rewards += reward
    print("Recorded epsiode:", path.name, info, "\ncumulated rewards:", rewards)
    env.unwrapped.stop_record()

    video = np.stack(video)[None,:,[0,0,0]] # (B,T,C,H,W)
    video = video + 128  
    logger.tb.add_video("observations", video.astype('uint8'), global_step=global_step, fps=6)
    if q_values:
        q_values = np.stack(q_values)
        moves = [f"{i}({'-'.join(m)})" for i,m in enumerate(env.combos)]
        for move, values in zip(moves, q_values.T):
            logger.tb.add_histogram(f'q-values/{move}', q_values, global_step=global_step)
    return



def main():
    resume = cfg["resume"]
    name = cfg['log']['name']
    logdir = cfg['log']['logdir']
    log_frequency = cfg['log']['log_frequency']
    save_steps = cfg['log']['save_steps']
    
    n_steps = cfg['n_steps']
    training_interval = cfg['training_interval']
    replay_buffer_fill_samples = cfg['replay_buffer_fill_samples']
    replay_memory_size = cfg['replay_memory_size']
    copy_steps = cfg['copy_steps']
    eps_max = cfg['eps_max']
    eps_min = cfg['eps_min']
    eps_decay_steps = cfg['eps_decay_steps']
    batch_size = cfg['batch_size'] 


    logger = Logger(logdir=logdir, log_frequency=log_frequency, name=name, cfg=cfg)
    recordings_dir = logger.logdir / "recordings"
    recordings_dir.mkdir()
    weights_dir = logger.logdir / "weights"
    weights_dir.mkdir()
    
    env = make_env()
    dqnetwork = make_dqn(env)
    
    if resume:
        dqnetwork.load(resume)
        start_step = int(resume.stem)
        _epsilon = max(eps_min, eps_max - (eps_max-eps_min) * start_step/eps_decay_steps)
        action_fn = functools.partial(dqnetwork.afn_epsilon_greedy, epsilon=_epsilon)
    else:
        start_step = 0
        action_fn = dqnetwork.afn_random
    
    print("filling memory...", end=" ", flush=True)
    replay_memory = ReplayMemory(replay_memory_size)
    replay_memory.fill_replay_memory(replay_buffer_fill_samples, env, action_fn)
    print("done")
    
    obs, info = env.reset()
    done = False
    truncated = False
    for global_step in range(start_step, n_steps):

        if done or truncated: # game over, start again
            obs, info = env.reset()
            game_frames = 0
            total_episode_reward = 0
    
        # epsilon greedy -> choose action
        epsilon = max(eps_min, eps_max - (eps_max-eps_min) * global_step/eps_decay_steps)
        if np.random.rand() < epsilon:
            action = dqnetwork.afn_random()
        else:
            action = dqnetwork.afn_greedy(obs)
    
        # take step
        next_obs, reward, done, truncated, info = env.step(action)
        replay_memory.append((obs, action, reward, next_obs, 1.0 - done))
        obs = next_obs
    
        logged = logger.episode_step(reward, done, truncated, info, global_step=global_step)
        if logged:
            logger.tb.add_scalar("info/epsilon", epsilon, global_step)
            logger.tb.add_scalar("info/replay_memory", len(replay_memory), global_step)
    
        # optimize q_function
        if global_step % training_interval == 0:
            # perform network optimization steps only every training_interval steps
            # vectors: obs, action, reward, next_obs, continue
            memories = replay_memory.sample_memories(batch_size)
            loss_info = dqnetwork.training_step(memories)
            logger.training_step(loss_info, global_step)
    
            training_step = global_step // training_interval
            if training_step % copy_steps == 0:
                dqnetwork.copy_online_to_target()
    
        if global_step > 0 and \
                global_step % save_steps == 0 or \
                global_step == (n_steps-1):
            print(f"checkpoint : runtime {logger.uptime()}")
            # record episode should return some metrics to log!
            afn = functools.partial(dqnetwork.afn_epsilon_greedy, epsilon=epsilon)
            record_episode(logger.logdir / "recordings" / f"{global_step}.bk2", env, dqnetwork, afn, logger, global_step)
            dqnetwork.save(logger.logdir / "weights" / f"{global_step}.pt")
    
    env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Sonic the Hedgehog 2 - Emerald Hill Zone Part 1 DQN learning"
        )
    parser.add_argument(
            '--config', type=Path,
            default="./config.yaml", help='YAML configuration file.'
        )
    parser.add_argument(
            '--resume', type=Path,
            default=None, help='Checkpoint [step.pt] to resume training.'
        )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if "resume" not in cfg or args.resume:
        cfg["resume"] = args.resume

    main()
