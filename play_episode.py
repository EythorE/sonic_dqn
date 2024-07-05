"""
Script to play an episode using trained dqn action function.
"""

import argparse
from pathlib import Path
import retro
from gymnasium.wrappers.time_limit import TimeLimit
from environment import SlowResponse, SonicDiscretizer
from cnn_feature_extractor import SonicCNN
from dqn import DqnNetwork

scenario="./scenario.json"
max_episode_steps = 3*60*60 # 90 # stop episode after number of seconds
n_action_repeats = 20 # repeat action for a number of frames
frame_diff_length = 5 # motion blur
n_actions = 10

def make_env(
        game="SonicTheHedgehog2-Genesis", state=retro.State.DEFAULT,
        scenario=scenario, render_mode=None, record=False
    ):
    env = retro.make(
            game=game, state=state,
            scenario=scenario, render_mode=render_mode, record=record
        )
    env = TimeLimit(env, max_episode_steps=max_episode_steps) # adds truncated to returns
    env = SlowResponse(env, n_action_repeats, frame_diff_length)
    env = SonicDiscretizer(env)
    return env

def play(env, weights, strategy, epsilon, record):
    dqnetwork = DqnNetwork(
            SonicCNN,
            observation_space=env.observation_space,
            n_actions=10,
            optim_params=None,
            discount_rate=None,
            loss_fn=None,
            max_grad_norm=None,
            device='cpu'
            )
    dqnetwork.load(weights)
    action_fn = {
            "random": dqnetwork.afn_random,
            "greedy": dqnetwork.afn_greedy,
            "epsilon_greedy": lambda obs: dqnetwork.afn_epsilon_greedy(obs, epsilon)
            }[strategy]
    if record:
        env.unwrapped.record_movie(record.as_posix())
    obs, info = env.reset()
    done = truncated = False
    rewards = 0
    time_steps = 0
    while not (done or truncated):
        action = action_fn(obs) 
        obs, reward, done, truncated, info = env.step(action)
        rewards += reward
        print(info)
        print(f"reward: {reward:4} cumulated_rewards: {rewards}")
        time_steps += n_action_repeats
    if record:
        env.unwrapped.stop_record()
    env.close()
    print("info:", info)
    print("cumulated_rewards:", rewards)
    print(f"time: {time_steps / 60:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play episode with specified strategy using trained dqn.")
    parser.add_argument("weights", type=Path, help="Path to the model weights file (e.g., path/to/weights.pt).")
    parser.add_argument("strategy", choices=["random", "greedy", "epsilon_greedy"], help="Strategy for action selection.")
    parser.add_argument("--epsilon", type=float, default=None, help="Epsilon value for epsilon_greedy strategy (e.g., 0.1).")
    parser.add_argument("--record", type=Path, default=False, help="Record a bk2 movie. (Playback and convert to mp4 with 'python -m retro.scripts.playback_movie file.bk2')")
    parser.add_argument("--scenario", type=str, default='./scenario.json', help="scenario.json file to use (or 'scenario' or 'xpos')")

    args = parser.parse_args()

    print(f"Using weights from: {args.weights}")
    print(f"Selected strategy: {args.strategy}")

    if args.strategy == "epsilon_greedy":
        if args.epsilon is None:
            parser.error("epsilon_greedy strategy requires --epsilon value.")
        print(f"Epsilon value: {args.epsilon}")

    env = make_env(render_mode="human", scenario=args.scenario)
    play(env, args.weights, args.strategy, args.epsilon, args.record)


if __name__ == "__main__":
    main()
