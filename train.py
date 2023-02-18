""" Implementation of an on-policy Temporal Difference soft-policy deep SARSA reinforcement learning agent
utilizing bayesian value networks. The purpose of this project is to learn an optimal policy for various
OpenAI Gym environments using pixel-to-control mapping.

References:
    [1] ”Playing Atari with Deep Reinforcement Learning”, Minh, Volodymyr, et al (https://arxiv.org/pdf/1312.5602.pdf)
    [2] "Deep Reinforcement Learning with Experience Replay Based on SARSA", Dongbin Zhao, Haitao Wang, Kun Shao and Yuanheng Zhu (https://www.researchgate.net/publication/313803199_Deep_reinforcement_learning_with_experience_replay_based_on_SARSA)

Authors:
    Frederik Gram Kortegaard,
    Lasse Usbeck Andresen

"""

import os
import cv2
import gym
import time
import torch
import logging
import argparse
import itertools
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import json
from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
import random

from matplotlib.ticker import MaxNLocator

from agent import DDPG
from prefigs import configs

random.seed(0)


def save_metrics(metrics, args):
    """Save the metrics to a file"""
    current_time = str(time.time())

    if not os.path.isdir(current_time):
        os.mkdir(current_time)

    # Save full matrics
    np.save(os.path.join(current_time, f"metrics_{args.environment}"), metrics)

    # Quicksave a readable version of the arguments
    t = vars(args)
    t = {k: str(v) for k, v in t.items() if not k.startswith("_")}
    json.dump(t, open(os.path.join(current_time, "args.json"), "w"))

    # Save plots
    if args.plot:
        plt.savefig(os.path.join(current_time, f"metrics_{args.environment}.png"))


def run_agent(args):

    print("Using device: {}".format("CUDA" if args.use_cuda else "CPU"))

    # Fetch the premade configurations for the given environment
    try:
        prefig = configs[args.environment]
    except KeyError:
        raise KeyError(
            "Environment '{args.environment}' not found. Please choose from the following: {}".format(
                configs.keys()
            )
        )

    preprocesser = prefig["preprocessor"]

    env = gym.make(prefig["environment_name"])
    # env.seed(args.seed)

    # Initialize the agent
    agent = DDPG(
        prefig["state_dimensionality"],
        prefig["action_dimensionality"],
        args.memory_capacity,
        args.lr_model,
        args.kl_weight,
        args.gamma,
        args.n_step,
        args.use_cuda,
        args.frames,
        args.params_update,
    )

    # Load the model if we are testing
    if args.test_model is not None:
        if args.verbose:
            print("Loading model from {}".format(args.test_model))
            agent.load_model(args.test_model)
            print("Model loaded")
        else:
            agent.load_model(args.test_model)

    # Setup Monitoring
    metrics = {
        "episodes": list(),
        "prefig": {  # we can't store the preprocessor callable so we manually unpack the relevant fields
            "environment_name": prefig["environment_name"],
            "state_dimensionality": prefig["state_dimensionality"],
            "action_dimensionality": prefig["action_dimensionality"],
        },
    }

    # Create subplots for the learning curve and the loss
    if args.plot:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
        fig.canvas.set_window_title(
            f'Deep Sarsa training on {prefig["environment_name"]} for {args.max_episodes} episodes with {args.max_steps if args.max_steps is not None else "unlimited"} steps'
        )

        ax1.set_ylabel("Learning Curve")
        ax1.set_xlabel("Episode")

        ax2.set_ylabel("Loss 1")
        ax2.set_xlabel("Episode")

        ax3.set_ylabel("Loss 2")
        ax3.set_xlabel("Episode")

        # Force the x-axis ticks to be integers
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

    # As our model expects actions as a probability distribution,
    # we synthetically create one based on the number of possible actions
    # for use when we directly sample the environments action space,
    # and not recieve a probability distribution from our model
    synthetic_probabilities = np.array(
        [1 / prefig["action_dimensionality"]] * prefig["action_dimensionality"],
        dtype=np.float32,
    )

    try:
        stacked_frames = deque(
            [np.zeros((84, 84), dtype=np.int32) for i in range(args.frames)],
            maxlen=args.frames,
        )
        # Start the training loop
        for i in range(args.max_episodes):

            # Store for metrics
            losses_in_episode = []

            # Randomly initialize the state
            # env.seed(i)

            # Reset the environment
            state = env.reset()

            # Manipulate the state to be in the correct format
            state = preprocesser(state)
            state = np.expand_dims(state, axis=2)
            state = np.transpose(state, (2, 1, 0))
            state[0] = [1]

            state, stacked_frames = agent.stack_frames(
                stacked_frames, state, True, args.frames
            )
            state = state.detach().numpy()

            episode_reward = 0

            steps_since_reward = 0

            # Start the episode
            last_lives = 3

            for j in (
                range(args.max_steps)
                if args.max_steps is not None
                else itertools.count(start=0)
            ):

                if args.render or args.slow_render:

                    if args.imwrite:
                        # if j % args.frame_skip == 0:
                        img = env.render(mode="rgb_array")
                        cv2.imwrite(f"render.png", img)
                    else:
                        env.render()

                    if args.slow_render:
                        time.sleep(0.05)

                # We're utilizing frame skipping as the gameplay has a lot of superfluous frames
                if j % args.frame_skip == 0:

                    # Use an epsilon greedy policy to select an action either
                    # randomly or from the model. if we're testing,
                    # there is no need to explore.
                    if (
                        np.random.uniform(0, 1) >= args.epsilon
                        or args.test_model != None
                    ):
                        action = agent.get_action(state)
                        action = np.squeeze(action)
                        action_choice = np.argmax(action)

                    # Explore by sampling from the environment's action space
                    else:
                        action_choice = env.action_space.sample()
                        action = synthetic_probabilities

                # Take action
                next_state, reward, done, info = env.step(action_choice)

                # Manipulate the state to be in the correct format
                next_state = preprocesser(next_state)
                next_state = np.expand_dims(next_state, axis=2)
                next_state = np.transpose(next_state, (2, 1, 0))
                # next_state = np.expand_dims(next_state, axis=0)
                next_state[0] = [1]
                next_state, stacked_frames = agent.stack_frames(
                    stacked_frames, next_state, False, args.frames
                )

                next_state = next_state.detach().numpy()

                if (
                    prefig["environment_name"] == "SpaceInvaders-v0"
                    and last_lives != info["ale.lives"]
                ):
                    steps_since_reward += 1
                    last_lives -= 1
                    reward -= 1

                if done and prefig["environment_name"] == "SpaceInvaders-v0":
                    reward -= 1
                    steps_since_reward += 1
                elif reward == 0:
                    steps_since_reward += 1
                else:
                    steps_since_reward = 0

                # reward -= steps_since_reward * 0.00001
                reward = np.clip(reward, -1, 1)

                # Store transition
                agent.store_transition(state, action, reward, next_state, action_choice)

                # Attempt to batch train if we have enough samples and we're not testing
                if (
                    len(agent.memory) > args.memory_capacity - 1
                    and args.test_model == None
                    and j % args.frame_skip == 0
                ):

                    loss = agent.learn(args.batch_size)
                    loss = loss

                    losses_in_episode.append((loss[0].item(), loss[1]))
                    # Decay the epsilon value
                    args.epsilon = max(
                        args.epsilon * args.epsilon_decay, args.epsilon_min
                    )

                state = next_state
                episode_reward += reward

                # Calculate and log various live-metrics
                if args.verbose:
                    average_reward = (
                        sum([ep["reward"] for ep in metrics["episodes"]])
                        / len(metrics["episodes"])
                        if len(metrics["episodes"]) > 0
                        else 0.0
                    )

                    if j % 20 == 0:
                        print(
                            f"Episode: {i}/{args.max_episodes} | Step: {j}{f'/{args.max_steps}' if args.max_steps is not None else ''} | Reward: {episode_reward} | Action: {action_choice} | Avg Reward: {average_reward:.4f} | Epsilon: {args.epsilon * 100}%"
                        )

                # If the episode is done, break
                if done:
                    break

            # Save the episode reward and calculate
            # the cumulative reward of all the episodes
            metrics["episodes"].append(
                {
                    "reward": episode_reward,
                    "cumulative_reward": episode_reward
                    + metrics["episodes"][-1]["cumulative_reward"]
                    if len(metrics["episodes"]) > 0
                    else episode_reward,
                    "losses": losses_in_episode,
                }
            )

            # Plot metrics live
            if args.plot:

                # Live plot the learning rate
                ax1.scatter(
                    i,
                    metrics["episodes"][-1]["cumulative_reward"] / (i + 1),
                    c="g",
                )

                ax1.set_ylim(
                    min(
                        [
                            ep["cumulative_reward"] / (enum + 1)
                            for enum, ep in enumerate(metrics["episodes"])
                        ]
                    ),
                    max(
                        [
                            ep["cumulative_reward"] / (enum + 1)
                            for enum, ep in enumerate(metrics["episodes"])
                        ]
                    ),
                )

                ax1.set_xlim(0, i + 2)

                # Live plot the loss
                if args.imwrite:
                    plt.savefig(f"plot.png")
                else:
                    plt.pause(0.001)

        # Save the metrics
        if args.save:
            save_metrics(metrics, args)

        # In case we're calling this from someplace else
        # it's nice to be able to store this
        return agent, metrics

    except KeyboardInterrupt:
        logging.warning(
            f"Keyboard interrupt detected. {'' if not args.backup else 'Saving backup...'}"
        )
    except Exception as e:
        logging.error(
            f"Exception detected: {str(e)}.{'' if not args.backup else 'Saving backup...'}"
        )
        raise e
    finally:

        # If any error occurs we save all relevant information in case
        # we want to resume training or use the debugging information
        if args.backup:
            save_metrics(metrics, args)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = argparse.ArgumentParser()
    args.add_argument(
        "--environment",
        type=str,
        default="space_invaders",
        choices=configs.keys(),
    )
    args.add_argument("--max_episodes", type=int, default=1000)
    args.add_argument("--max_steps", type=int, default=None)

    args.add_argument("--memory_capacity", type=int, default=100000)

    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--epsilon", type=float, default=1.0)

    args.add_argument("--epsilon_decay", type=float, default=0.995)
    args.add_argument("--epsilon_min", type=float, default=1e-2)

    args.add_argument("--lr_model", type=float, default=2.5e-4)
    args.add_argument("--seed", type=int, default=1234)
    args.add_argument("--kl_weight", type=float, default=0.1)
    args.add_argument("--gamma", type=float, default=0.9)

    args.add_argument("--save", action="store_true")
    args.add_argument("--plot", action="store_true")
    args.add_argument("--verbose", action="store_true", default=True)
    args.add_argument("--use_cuda", action="store_true", default=False)

    args.add_argument("--frame_skip", type=int, default=3)
    args.add_argument("--test_model", type=str, default=None)

    args.add_argument("--backup", action="store_true", default=False)

    args.add_argument("--n_step", type=int, default=4)
    args.add_argument("--frames", type=int, default=3)

    args.add_argument("--params_update", type=int, default=150)

    args.add_argument("--imwrite", action="store_true", default=False)

    render_arguments = args.add_mutually_exclusive_group(required=False)
    render_arguments.add_argument("--render", action="store_true")
    render_arguments.add_argument("--slow_render", action="store_true")

    args = args.parse_args()

    agent, metrics = run_agent(args)
