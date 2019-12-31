# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import datetime
import logging
import argparse

import gym
import ma_gym
from gym import wrappers, logger

from ma_gym.wrappers import Monitor

import tensorflow as tf

from tensorforce.agents import Agent, MultiAgent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# ENVIRONMENT = 'CartPole-v1'
ENVIRONMENT = 'PongDuel-v0'


def train_agent(env, num_episodes, max_episode_timesteps=5):
    # Create an OpenAI-Gym environment
    gym_env = gym.make(env)
    environment = Environment.create(
        environment='gym', level=gym_env,
        max_episode_timesteps=max_episode_timesteps)

    # Create a PPO agent
    agent = MultiAgent.create(
        agents=['ppo', 'ppo'], environment=environment
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)

    # Note this didnt work...
    #save_best_agent=os.path.join(os.getcwd(), 'best_cart_pole_ppo')

    # Start the runner
    runner.run(num_episodes=num_episodes)

    agent.save(
        directory=os.path.join(os.getcwd(), 'best_cart_pole_ppo'),
        filename='best-model',
        append_timestep=False
    )

    # Evaluating - according to https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html#training-and-evaluation
    # NOTE: this will not work together with save_best_agent
    #runner.run(num_episodes=10, evaluation=True)

    runner.close()


def run_from_agent(env):
    # TODO: use the proper
    agent_path = os.path.join(
        os.getcwd(), 'best_cart_pole_ppo', 'best-model.json')
    print(f"Evaluating agent: {agent_path}")

    # TODO: this require unique folder
    dt_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    outdir = f'/tmp/random-agent-results-{dt_str}'
    # # from gym:example/agents/ramdom_agents.py
    # gym_env = gym.make(env_id)
    # gym_env = wrappers.Monitor(gym_env, directory=outdir, force=True)
    # environment = Environment.create(gym_env)

    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level=env,
                                     visualize=True, visualize_directory=outdir)

    # Now create the new agent
    runner = Runner(agent=agent_path, environment=environment)

    runner.run(num_episodes=100, evaluation=True)
    runner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running the agent')
    parser.add_argument('--env', default=ENVIRONMENT,
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--pdb', action='store_true',
                        help='Trigger pdb on error')
    parser.add_argument('learn_or_run', type=str, help="(learn|run)")
    args = parser.parse_args()

    if args.learn_or_run == 'learn':
        if not args.pdb:
            train_agent(args.env, args.episodes)
        else:
            try:
                train_agent(args.env, args.episodes)
            except:  # noqa: E722
                import pdb
                import traceback
                import sys
                extype, value, tb = sys.exc_info()
                traceback.print_exc()
                pdb.post_mortem(tb)
    elif args.learn_or_run == 'run':
        run_from_agent(args.env)
    else:
        raise RuntimeError("Invalid option: {}".format(args.learn_or_run))
