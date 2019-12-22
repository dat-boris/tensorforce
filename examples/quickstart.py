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

import gym
from gym import wrappers, logger

import tensorflow as tf

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def main():
    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level='CartPole-v1')

    # Create a PPO agent
    agent = Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        optimization_steps=5,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        # Critic
        critic_network='auto',
        critic_optimizer=dict(
            optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None, recorder=None
    )

    # Initialize the runner
    runner = Runner(agent=agent, environment=environment)
    # Note this didnt work...
    #save_best_agent=os.path.join(os.getcwd(), 'best_cart_pole_ppo')

    # Start the runner
    runner.run(num_episodes=10)

    agent.save(
        directory=os.path.join(os.getcwd(), 'best_cart_pole_ppo'),
        filename='best-model',
        append_timestep=False
    )

    # Evaluating - according to https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html#training-and-evaluation
    # NOTE: this will not work together with save_best_agent
    #runner.run(num_episodes=10, evaluation=True)

    runner.close()


def run_from_agent():
    # TODO: use the proper
    agent_path = os.path.join(
        os.getcwd(), 'best_cart_pole_ppo', 'best-model.json')
    print(f"Evaluating agent: {agent_path}")

    env_id = 'CartPole-v1'
    # TODO: this require unique folder
    dt_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    outdir = f'/tmp/random-agent-results-{dt_str}'
    # # from gym:example/agents/ramdom_agents.py
    # gym_env = gym.make(env_id)
    # gym_env = wrappers.Monitor(gym_env, directory=outdir, force=True)
    # environment = Environment.create(gym_env)

    # Create an OpenAI-Gym environment
    environment = Environment.create(environment='gym', level=env_id,
                                     visualize=True, visualize_directory=outdir)

    # Now create the new agent
    runner = Runner(agent=agent_path, environment=environment)

    runner.run(num_episodes=100, evaluation=True)
    runner.close()


if __name__ == '__main__':
    # main()
    run_from_agent()
