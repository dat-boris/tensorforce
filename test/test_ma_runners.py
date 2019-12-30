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

import copy
import time
import unittest
from copy import deepcopy

from tensorforce.agents import Agent
from tensorforce.execution import ParallelRunner, Runner
from tensorforce.core.layers import Layer
from tensorforce.environments import Environment

from test.unittest_base import UnittestBase
from test.unittest_environment import UnittestEnvironment


class MultiAgentEnvironment(UnittestEnvironment):
    """ A mock of the multi-agent environment

    The multi-agent return multiple states and actions spaces.
    """
    agent_count = 4

    # def __init__(self, states, actions, min_timesteps):
    #     pass

    def states(self):
        # XXX: do we want to mock multiple of this
        return self.states_spec

    def actions(self):
        return self.actions_spec

    # @classmethod
    # def random_states_function(cls, states_spec, actions_spec=None):
    # def random_state_function(cls, state_spec):
    #     return random_state

    # @classmethod
    # def random_mask(cls, action_spec):
    #     return mask

    # @classmethod
    # def is_valid_actions_function(cls, action_spec):
    # def is_valid_action_function(cls, action_spec):
    #     return is_valid

    def reset(self):
        _states = super(MultiAgentEnvironmnet, self).reset()
        return [states] * self.agent_count

    def execute(self, actions):
        _states, terminal, reward = super(
            MultiAgentEnvironmnet, self).execute(actions)
        count = self.agent_count
        # XXX: is terminal multiple?
        return [self._states]*count, [terminal]*count, [reward]*count

class TestMultiAgentRunner(UnittestBase, unittest.TestCase):

    require_observe = True
    num_agents = 4

    def test_multi_agent_runner(self):
        self.start_tests(name='multi-agent-runner')

        Layer.layers = None

        # Prepare multiple agents
        # XXX: pass in a mock environment that is multi-agent
        # agent1, environment = self.prepare(
        #     update=dict(unit='episodes', batch_size=1), parallel_interactions=2
        # )
        # # XXX: can we just copy the agent?
        # agent2 = copy.deepcopy(agent1)

        # Let's not do this!  Let's filter out individually
        # NOTE: this is simplified from self.prepare from unittest_base
        states = deepcopy(self.__class__.states)
        actions = deepcopy(self.__class__.actions)
        min_timesteps = self.__class__.min_timesteps
        # environment = MultiAgentEnvironment(
        #     states=states, actions=actions, min_timesteps=min_timesteps,
        #     num_agents=self.__class__.num_agents
        # )

        environment = UnittestEnvironment(
            states=states, actions=actions, min_timesteps=min_timesteps,
        )
        environment = Environment.create(environment=environment, max_episode_timesteps=5)

        agent = deepcopy(self.__class__.agent)
        config = dict(api_functions=['reset', 'act', 'observe'])
        # TODO: 1. do I acatually want a multi-agent runner instead?
        # I want to enmsure that
        agent = Agent.create(agent=agent, environment=environment, config=config)

        runner = Runner(agent=agent, environment=environment)

        # XXX: ensure that we run the agent, with each agent should:
        # 1. be trained to the environemnt and see a wrap for the agent
        # 2. ensure that it can be seen to train for the setup of the place
        # runner.run(
        #     num_episodes=1, callback=callback,
        #     callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        # )

        # XXX: to do I think we dont need to setup the setup
        # self.is_callback1 = False
        # self.is_callback2 = False

        # def callback1(r):
        #     self.is_callback1 = True

        # def callback2(r):
        #     self.is_callback2 = True

        runner.run(
            num_episodes=1,
            #callback=[callback1, callback2],
            #callback_timestep_frequency=callback_timestep_frequency, use_tqdm=False
        )
        runner.close()

        #self.finished_test(assertion=(self.is_callback1 and self.is_callback2))
        self.finished_test()
