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

from tensorforce.agents import Agent, MultiAgent
from tensorforce.execution import Runner
from tensorforce.core.layers import Layer
from tensorforce.environments import Environment

from test.unittest_base import UnittestBase
from test.unittest_environment import UnittestEnvironment


class MultiAgentEnvironment(UnittestEnvironment):
    """ A mock of the multi-agent environment

    The multi-agent return multiple states and actions spaces.
    """
    num_agents: int

    def __init__(self, states, actions, min_timesteps, num_agents):
        super(MultiAgentEnvironment, self).__init__(
            states, actions, min_timesteps)
        self.num_agents = num_agents
        self.random_states = self.__class__.random_states_function(
            states_spec=self.states_spec, actions_spec=self.actions_spec,
            num_agents=num_agents
        )

    def states(self):
        return [self.states_spec] * self.num_agents

    def actions(self):
        return [self.actions_spec] * self.num_agents

    @classmethod
    def random_states_function(cls, states_spec, actions_spec, num_agents=None):
        parent_state_func = super().random_states_function(states_spec,
                                                           actions_spec)
        return lambda: [parent_state_func()] * num_agents

    @classmethod
    def is_valid_actions_function(cls, actions_spec):
        parent_valid_func = super().is_valid_actions_function(actions_spec)
        return lambda actions, states: (
            isinstance(actions, list)
            and isinstance(states, list)
            and len(actions) == len(states)
            and all([
                parent_valid_func(a, s)
                for a, s in zip(actions, states)
            ])
        )

    def reset(self):
        _states = super(MultiAgentEnvironment, self).reset()
        # _states should already multiplied by random function above
        # TODO: we should tidying this up to ensuare all states is at one place
        assert isinstance(_states, list)
        return _states

    def execute(self, actions):
        _states, terminal, reward = super(
            MultiAgentEnvironment, self).execute(actions)
        c = self.num_agents
        # _states should already multiplied by random function above
        # TODO: we should tidying this up to ensuare all states is at one place
        assert isinstance(_states, list)
        assert not isinstance(terminal, list)
        assert not isinstance(reward, list)
        return self._states, [terminal]*c, [reward]*c


class TestMultiAgentRunner(UnittestBase, unittest.TestCase):
    """Test for MultiAgent Runner

    This is a test runner which runs a multiple version of agents, setting up
    to ensure that we can get up and running with multiple agents.

    A multiple agent environment will return an environment for each state.

    In tensorforce, there are 3 main entities:

    * Environment - same as before
    * MultiAgent - reads environment and apply specific mask to agent
    * MultiRunner - apply runners to ensure that we can synchronize the running

    """

    require_observe = True
    # TODO: just map to one agent for now to ensure this runs
    num_agents = 4
    # TODO: debug between correct / wrong version
    use_multi = True

    def test_multi_agent_runner(self):
        self.start_tests(name='multi-agent-runner')

        Layer.layers = None

        # Mock up multienvironment process
        states = deepcopy(self.__class__.states)
        actions = deepcopy(self.__class__.actions)
        min_timesteps = self.__class__.min_timesteps
        # Note that this is just a normal environemnt, but return multiple
        # states in return. (See mock above)
        if self.use_multi:
            environment = MultiAgentEnvironment(
                states=states, actions=actions, min_timesteps=min_timesteps,
                num_agents=self.__class__.num_agents
            )
            self.assertEqual(len(environment.states()), self.num_agents)
            self.assertEqual(len(environment.actions()), self.num_agents)
        else:
            environment = UnittestEnvironment(
                states=states, actions=actions, min_timesteps=min_timesteps,
            )
        environment = Environment.create(
            environment=environment, max_episode_timesteps=5)

        # Now Mock up the multi-agent environment
        config = dict(api_functions=['reset', 'act', 'observe'])
        # TODO: provide the mapping for multiagent
        # This allow running MultipleAgent as one agent, that would be cooL!
        if self.use_multi:
            agent = MultiAgent.create(
                agents=[deepcopy(self.__class__.agent)
                        for _ in range(self.__class__.num_agents)],
                environment=environment, config=config)
        else:
            agent = Agent.create(agent=self.__class__.agent,
                                 environment=environment, config=config)
        runner = Runner(agent=agent, environment=environment)

        # XXX: ensure that we run the agent, with each agent should:
        # 1. be trained to the environment and see a wrap for the agent
        # 2. ensure that it can be seen to train for the setup of the place
        #
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
