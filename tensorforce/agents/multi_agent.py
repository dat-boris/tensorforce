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

from collections import OrderedDict
import importlib
import json
import os
import random
import time
import logging

import numpy as np

from tensorforce.environments import Environment
from tensorforce.agents.agent import Agent
from tensorforce import util, TensorforceError


# We would rather not inherit Environment and use composition here,
# but various agent depends on the `isinstance` to identify code path
class MultiAgentEnvironmentWrapper(Environment):
    """A Wrapper around existing environment

    Allow returning only part of the states of the environment
    """
    agent_idx: int

    def __init__(self, environment, agent_idx):
        super().__init__()

        if isinstance(environment, MultiAgentEnvironmentWrapper):
            raise TensorforceError.unexpected()
        self.environment = environment
        self.agent_idx = agent_idx

    def __str__(self):
        return str(self.environment)

    def max_episode_timesteps(self):
        return self.environment.max_episode_timesteps()

    def states(self):
        return self.environment.states()[self.agent_idx]

    def actions(self):
        return self.environment.actions()[self.agent_idx]

    def close(self):
        # Should not execute write on wrapper
        raise TensorforceError.unexpected()

    def reset(self):
        # Should not execute write on wrapper
        raise TensorforceError.unexpected()

    def execute(self, actions):
        # Should not execute write on wrapper
        raise TensorforceError.unexpected()

# We would rather not inherit Agent and use composition here,
# but various agent depends on the `isinstance` to identify code path

class MultiAgent(Agent):
    """Responsible for multiplexing environment requests to multiple agents
    """

    @staticmethod
    def create(agents, environment=None, **kwargs):
        assert isinstance(environment, Environment)
        return MultiAgent([
            Agent.create(
                agent=agent,
                environment=MultiAgentEnvironmentWrapper(
                    environment, agent_idx),
                **kwargs
            ) for agent_idx, agent in enumerate(agents)
        ])

    @staticmethod
    def load(directory, filename=None, environment=None, **kwargs):
        raise NotImplementedError()

    def __init__(self, agents):
        self.agents = agents
        self.terminal = [False for _ in agents]

    def __str__(self):
        return str([str(a) for a in self.agents])

    def __getattr__(self, name):
        err_msg = "Accessing unknown wrapper attribute: {}".format(name)
        # logging.error(err_msg)
        # return getattr(self.agents[0], name)
        raise NotImplementedError(err_msg)

    @property
    def num_agents(self):
        return len(self.agents)

    def initialize(self):
        for a in self.agents:
            a.initialize()

    def close(self):
        for a in self.agents:
            a.close()

    def reset(self):
        for a in self.agents:
            a.reset()

    def act(self, states, *args, query=None, **kwargs):
        """
        Returns:
            (dict[action], plus optional list[str]): Dictionary containing action(s), plus queried
            tensor values if requested.
        """

        assert len(states) == self.num_agents, (len(states), self.num_agents)
        # If your multi-agent have muti-wrapped states
        assert not isinstance(states[0], list), "You may have double wrapped"
        assert query is None, "Cannot return action query yet"

        return [
            a.act(states[agent_idx], *args, **kwargs)
            for agent_idx, a in enumerate(self.agents)
            if not self.terminal[agent_idx]
        ]

    def observe(self, reward, terminal, parallel=0, query=None,
                **kwargs):
        assert isinstance(reward, list)
        assert isinstance(terminal, list)
        assert len(reward) == self.num_agents, (len(reward), self.num_agents)
        assert len(terminal) == self.num_agents, \
            (len(terminal), self.num_agents)
        if query is not None:
            raise NotImplementedError()
        if parallel:
            raise NotImplementedError()

        update_performed = [False for _ in range(self.num_agents)]
        for agent_idx, a in enumerate(self.agents):
            if not self.terminal[agent_idx]:
                update_performed[agent_idx] = a.observe(
                    reward[agent_idx], terminal[agent_idx], **kwargs)

            if terminal[agent_idx]:
                self.terminal[agent_idx] = True
        return any(update_performed)

    def save(self, directory=None, filename=None, append_timestep=True):
        raise NotImplementedError("Not supported yet")

    def restore(self, directory=None, filename=None):
        raise NotImplementedError("Not supported yet")

    def get_output_tensors(self, function):
        raise NotImplementedError("Not supported yet")

    def get_query_tensors(self, function):
        raise NotImplementedError("Not supported yet")

    def get_available_summaries(self):
        raise NotImplementedError("Not supported yet")

    def should_stop(self):
        should_stop_flags = [a.should_stop() for a in self.agents]
        return all(should_stop_flags)
