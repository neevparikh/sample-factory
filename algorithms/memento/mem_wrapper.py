import numpy as np

import gym
import torch
from gym.spaces import Discrete
from torch.distributions import Categorical

from utils.utils import log


class MemActionSpace(Discrete):
    def __init__(self, n, prior_probs):
        super().__init__(n)

        self.prior_probs = prior_probs
        assert n == len(self.prior_probs)


def split_env_and_memory_actions(actions, memory_size):
    num_memory_actions = memory_size
    env_actions = actions[:-num_memory_actions]
    memory_actions = actions[-num_memory_actions:]
    return env_actions, memory_actions


class MemWrapper(gym.core.Wrapper):
    def __init__(self, env, mem_size, mem_feature):
        super().__init__(env)

        self.mem_size = mem_size
        self.mem_feature = mem_feature

        memory_obs_size = self.mem_size * self.mem_feature
        memory_obs_space = gym.spaces.Box(low=-1e3, high=1e3, shape=[memory_obs_size])

        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = env.observation_space
        else:
            self.observation_space = gym.spaces.Dict(spaces=dict(obs=env.observation_space))

        self.observation_space.spaces['mem'] = memory_obs_space

        # two actions for each "memory" cell (noop, rewrite)
        memory_action_spaces = []
        priors = [0.5, 0.1, 0.05, 0.01]
        for i in range(self.mem_size):
            write_probability = priors[i] if i < len(priors) else 0.1
            memory_action_space = MemActionSpace(2, prior_probs=[1.0 - write_probability, write_probability])
            memory_action_spaces.append(memory_action_space)

        # modify the original action space
        if isinstance(env.action_space, gym.spaces.Tuple):
            self.action_space = gym.spaces.Tuple(list(env.action_space.spaces) + memory_action_spaces)
        else:
            self.action_space = gym.spaces.Tuple([env.action_space] + memory_action_spaces)

        self.last_mem_actions = None

    def render(self, mode='human', **kwargs):
        self.env.render(mode, **kwargs)

        if self.last_mem_actions is not None:
            for i, mem_action in enumerate(self.last_mem_actions):
                if mem_action > 0:
                    log.warning('Write to memory cell %d', i)

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        env_actions, memory_actions = split_env_and_memory_actions(actions, self.mem_size)
        self.last_mem_actions = memory_actions

        if len(env_actions) == 1:
            env_actions = env_actions[0]

        observation, reward, done, info = self.env.step(env_actions)
        return observation, reward, done, info