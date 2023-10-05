import copy
import ctypes
import random
from typing import Any, List, Optional, Tuple, TypedDict, TypeVar, Union

import gymnasium as gym
import numpy as np


class HangmanEnv(gym.Env):
    def __init__(
        self,
        words: List[str],
        max_word_length: int = 10,
        num_lives: int = 10,
    ):
        super(HangmanEnv, self).__init__()
        
        self.max_word_length = max_word_length
        self.max_num_lives = num_lives
        
        obs_shape = np.concatenate([
            28 * np.ones(self.max_word_length),
            2 * np.ones(26)
        ])
        # obs_shape = 28 * np.ones(self.max_word_length)
        self.observation_space = gym.spaces.MultiDiscrete(obs_shape)
        
        self.action_space = gym.spaces.Discrete(26)
        
        self.words = []
        for word, orig in words:
            self.words.append((self.transform_word(word), orig))
        
        self.word = None
        
        self.state = None
        self.used_actions = None # 0 if used, 1 if not used
        self.num_lives = None
        
        self._option = None
        self._seed = 0
        
    def seed(self, seed):
        self._seed = seed

    def step(self, action: int):
        char = chr(action + 97)
        self.used_actions[action] = 0
        
        reward = 0
        correct = False
        for i in range(self.length):
            if self.state[i] == 0 and self.word[i] == char:
                correct = True
                self.state[i] = action + 1
                # reward += 1 / self.length
        
        if not correct:
            self.num_lives -= 1
        
        if self.num_lives <= 0:
            return self.get_state(), 0, True, False, {}
        elif np.all(self.state != 0):
            return self.get_state(), reward + 1, True, False, {}
        else:
            return self.get_state(), reward, False, False, {}

    def reset(self, seed=None, option=None):
        if option is not None:
            self.state, self.word = option
            self.state = self.transform_word(self.state)
            self.used_actions = np.ones(26)
        elif self._option is not None:
            self.state, self.word = self._option
            self.state = self.transform_word(self.state)
            self._option = None
            self.used_actions = np.ones(26)
        else:
            index = np.random.choice(len(self.words))
            self.state, self.word = self.words[index]
            self.set_used_actions()
        
        self.length = len(self.word)
        
        self.state = np.concatenate([
            self.state,
            np.zeros(self.max_word_length - self.length) - 1
        ])
        
        self.num_lives = self.max_num_lives
        
        return self.get_state(), {}

    def render(self, mode=None) -> None:
        state = self.state[:self.length]
        print(f"Current state: {''.join([chr(int(x) + 96) if x > 0 else '_' for x in state])}")
        
    def get_state(self):
        return np.concatenate([self.state + 1, self.used_actions])
        # return self.state + 1 # +1 to avoid -1
        
    def set_option(self, option):
        self._option = option
    
    def transform_word(self, word):
        word = np.array([ord(c) - 96 for c in word])
        word[word == ord('_') - 96] = 0
        return word
    
    def set_used_actions(self):
        self.used_actions = np.ones(26)
        for i in range(len(self.word)):
            if self.state[i] != 0:
                self.used_actions[self.state[i] - 1] = 0

if __name__ == "__main__":
    env = HangmanEnv([("_e__o", "hello"), ("t_st", "test")])

    while True:
        env.reset()
        env.render()
        done = False
        while not done:
            action = random.randint(0, 25)
            _, reward, done, _, _ = env.step(action)
            print(chr(action+97), reward)
            env.render()
            print()
        breakpoint()

