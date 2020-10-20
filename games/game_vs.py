import numpy as np

from .abstract_game import AbstractGame

def make_vs(game, observation_shape):
  class Game(AbstractGame):
    def __init__(self, seed=None):
      self.envs = [game(seed), game(seed)]
      self.observation_shape = observation_shape
      self.reset()

    def to_play(self):
      return self.player

    def legal_actions(self):
      return self.envs[self.player].legal_actions()

    def reset(self):
      self.observations = [
        self.envs[0].reset(),
        self.envs[1].reset(),
      ]
      self.player = 0
      self.reward = [0, 0]
      self.done = [False, False]
      return self.make_observations()

    def step(self, action):
      if not self.done[self.player]:
        observation, reward, done = self.envs[self.player].step(action)
        self.observations[self.player] = observation
        self.reward[self.player] += reward
        if done:
          self.done[self.player] = True


      if self.done[0] and self.done[1]:
        a = self.reward[self.player]
        b = self.reward[1 - self.player]
        if a > b:
          reward = 1
        elif a < b:
          reward = -1
        else:
          reward = 0

        print(f'Reward: {self.reward}, {reward}')
        self.player = 1 - self.player

        return self.make_observations(), reward * 3, True

      self.player = 1 - self.player
      return self.make_observations(), 0, False



    def make_observations(self):
      return np.concatenate((
        self.observations[self.player],
        self.observations[1 - self.player],
        self.make_reward(),
        self.make_done(),
      ), axis=0)


    def make_reward(self):
      return np.full(
        (1, self.observation_shape[1], self.observation_shape[2]),
        self.reward[self.player] - self.reward[1 - self.player],
        dtype=np.float32,
      )

    def make_done(self):
      done = np.zeros((2, self.observation_shape[1], self.observation_shape[2]), dtype=np.float32)
      if self.done[0]:
        done[0] = 1

      if self.done[1]:
        done[1] = 1

      return done

    def render(self):
      self.envs[0].render()

    def close(self):
      self.envs[0].close()
      self.envs[1].close()

  return Game
