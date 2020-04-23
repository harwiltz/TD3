import numpy as np

from sac.replay import ReplayBuffer

from td3.TD3 import TD3

class TD3Agent:
    def __init__(self,
                 env,
                 env_kwargs=None,
                 pre_train_steps=1000,
                 max_replay_capacity=10000,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tau=5e-3,
                 gamma=0.999,
                 batch_size=32,
                 value_delay=2):
        self._env_name = env
        if env_kwargs is None:
            self._env_fn = lambda: gym.make(self._env_name)
        else:
            self._env_fn = lambda: gym.make(self._env_name, **env_kwargs)

        env = self._env_fn()
        self._obs_dim = np.prod(env.observation_space.shape)
        self._act_dim = np.prod(env.action_space.shape)
        max_action = env.action_space.high
        min_action = env.action_space.low
        self._replay_buf = ReplayBuffer(self._obs_dim, self._act_dim, max_replay_capacity)
        self._td3 = TD3(self._obs_dim,
                        self._act_dim,
                        max_action,
                        min_action=min_action,
                        discount=gamma,
                        tau=tau,
                        policy_freq=value_delay)

    def rollout(self, num_rollout=1, render=False):
        rewards = np.zeros(num_rollouts)
        for i in range(num_rollouts):
            env = self._env_fn()
            s = env.reset()
            episode_reward = 0
            done = False
            while not done:
                if render:
                    env.render()
                a = self.action(s)
                s, r, done, _ = env.step(a)
                episode_reward += r
            rewards[i] = episode_reward
        if render:
            env.close()
        return rewards

    def train(self, num_steps, win_condition=None, win_window=5, logger=None):
        env = self._env_fn()
        s = env.reset()
        episode_reward = 0
        num_episodes = 0
        if win_condition is not None:
            scores = [0. for _ in range(win_window)]
            idx = 0
        for i in range(num_steps):
            if self._training:
                a = self.action(s)
            else:
                a = env.action_space.sample()
            ns, r, d, _ = env.step(a)
            episode_reward += r
            self._replay_buf.store(s, a, r, ns, d)
            self._total_steps += 1
            if not self._training:
                if self._total_steps >= self._pre_train_steps:
                    self._training = True
            if self._training:
                losses = self.update()
                artifacts = {
                    'loss': losses,
                     'step': self._total_steps,
                     'episode': num_episodes,
                     'done': d,
                     'return': episode_reward,
                     'transition': {
                         'state': s,
                         'action': a,
                         'reward': r,
                         'next state': ns,
                         'done': d,
                     }
                }
                if logger is not None:
                    logger(self, artifacts)
            if d:
                s = env.reset()
                num_episodes += 1
                if win_condition is not None:
                    scores[idx] = episode_reward
                    idx = (idx + 1) % win_window
                    if (num_episodes >= win_window) and (np.mean(scores) >= win_condition):
                        print("SAC finished training: win condition reached")
                        break
                episode_reward = 0
            else:
                s = ns

    def update(self):
        self._td3.train(self._replay_buf, self._batch_size)
        return {}

    def action(self, x):
        return self._td3.select_action(x)
