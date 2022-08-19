"""Base classes for Ajents"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


def _split_seed(key):
    """Split PRNG key into new key and integer seed"""
    key, subkey = jax.random.split(key)
    return subkey[0], key

class Agent:
    """Abstract agent class"""
    def __init__(self, env, key):
        self.env = env
        self.key = key

    def reset(self, render=False):
        """Start new episode in environment and get first observation"""
        seed, self.key = _split_seed(self.key)
        obs = self.env.reset(seed=np.asarray(seed).item())
        if render:
            self.render()
        return obs

    def step(self, action, render=False):
        """Take one step in environment, return True if done"""
        obs, reward, done, info = self.env.step(np.array(action, dtype=np.float32))
        if render:
            print(action)
            self.render()
        return obs, reward, done, info

    def render(self):
        """Render state of the environment in window"""
        self.env.render()

    def rollout(self, steps=None, render=False, explore=True, pad=False):
        """Collect a rollout. If steps=None, one episode (until `done`) is collected."""
        # Initialize return variables
        observations = []
        actions = []
        rewards = []
        info = {'steps': 0, 'terminations': 0}

        # First observation
        obs = self.reset(render)
        observations.append(obs)

        # Run interaction loop
        steps_left = steps or np.inf
        while steps_left:
            # Sample and execute action
            action, pre_action = self.act(obs, explore)
            obs, reward, done, _ = self.step(action, render)

            # Store interaction
            actions.append(pre_action)
            rewards.append(reward)
            observations.append(obs)
            info['steps'] += 1
            info['terminations'] += done

            # Break if done
            steps_left -= 1
            if steps is None and done:
                break

        if pad:
            observations = np.array(observations
                + [np.ones_like(obs) * np.nan] * (self.env._max_episode_steps - len(observations) + 1))[:-1]
            actions = np.array(
                actions + [np.ones_like(action) * np.nan] * (self.env._max_episode_steps - len(actions)))
            rewards = np.array(rewards + [np.nan] * (self.env._max_episode_steps - len(rewards)))

        return observations, actions, rewards, info

    def rollouts(self, n_rollouts, array=True, progress=True, render=False, explore=True):
        """Collect multiple rollouts"""
        observations = []
        actions = []
        rewards = []

        it = range(n_rollouts)
        if progress:
            it = tqdm(it, leave=False)

        for _ in it:
            os, as_, rs, _ = self.rollout(render=render, pad=array, explore=explore)
            observations.append(os)
            actions.append(as_)
            rewards.append(rs)

        if array:
            observations = jnp.array(np.array(observations))
            actions = jnp.array(np.array(actions))
            rewards = jnp.array(np.array(rewards))

        info = {}
        if progress:
            info['time'] = it._time() - it.start_t

        return observations, actions, rewards, info

    def act(self, obs, explore=True):
        """Sample action from policy"""
        raise NotImplementedError


class GaussianPolicy:
    """Multivariate diagonal Gaussian policy"""
    def __init__(self, f, bounds=None):
        # Function of (params, obs) that returns (mu, log-sigma)
        self.f_ = f

        # Epsilon to avoid numerical instability
        self.eps = 1e-6

        # If bounds=(a, b), shift and apply tanh function (squash) actions to keep them in interval [a, b]
        self.bounds = bounds
        if bounds is not None:
            assert all(bounds[1] - bounds[0] > 2*self.eps)

    def f(self, params, obs):
        """Return mean and covariance of policy"""
        mu, log_sigma = self.f_(params, obs)
        # return mu.ravel(), jnp.diag(jnp.exp(log_sigma.ravel())) + jnp.eye(len(mu.ravel()))*self.eps
        return mu.ravel(), jnp.eye(len(mu.ravel()))

    def log_pi(self, params, obs, action):
        """Return log-probability/densits of (Gaussian/pre) action at observation"""
        # if self.bounds is not None:
        #     action_ = action
        #     action = self.unsquash(action)
            # if action_ == self.squash(self.f(params, obs)[0]):
            #     action = self.f(params, obs)[0]

        # The problem (I think) is:
        # When an action sampled from f is outside of the bounds, this information is lost
        # when squashing. Here, the "original" gaussian action is "reconstructed" by unsquashing,
        # which only works when the original action was inside the bounds. Otherwise, this
        # discrepancy (between orignal action, sampled fron actual Gaussian, and unsquashed action,
        # lying directly on the bound) will result in a very small probability. Imagin the original action
        # was sampled at -50, but the unsquashed action will lie on the bound at -7.68 or so. This 
        # results in a log-probability of -inf.
        # Solution:
        # 1. Maybe don't use the probability of the unsquashed action here, but somehow the probability of "any action"
        #    that is identical after squashing.
        # 2. Other idea: When sampling, also return gaussian action (I think SB3 is doing something similar). Then use 
        #    this here and don't unsquash.
        log_prob = jax.scipy.stats.multivariate_normal.logpdf(action, *self.f(params, obs))

        # if self.bounds is not None:
        #     # Transformation law of random variables. The Jacobian of unsquash is diagonal,
        #     # so the determinant is just a product (evaluated as sum outside log).
        #     log_prob += jnp.sum(jnp.log(jnp.diag(jax.jacobian(self.unsquash)(action_))))

        return log_prob

    def sample(self, params, obs, key, pre=False):
        """Sample action from policy"""
        sample_ = jax.random.multivariate_normal(key, *self.f(params, obs))
        if self.bounds is not None:
            sample = self.squash(sample_)
        return (sample, sample_) if pre else sample

    def greedy(self, params, obs, pre=False):
        """Greedy action"""
        action_ = self.f(params, obs)[0]
        if self.bounds is not None:
            action = self.squash(action_)
        return (action, action_) if pre else action

    def squash(self, x):
        """Squash action to interval `self.bounds` using ``tanh``"""
        a, b = self.bounds
        x = jnp.tanh(x)*(b - a)/2 + (a + b)/2
        return x.clip(a + self.eps, b - self.eps)

    def unsquash(self, x):
        """Unsquash action (see `self.squash`)"""
        a, b = self.bounds
        def us(x):
            return jnp.arctanh(2 * (x - (a + b)/2) / (b - a))
        return us(x).clip(us(a + self.eps), us(b - self.eps))


class CategoricalPolicy:
    """Categorical (Softmax / Multinoulli) policy"""
    def __init__(self, f):
        # Function of params, obs that returns logits (unnormalized scores) for all actions
        self.f = f

    def log_pi(self, params, obs, action):
        """Return log-probability/densits of action at observation"""
        return jax.nn.log_softmax(self.f(params, obs))[action.astype(int)]

    def sample(self, params, obs, key):
        """Sample action from policy"""
        return jax.random.categorical(key, self.f(params, obs))

    def greedy(self, params, obs):
        """Greedy action"""
        return jnp.argmax(self.f(params, obs))
