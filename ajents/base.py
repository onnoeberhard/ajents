"""Base classes for Ajents"""
from dataclasses import field
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


class Agent(nn.Module):
    """Abstract agent class"""
    def __call__(self, obs, rng, explore):
        raise NotImplementedError


def rollout(policy, env, rng_agent, rng_env, steps=None, live=False):
    """Collect a rollout of `agent` in `env`. If `steps` is `None`, one episode is collected."""
    # Initialize return variables
    observations = []
    actions = []
    rewards = []
    info = {'steps': 0, 'episodes': 0}

    # Initialize environment
    env.np_random = rng_env
    obs, _ = env.reset()
    observations.append(obs)
    if live:
        print('Actions: ', end='')

    # Run interaction loop
    steps_left = steps or np.inf
    while steps_left:
        # Sample and execute action
        rng_agent, key = jax.random.split(rng_agent)
        action = policy(obs, key)
        obs, reward, terminated, truncated, _ = env.step(np.array(action))
        if live:
            print(action, end='\n' if terminated or truncated else '', flush=True)

        # Store interaction
        actions.append(action)
        rewards.append(reward)
        observations.append(obs)
        info['steps'] += 1
        info['episodes'] += terminated or truncated

        # Break if done
        steps_left -= 1
        if steps is None and (terminated or truncated):
            break

    return observations, actions, rewards, info


def rollouts(policy, env, rng_agent, rng_env, n_rollouts, steps=None, pb=None):
    """Collect multiple rollouts"""
    observations = []
    actions = []
    rewards = []

    pb = pb or (lambda x: x)
    for _ in pb(range(n_rollouts)):
        os, as_, rs, _ = rollout(policy, env, rng_agent, rng_env, steps)
        observations.append(os)
        actions.append(as_)
        rewards.append(rs)

    return observations, actions, rewards


class Policy(nn.Module):
    """Abstract policy class"""
    du: int  # Dimensionality of action space
    f_cls: type = nn.Dense  # Flax module mapping observations to outputs
    f_kwargs: dict = field(default_factory=dict)  # Additional keyword arguments for mapping f

    def __call__(self):
        """Return distribution over actions at observation"""
        raise NotImplementedError

    def log_pi(self, obs, action):
        """Return log-probability/density of action at observation"""
        return self(obs).log_prob(action)

    def sample(self, obs, rng):
        """Sample action from policy"""
        return self(obs).sample(seed=rng)

    def greedy(self, obs):
        """Greedy action"""
        return self(obs).mode()


class BoltzmannPolicy(Policy):
    """Boltzmann (softmax) policy for discrete control"""
    temp: float = 1.

    @nn.compact
    def __call__(self, obs):
        """Return distribution over actions at observation"""
        logits = self.f_cls(self.du, **self.f_kwargs)(obs)
        return distrax.Softmax(logits, self.temp)


class GaussianPolicy(Policy):
    """Gaussian policy for continuous control"""
    bounds: tuple = None

    def setup(self):
        self._f = self.f_cls(2*self.du, **self.f_kwargs)
        self.f = lambda obs: jnp.split(self._f(obs), 2, -1)
        self.bijector = squash(*self.bounds) if self.bounds else distrax.DiagLinear(jnp.ones(self.du))

    def __call__(self, obs):
        """Return distribution over actions at observation"""
        mu, log_std = self.f(obs)
        return distrax.Transformed(distrax.MultivariateNormalDiag(mu, jnp.exp(log_std)), self.bijector)

    def greedy(self, obs):
        """Greedy action"""
        mu, _ = self.f(obs)
        return self.bijector.forward(mu)    # Mode is difficult to compute

def squash(min_, max_, slope=1):
    """Bijector squashing input to (min_, max_) with given slope at x=0."""
    return distrax.Block(distrax.Chain([
        distrax.ScalarAffine(shift=min_, scale=max_ - min_),
        distrax.Sigmoid(),
        distrax.ScalarAffine(shift=0, scale=4 * slope / (max_ - min_))
    ]), 1)
