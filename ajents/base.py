"""Base classes for Ajents"""
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

class BoltzmannPolicy(nn.Module):
    """Boltzmann (softmax) categorical policy"""
    du: int  # Dimensionality of action space
    f_cls: type = nn.Dense  # Policy class. Flax module mapping observations to logits
    temp: float = 1.  # Function of (params, obs) that returns logits (unnormalized scores) for all actions

    def setup(self):
        self.f = self.f_cls(self.du)

    def log_pi(self, obs, action):
        """Return log-probability/density of action at observation"""
        return jax.nn.log_softmax(self.temp * self.f(obs))[action.astype(int)]

    def sample(self, obs, rng):
        """Sample action from policy"""
        return jax.random.categorical(rng, self.temp * self.f(obs))

    def greedy(self, obs):
        """Greedy action"""
        return jnp.argmax(self.f(obs))
