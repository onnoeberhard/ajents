"""Train linear policy with REINFORCE on CartPole"""
from datetime import datetime

import gym
import jax
import jax.numpy as jnp
from ajents.base import CategoricalPolicy
from ajents.pg import REINFORCE


@jax.jit
def policy_logits(params, obs):
    """Unnormalized log-policy. Simple policy linear in observation."""
    return params['weights'] @ obs + params['bias']

def init_params(key, env):
    """Initialize parameters of log-linear policy."""
    key1, key2 = jax.random.split(key)
    params = {'weights': 0.01 * jax.random.normal(key1, (env.action_space.n, env.observation_space.shape[0])),
              'bias': 0.01 * jax.random.normal(key2, (env.action_space.n,))}
    return params

def main(test=True, view=True):
    """Train linear policy with REINFORCE on CartPole"""
    key = jax.random.PRNGKey(42)

    # Initialize environment
    env = gym.make('CartPole-v1')

    # Initialize policy
    key, subkey = jax.random.split(key)
    policy = CategoricalPolicy(policy_logits)
    params = init_params(subkey, env)

    # Initialize agent
    key, subkey = jax.random.split(key)
    agent = REINFORCE(env, subkey, policy, params)

    # Train agent
    start = datetime.now()
    agent.learn(1000, 10, learning_rate=0.001, threshold=500)    # 0:01:47.527730 w/o jit
    print(f"Training finished after {datetime.now() - start}!")

    # Test agent
    if test:
        _, _, rewards, _ = agent.rollouts(500, explore=False)
        print(f"Average test return: {jnp.nansum(rewards, 1).mean()}")

    # Watch final policy in action
    while view:
        _, _, rewards, _ = agent.rollout(explore=False, render=True)
        print(f"Rollout score: {sum(rewards)}")

if __name__ == '__main__':
    main()
