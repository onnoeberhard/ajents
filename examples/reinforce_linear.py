"""Train affine policy with REINFORCE on CartPole"""
from datetime import datetime

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.rich import tqdm

from ajents import REINFORCE, BoltzmannPolicy, rollout, rollouts, pad_rect


def policy_logits(params, obs):
    """Unnormalized log-policy. Simple policy affine in observation."""
    return params['weights'] @ obs + params['bias']

def init_params(env, rng):
    """Initialize parameters of log-affine policy."""
    key1, key2 = jax.random.split(rng)
    params = {'weights': 0.01 * jax.random.normal(key1, (env.action_space.n, env.observation_space.shape[0])),
              'bias': 0.01 * jax.random.normal(key2, (env.action_space.n,))}
    return params

def main(test=True, view=True):
    """Train affine policy with REINFORCE on CartPole"""
    jax.config.update('jax_platforms', 'cpu')
    seed = 42
    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    # Initialize environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    # Initialize policy
    rng, key = jax.random.split(rng)
    policy = BoltzmannPolicy(policy_logits, temp=2.)
    params = init_params(env, key)

    # Initialize agent
    agent = REINFORCE(policy, params)

    # Train agent
    start = datetime.now()
    rng, key = jax.random.split(rng)
    agent.learn(env, key, np_rng, 1000, 10, threshold=500)
    print(f"Training finished after {datetime.now() - start}!")

    # Test agent
    if test:
        print("Testing policy...")
        rng, key = jax.random.split(rng)
        _, _, rewards = rollouts(agent, env, key, np_rng, 500, explore=False, pb=tqdm)
        rewards = pad_rect(rewards)
        print(f"Average test return: {jnp.nansum(rewards, 1).mean()}")

    # Watch final policy in action
    print("Finished. Press '^C' to exit.")
    env = gym.make(env_name, render_mode='human')
    while view:
        rng, key = jax.random.split(rng)
        _, _, rewards, _ = rollout(agent, env, key, np_rng, explore=False, live=True)
        print(f"Episode return: {sum(rewards)}")

if __name__ == '__main__':
    main()
