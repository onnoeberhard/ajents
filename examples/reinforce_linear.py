"""Train affine policy with REINFORCE on CartPole"""
from datetime import datetime

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.rich import tqdm

from ajents import REINFORCE, pad_rect, rollout, rollouts
from ajents.base import GaussianPolicy


def main(test=True, view=True):
    """Train affine policy with REINFORCE on CartPole"""
    # jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_log_compiles', True)
    seed = 42
    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    # Initialize environment
    env_name = 'CartPole-v1'
    # env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    du = env.action_space.n
    # du = env.action_space.shape[0]
    obs, _ = env.reset(seed=0)

    # Initialize agent
    rng, key = jax.random.split(rng)
    agent = REINFORCE(du)
    # agent = REINFORCE(du, GaussianPolicy)
    params = agent.init(key, obs, key, False)

    # Train agent
    start = datetime.now()
    rng, key = jax.random.split(rng)
    params, _ = agent.learn(params, env, key, np_rng, 2000, 10, 500, threshold=500)
    print(f"Training finished after {datetime.now() - start}!")
    policy = jax.jit(lambda obs, rng: agent.apply(params, obs, rng, False))

    # Test agent
    if test:
        print("Testing policy...")
        rng, key = jax.random.split(rng)
        _, _, rewards = rollouts(policy, env, key, np_rng, 500, pb=tqdm)
        rewards = pad_rect(rewards)
        print(f"Average test return: {jnp.nansum(rewards, 1).mean()}")

    # Watch final policy in action
    print("Finished. Press '^C' to exit.")
    env = gym.make(env_name, render_mode='human')
    while view:
        rng, key = jax.random.split(rng)
        _, _, rewards, _ = rollout(policy, env, key, np_rng, live=True)
        print(f"Episode return: {sum(rewards)}")

if __name__ == '__main__':
    main()
