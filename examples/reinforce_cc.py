"""REINFOCE continuous control on the Pendulum"""
from datetime import datetime
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.rich import tqdm

from ajents import REINFORCE, pad_rect, rollout, rollouts
from ajents.base import GaussianPolicy
from ajents.nn import MLP


def main(seed=42, test=True, view=True):
    """Train affine policy with REINFORCE on CartPole"""
    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    # Initialize environment
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    du = env.action_space.shape[0]
    obs, _ = env.reset(seed=0)
    bounds = env.action_space.low[0], env.action_space.high[0]

    # Initialize agent
    rng, key = jax.random.split(rng)
    policy_cls = partial(GaussianPolicy, f_cls=MLP, bounds=bounds)
    agent = REINFORCE(du, policy_cls)
    params = agent.init(key, obs, key, False)

    # Train agent
    start = datetime.now()
    rng, key = jax.random.split(rng)
    params, _ = agent.learn(params, env, key, np_rng, 2000, 10, 1000, threshold=500)
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
    jax.config.update('jax_platforms', 'cpu')
    # jax.config.update('jax_log_compiles', True)
    main()
