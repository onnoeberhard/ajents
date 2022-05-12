"""Train linear policy with REINFORCE on CartPole"""
import contextlib
from datetime import datetime

import click
import gym
import jax
import jax.numpy as jnp
from ajents.pg import REINFORCE


@click.command()
@click.option('--profile', is_flag=True, default=False, show_default=True)
@click.option('--test/--no-test', is_flag=True, default=True, show_default=True)
@click.option('--view/--no-view', is_flag=True, default=True, show_default=True)
def main(profile, test, view):
    """Train linear policy with REINFORCE on CartPole"""
    env = gym.make('CartPole-v1')

    @jax.jit
    def log_policy(params, obs):
        """Policy linear in observation"""
        return params['weights'] @ obs + params['bias']

    key = jax.random.PRNGKey(42)
    key, sk1, sk2 = jax.random.split(key, 3)
    params = {'weights': 0.01 * jax.random.normal(sk1, (env.action_space.n, env.observation_space.shape[0])),
            'bias': 0.01 * jax.random.normal(sk2, (env.action_space.n,))}

    key, subkey = jax.random.split(key)
    agent = REINFORCE(env, subkey, log_policy, params, causal=False, baseline=False)
    start = datetime.now()
    with jax.profiler.trace('tmp/tensorboard') if profile else contextlib.nullcontext():
        agent.learn(20, 100)
        print(f"Traning finished after {datetime.now() - start}!")

    if test:
        _, _, rewards, _ = agent.rollouts(500, explore=False)
        print(f"Average test return: {jnp.nansum(rewards, 1).mean()}")

    while view:
        _, _, rewards, _ = agent.rollout(explore=False, render=True)
        print(f"Rollout score: {sum(rewards)}")

if __name__ == '__main__':
    main()    # pylint: disable=no-value-for-parameter
