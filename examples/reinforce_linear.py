"""Train linear policy with REINFORCE on CartPole"""
from datetime import datetime

import click
import gym
import jax
import jax.numpy as jnp
from ajents.base import CategoricalPolicy
from ajents.pg import REINFORCE


@jax.jit
def policy_logits(params, obs):
    """Policy linear in observation"""
    return params['weights'] @ obs + params['bias']

def init_params(key, env):
    """Initialize parameters of log-linear policy."""
    key1, key2 = jax.random.split(key)
    params = {'weights': 0.01 * jax.random.normal(key1, (env.action_space.n, env.observation_space.shape[0])),
              'bias': 0.01 * jax.random.normal(key2, (env.action_space.n,))}
    return params

@click.command()
@click.option('--profile/--no-profile', is_flag=True, default=False, show_default=True, help='If enabled, the JAX'
    ' profiler will be enabled and results can be viewed in TensorBoard.')
@click.option('--test/--no-test', is_flag=True, default=True, show_default=True)
@click.option('--view/--no-view', is_flag=True, default=True, show_default=True)
def main(profile, test, view):
    """Train linear policy with REINFORCE on CartPole"""
    env = gym.make('CartPole-v1')
    key = jax.random.PRNGKey(42)

    # Initialize policy
    key, subkey = jax.random.split(key)
    policy = CategoricalPolicy(policy_logits)
    params = init_params(subkey, env)

    # Initialize agent
    key, subkey = jax.random.split(key)
    agent = REINFORCE(env, subkey, policy, params)

    if profile:
        print("Starting profiler...")
        jax.profiler.start_trace('tmp/tensorboard')

    # Train agent
    start = datetime.now()
    agent.learn(1000, 10, learning_rate=0.001, threshold=500)
    print(f"Training finished after {datetime.now() - start}!")

    if profile:
        print("Stopping profiler...")
        jax.profiler.stop_trace()
        print("Profiling results in TensorBoard (logdir=tmp/tensorboard).")

    # Test agent
    if test:
        _, _, rewards, _ = agent.rollouts(500, explore=False)
        print(f"Average test return: {jnp.nansum(rewards, 1).mean()}")

    # Watch final policy in action
    while view:
        _, _, rewards, _ = agent.rollout(explore=False, render=True)
        print(f"Rollout score: {sum(rewards)}")

if __name__ == '__main__':
    main()    # pylint: disable=no-value-for-parameter
