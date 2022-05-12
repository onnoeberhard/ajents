"""Train linear policy with REINFORCE on CartPole"""
import contextlib
from datetime import datetime
import gym
import jax
import click

from ajents.pg import REINFORCE

@click.command()
@click.option('--profile', is_flag=True, default=False)
def main(profile):
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
    agent = REINFORCE(env, subkey, log_policy, params)
    start = datetime.now()
    with jax.profiler.trace('tmp/tensorboard') if profile else contextlib.nullcontext():
        agent.learn(10, 100, learning_rate=0.01)
        print(f"Traning finished after {datetime.now() - start}!")
    while True:
        _, _, rewards, _ = agent.rollout(explore=False, render=True)
        print(f"Rollout score: {sum(rewards)}")

if __name__ == '__main__':
    main()    # pylint: disable=no-value-for-parameter
