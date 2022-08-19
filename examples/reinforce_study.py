"""Train MLP policy with REINFORCE; study influence of hyperparameters."""
from datetime import datetime

import click
import flax.linen as nn
import gym
import jax
import numpy as np
from ajents.pg import REINFORCE


class MLP(nn.Module):
    """Simple MLP policy"""
    n_actions: int

    @nn.compact
    def __call__(self, x):    # pylint: disable=arguments-differ
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)    # Logits (unnormalized scores)
        return x


@click.command()
@click.option('--env', default='CartPole-v1', help='Gym environment used in experiments')
def main(env):
    """Train MLP policy with REINFORCE; study influence of hyperparameters."""
    env = gym.make(env)

    results = np.zeros((2, 2, 3, 4, 2))

    for i, causal in enumerate([True, False]):
        for j, baseline in enumerate([True , False]):
            for k, n_rollouts in enumerate([10, 100, 1000]):
                for l, learning_rate in enumerate([0.1, 0.01, 0.001, 0.0001]):
                    print(causal, baseline, n_rollouts, learning_rate)
                    key = jax.random.PRNGKey(42)
                    log_policy = MLP(env.action_space.n)

                    # Initialize policy params
                    key, subkey = jax.random.split(key)
                    params = log_policy.init(subkey, env.observation_space.sample())

                    # Initialize agent
                    key, subkey = jax.random.split(key)
                    agent = REINFORCE(env, subkey, log_policy.apply, params, causal=causal, baseline=baseline)

                    # Train agent
                    start = datetime.now()
                    steps = agent.learn(10_000 // n_rollouts, n_rollouts, learning_rate=learning_rate, threshold=500)
                    results[i, j, k, l, 0] = steps
                    print(f"Traning finished after {steps} iterations ({datetime.now() - start})!")

                    # Test agent
                    _, _, rewards, _ = agent.rollouts(500, explore=False)
                    results[i, j, k, l, 1] = np.nansum(rewards, 1).mean()
                    print(f"Average test return: {results[i, j, k, l, 1]}")

                    # Checkpoint
                    np.save('tmp/reinforce_study.npy', results)

if __name__ == '__main__':
    main()    # pylint: disable=no-value-for-parameter
