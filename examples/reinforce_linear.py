from datetime import datetime
import gym
import jax
import jax.numpy as jnp

from ajents.pg import REINFORCE
env = gym.make('CartPole-v1')

@jax.jit
def log_policy(params, obs):
    """Linear policy with polynomial features"""
    # x = jnp.concatenate((obs, jnp.r_[1]))
    # x = jnp.outer(x, x).ravel()
    return params['weights'] @ obs + params['bias']

key = jax.random.PRNGKey(42)
key, sk1, sk2 = jax.random.split(key, 3)
# (1 + env.observation_space.shape[0])**2
params = {'weights': 0.01 * jax.random.normal(sk1, (env.action_space.n, env.observation_space.shape[0])), 
          'bias': 0.01 * jax.random.normal(sk2, (env.action_space.n,))}

key, subkey = jax.random.split(key)
agent = REINFORCE(env, subkey, log_policy, params)
start = datetime.now()
agent.learn(25, 100, lr=0.001)
print(f"Traning finished after {datetime.now() - start}!")
while True:
    _, _, rewards, _ = agent.rollout(explore=False, render=True)
    print(f"Rollout score: {sum(rewards)}")
