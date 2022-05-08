"""DQN implementation"""

import jax
import jax.numpy as jnp
import flax.linen as nn

# First: For the specific example of CartPole
# Later: Make general.
# TODO: basically, do monitor callback wrapper, optionally pass gen_env function to agent for monitor functionality.

class DQN:
    def __init__(self, env):
        env.action_space.n
        self.q = nn.Sequential([
            nn.Dense(64), nn.relu,
            nn.Dense(64), nn.relu,
            nn.Dense(env.action_space.n)
        ])

        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        self.q_params = self.q.init(key1, jnp.ones((1, *env.observation_space.shape)))
        

    def train(self):
        pass

    def predict(self):
        pass




if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    
    # Initialize agent
    agent = DQN(env)

    # Train agent
    agent.train()

    # Test agent
    env.reset()
