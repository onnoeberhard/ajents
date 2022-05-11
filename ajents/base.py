import jax
import jax.numpy as jnp
import numpy as np


class Agent:
    def __init__(self, env, key):
        self.env = env
        self.key = key
    
    def reset(self, render=False):
        """Start new episode in environment and get first observation"""
        self.key, subkey = jax.random.split(self.key)
        obs = self.env.reset(seed=subkey[0].item())
        if render:
            self.render()
        return obs

    def step(self, action, render=False):
        """Take one step in environment, return True if done"""
        obs, reward, done, info = self.env.step(np.asarray(action))
        if render:
            self.render()
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def rollout(self, steps=None, render=False, explore=True, pad=False):
        """Collect a rollout. If steps=None, one episode (until `done`) is collected."""
        # Initialize return variables
        observations = []
        actions = []
        rewards = []
        info = {'steps': 0, 'terminations': 0}
        
        # First observation
        obs = self.reset(render)
        observations.append(obs)

        # Run interaction loop
        steps_left = steps or np.inf
        while steps_left:
            # Sample and execute action
            action = self.act(obs, explore)
            obs, reward, done, _ = self.step(action, render)

            # Store interaction
            actions.append(action)
            rewards.append(reward)
            observations.append(obs)
            info['steps'] += 1
            info['terminations'] += done

            # Break if done
            steps_left -= 1
            if steps is None and done:
                break
        
        if pad:
            observations = np.array(
                observations + [np.ones_like(obs) * np.nan] * (self.env._max_episode_steps - len(observations)))
            actions = np.array(
                actions + [np.ones_like(action) * np.nan] * (self.env._max_episode_steps - len(actions)))
            rewards = np.array(rewards + [np.nan] * (self.env._max_episode_steps - len(rewards)))

        return observations, actions, rewards, info

    def predict(self, obs):
        pass    # for distribution?

    def act(self, obs, explore):
        pass   # for determinstic action?
