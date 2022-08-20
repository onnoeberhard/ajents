"""Policy gradient algorithms"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ajents.base import Agent


class REINFORCE(Agent):
    """REINFORCE (vanilla policy gradient) agent class"""
    def __init__(self, env, key, policy, params, causal=True, baseline=True):
        super().__init__(env, key)

        # Fixed parameters
        self.policy = policy
        self.causal = causal
        self.baseline = baseline

        # Variables
        self.params = params

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def grad_log_policy(self, params, obs, action):
        """Gradient (wrt. params) of log-policy at given state-action pair"""
        return jax.lax.cond(jnp.isnan(action).any(),
            lambda: jax.tree_map(lambda x: x*jnp.nan, params),
            lambda: jax.jacobian(self.policy.log_pi)(params, obs, action)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _act_explore(self, params, obs, key):
        key, subkey = jax.random.split(key)
        return self.policy.sample(params, obs, subkey), key

    @partial(jax.jit, static_argnums=(0,))
    def _act_exploit(self, params, obs):
        return self.policy.greedy(params, obs)

    def act(self, obs, explore=True):
        """Sample action from current policy"""
        if explore:
            action, self.key = self._act_explore(self.params, obs, self.key)
            return action
        return self._act_exploit(self.params, obs)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, params, observations, actions, rewards, learning_rate):
        """Calculate policy gradient and take one gradient ascent step."""
        # Calculate policy gradient from rollouts
        grads = self.grad_log_policy(params, observations, actions)
        returns = jnp.nansum(rewards, 1)
        if self.causal:
            rewards_to_go = returns[:, None] - jnp.nancumsum(rewards, 1) + rewards
            advantage = rewards_to_go - rewards_to_go.mean(0) if self.baseline else rewards_to_go
            grads = jax.tree_map(lambda x: jax.vmap(jax.vmap(jnp.multiply))(x, advantage), grads)
            grads = jax.tree_map(lambda x: jnp.nansum(x, 1), grads)
        else:
            advantage = returns - returns.mean() if self.baseline else returns
            grads = jax.tree_map(lambda x: jnp.nansum(x, 1), grads)
            grads = jax.tree_map(lambda x: jax.vmap(jnp.multiply)(x, advantage), grads)
        grads = jax.tree_map(lambda x: x.mean(0), grads)

        # Update policy
        return jax.tree_map(lambda p, g: p + learning_rate*g, params, grads)

    def learn(self, n_iterations, n_rollouts, learning_rate, render=False, threshold=None):
        """Train REINFORCE agent"""
        for j in range(n_iterations):
            # Collect rollouts
            observations, actions, rewards, info = self.rollouts(n_rollouts, render=render)
            ret = np.nansum(rewards, 1).mean()

            # Monitoring
            print(f"Iteration {j + 1:{len(str(n_iterations))}d}/{n_iterations}. "
                  f"Average return = {ret:f}, Completed in {info['time']:.2f}s.")

            # Break if threshold is reached
            if threshold is not None and ret >= threshold:
                print("Solved!")
                break

            # Calculate policy gradient and update policy
            self.params = self.update(self.params, observations, actions, rewards, learning_rate)

        return j + 1
