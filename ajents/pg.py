"""Policy Gradient algorithms"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ajents.base import Agent


class REINFORCE(Agent):
    """REINFORCE (Vanilla Policy Gradient) Agent class"""
    def __init__(self, env, key, log_policy, params, causal=True, baseline=True):
        super().__init__(env, key)
        # Fixed parameters
        self.log_policy = log_policy
        self.baseline = baseline
        self.causal = causal

        # Variables
        self.params = params

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def grad_log_policy(self, params, obs, action):
        """Gradient (wrt. params) of log-policy at given state-action pair"""
        return jax.lax.cond(jnp.isnan(action),
            lambda: jax.tree_map(lambda x: x*jnp.nan, params),
            lambda: jax.tree_map(lambda leaf: leaf[action.astype(int)], jax.jacobian(self.log_policy)(params, obs))
        )

    @partial(jax.jit, static_argnums=(0,))
    def _act_explore(self, params, obs, key):
        key, subkey = jax.random.split(key)
        return jax.random.categorical(subkey, self.log_policy(params, obs)), key

    @partial(jax.jit, static_argnums=(0,))
    def _act_exploit(self, params, obs):
        return jnp.argmax(self.log_policy(params, obs))

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
        return jax.tree_map(lambda p, g: p + learning_rate*g, params, grads), returns.mean()

    def learn(self, n_iterations, n_rollouts, learning_rate=None, render=False):
        """Train REINFORCE agent"""
        if learning_rate is None:
            learning_rate = lambda i: 1/(i+1)
        elif np.isscalar(learning_rate):
            learning_rate = lambda _: learning_rate

        for j in range(n_iterations):
            # Collect rollouts
            observations, actions, rewards, info = self.rollouts(n_rollouts, render=render)

            # Update policy
            self.params, ret = self.update(self.params, observations, actions, rewards, learning_rate(j))

            # Monitoring
            print(f"Iteration {j + 1:{len(str(n_iterations))}d}/{n_iterations}. "
                  f"Average return = {ret:f}, Completed in {info['time']:.2f}s.")
