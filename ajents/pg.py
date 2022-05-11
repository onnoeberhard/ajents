"""Policy Gradient algorithms"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
# from rlax import policy_gradient_loss
from tqdm import trange

from ajents.base import Agent


class REINFORCE(Agent):
    def __init__(self, env, key, log_policy, params, causal=True, baseline='mean'):
        super().__init__(env, key)
        self.log_policy = log_policy
        self.params = params
        self.baseline = baseline
        self.causal = causal

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
    def update(self, params, observations, actions, rewards, lr):
        # Calculate policy gradient from rollouts
        grads = self.grad_log_policy(params, observations, actions)
        grads = jax.tree_map(lambda x: jnp.nansum(x, 1), grads)
        returns = jnp.nansum(rewards, 1)
        grads = jax.tree_map(lambda x: (x.T @ returns).T, grads)

        # Update policy
        return jax.tree_map(lambda p, g: p + lr*g, params, grads)


    def learn(self, n_iterations, n_rollouts, lr=0.01):
        for j in range(n_iterations):
            # - Collect rollouts -
            observations = []
            actions = []
            rewards = []
            for _ in (pb := trange(n_rollouts, leave=False)):
                os, as_, rs, _ = self.rollout(render=False, pad=True)
                observations.append(os)
                actions.append(as_)
                rewards.append(rs)

            observations = jnp.array(observations)
            actions = jnp.array(actions)
            rewards = jnp.array(rewards)

            print(f"Iteration {j + 1:{len(str(n_iterations))}d}/{n_iterations}. Average return = {jnp.nansum(rewards, 1).mean():f}, Completed in {pb._time() - pb.start_t:.2f}s.")

            # - Update policy -
            self.params = self.update(self.params, observations, actions, rewards, lr)
