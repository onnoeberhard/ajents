"""Policy gradient algorithms"""
from functools import partial
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import TqdmExperimentalWarning
from tqdm.rich import trange

from ajents.base import Agent, rollouts
from ajents.util import pad_rect


class REINFORCE(Agent):
    """REINFORCE (vanilla policy gradient) agent"""
    def __init__(self, policy, params, causal=True, baseline=True, learning_rate=0.001):
        # Fixed parameters
        self.policy = policy
        self.causal = causal
        self.baseline = baseline

        # Initialize optimizer
        self.optimizer = optax.sgd(-learning_rate)

        # Variables
        self.params = params
        self.opt_state = self.optimizer.init(params)

    @partial(jax.jit, static_argnums=(0, 4))
    def _act(self, params, obs, rng, explore):
        return self.policy.sample(params, obs, rng) if explore else self.policy.greedy(params, obs)

    def act(self, obs, rng, explore=True):
        """Sample action from current policy (or greedy if `explore` is `False`)"""
        return self._act(self.params, obs, rng, explore)

    @partial(jax.jit, static_argnums=(0,))
    def update(self, params, opt_state, observations, actions, rewards):
        """Calculate policy gradient and take one optimization step."""
        def grad_log_policy(obs, action):
            """Gradient (wrt. params) of log-policy at given state-action pair"""
            return jax.lax.cond(jnp.isnan(action).any(),
                lambda: jax.tree_map(lambda x: x*jnp.nan, params),
                lambda: jax.jacobian(self.policy.log_pi)(params, obs, action)
            )

        # Calculate policy gradient from rollouts
        grads = jax.vmap(jax.vmap(grad_log_policy))(observations, actions)
        returns = jnp.nansum(rewards, 1)
        if self.causal:
            # Use rewards to go (account for causality)
            rewards_to_go = returns[:, None] - jnp.nancumsum(rewards, 1) + rewards
            advantage = rewards_to_go - rewards_to_go.mean(0) if self.baseline else rewards_to_go
            grads = jax.tree_map(lambda x: jax.vmap(jax.vmap(jnp.multiply))(x, advantage), grads)
            grads = jax.tree_map(lambda x: jnp.nansum(x, 1), grads)
        else:
            # Use total episode return
            advantage = returns - returns.mean() if self.baseline else returns
            grads = jax.tree_map(lambda x: jnp.nansum(x, 1), grads)
            grads = jax.tree_map(lambda x: jax.vmap(jnp.multiply)(x, advantage), grads)
        grads = jax.tree_map(lambda x: x.mean(0), grads)

        # Update policy
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def learn(self, env, rng_agent, rng_env, n_iterations, n_rollouts, threshold=None):
        """Train agent"""
        for j in (pb := trange(n_iterations)):
            # Collect rollouts
            os, as_, rs = rollouts(self, env, rng_agent, rng_env, n_rollouts)
            observations, actions, rewards = (pad_rect(x) for x in (os, as_, rs))
            observations = observations[:, :-1]
            ret = np.nansum(rewards, 1).mean()

            # Monitoring
            pb.write(f"Iteration {j + 1:{len(str(n_iterations))}d}/{n_iterations}. Average return = {ret:f}")

            # Break if threshold is reached
            if threshold is not None and ret >= threshold:
                print("Solved!")
                break

            # Calculate policy gradient and update policy
            self.params, self.opt_state = self.update(self.params, self.opt_state, observations, actions, rewards)

        return j + 1  # pylint: disable=undefined-loop-variable

# Suppress warning from tqdm rich library
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)
