"""Policy gradient algorithms"""
import warnings
from dataclasses import field
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import TqdmExperimentalWarning
from tqdm.rich import trange

from ajents.base import Agent, BoltzmannPolicy, rollouts
from ajents.util import pad_rect


class REINFORCE(Agent):
    """REINFORCE (vanilla policy gradient) agent"""
    du: int  # Dimensionality of action space
    policy_cls: type = BoltzmannPolicy
    policy_kwargs: dict = field(default_factory=dict)
    causal: bool = True
    baseline: bool = True
    learning_rate: float = 0.001

    def __post_init__(self):
        # self.optimizer = optax.sgd(-self.learning_rate)
        self.optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(100),
            optax.sgd(-self.learning_rate),
            optax.zero_nans(),
        )
        super().__post_init__()

    def setup(self):
        self._policy = self.policy_cls(self.du, **self.policy_kwargs)

    def __call__(self, obs, rng, explore):
        """Sample action from current policy (or greedy if `explore` is `False`)"""
        return self._policy.sample(obs, rng) if explore else self._policy.greedy(obs)

    @partial(nn.jit, static_argnums=(0,))
    def update(self, params, opt_state, observations, actions, rewards):
        """Calculate policy gradient and take one optimization step."""
        lp = nn.apply(lambda self, obs, action: self._policy.log_pi(obs, action), self)
        def grad_log_policy(obs, action):
            """Gradient (wrt. params) of log-policy at given state-action pair"""
            return jax.lax.cond(jnp.isnan(action).any(),
                lambda: jax.tree_map(lambda x: x*jnp.nan, params),
                lambda: jax.jacobian(lp)(params, obs, action)
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

    def learn(self, params, env, rng_agent, rng_env, n_iterations, n_rollouts, max_ep_len, threshold=None):
        """Train agent"""
        opt_state = self.optimizer.init(params)
        policy = jax.jit(lambda params, obs, rng: self.apply(params, obs, rng, True))

        for j in (pb := trange(n_iterations)):
            # Collect rollouts
            rng_agent, key_agent = jax.random.split(rng_agent)
            os, as_, rs = rollouts(partial(policy, params), env, key_agent, rng_env, n_rollouts)
            observations, actions, rewards = (pad_rect(x, max_ep_len + 1) for x in (os, as_, rs))
            ret = np.nansum(rewards, 1).mean()

            # Monitoring
            pb.write(f"Iteration {j + 1:{len(str(n_iterations))}d}/{n_iterations}. Average return = {ret:f}")

            # Break if threshold is reached
            if threshold is not None and ret >= threshold:
                print("Solved!")
                break

            # Calculate policy gradient and update policy
            params, opt_state = self.update(params, opt_state, observations, actions, rewards)

        return params, j + 1  # pylint: disable=undefined-loop-variable

# Suppress warning from tqdm rich library
warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)
