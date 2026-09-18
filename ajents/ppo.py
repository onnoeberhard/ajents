from collections import namedtuple

import distrax
from einops import rearrange
from flax import nnx
import jax
import jax.numpy as jnp
from oetils import JaxTqdm
import optax
import wandb

Transition = namedtuple(
    'Transition', ['y', 'u', 'y_', 'r', 't', 'd', 'R', 'A', 'lp'])

class ActorCritic(nnx.Module):
    def __init__(self, env, rngs):
        dy = env.obs_shape[0]
        nu = env.num_actions
        kinit = nnx.initializers.orthogonal
        self.actor = nnx.Sequential(
            nnx.Linear(dy, 64, kernel_init=kinit(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(64, 64, kernel_init=kinit(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(64, nu, kernel_init=kinit(0.01), rngs=rngs)
        )
        self.critic = nnx.Sequential(
            nnx.Linear(dy, 64, kernel_init=kinit(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(64, 64, kernel_init=kinit(jnp.sqrt(2)), rngs=rngs),
            nnx.tanh,
            nnx.Linear(64, 1, kernel_init=kinit(1.0), rngs=rngs)
        )

    def __call__(self, y):
        return distrax.Categorical(self.actor(y))

    def v(self, y):
        return self.critic(y).squeeze(-1)

def ppo(env, rng, total_steps=512_000, n_envs=4, n_steps=128, lr=3.e-4,
        gam=0.99, lam=0.95, n_epochs=4, n_minibatches=4, clip_eps=0.2,
        vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, wablog=False):
    """Proximal policy optimization"""
    n_updates = total_steps // (n_envs * n_steps)
    pbar = JaxTqdm(total_steps // n_envs, n_updates)

    @nnx.vmap(in_axes=(None, 0))
    def loss(agent, t):
        pi = agent(t.y)
        value_loss = 1/2 * (agent.v(t.y) - t.R)**2  # No value clipping
        ratio = jnp.exp(logratio := pi.log_prob(t.u) - t.lp)
        policy_loss = -jnp.minimum(ratio * t.A,
            jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * t.A)
        entropy = pi.entropy()
        return policy_loss + vf_coef * value_loss - ent_coef * entropy, {
            'policy_loss': policy_loss, 'value_loss': value_loss,
            'entropy': entropy, 'approx_kl': ((ratio - 1) - logratio),
            'clipfrac': (jnp.abs(ratio - 1) > clip_eps)}

    @nnx.scan
    def update_batch(carry, batch):
        agent, opt = carry
        (losses, stats), grads = nnx.value_and_grad(lambda a, b: 
            jax.tree.map(jnp.mean, loss(a, b)), has_aux=True)(agent, batch)
        opt.update(agent, grads)
        return (agent, opt), (losses, stats)

    @nnx.scan(length=n_epochs, in_axes=(nnx.Carry, None))
    def update_epoch(carry, buf):
        agent, opt, rngs = carry
        perm = jax.random.permutation(rngs(), n_envs * n_steps)
        batch = jax.tree.map(lambda x:
            rearrange(x[perm], '(m b) ... -> m b ...', m=n_minibatches), buf)
        _, (losses, stats) = update_batch((agent, opt), batch)
        return (agent, opt, rngs), \
            (losses.mean(), jax.tree.map(jnp.mean, stats))

    def monitor_cb(i, metrics, stats):
        pbar.write(
            f"Steps: {metrics['t'][i]:{int(jnp.log10(total_steps)) + 1}d}. "
            f"Mean episode reward: {metrics['ep_rew'][i]:8.3f}, "
            f"mean episode length: {metrics['ep_len'][i]:8.3f}, "
            f"total loss: {stats['tot_loss']:8.3f}.")
        if wablog: wandb.log(
            {k: v[i] for k, v in metrics.items()} | stats, metrics['t'][i])

    def monitor(metrics, t, buf, stats, losses):
        metrics, R, L = metrics
        i = t // n_steps
        R, Rs = jax.lax.scan(
            lambda R, t: (jnp.where(t.d, 0, R + t.r), R + t.r), R, buf)
        L, Ls = jax.lax.scan(
            lambda L, t: (jnp.where(t.d, 0, L + 1), L + 1), L, buf)
        n_eps = buf.d.sum()
        stats = jax.tree.map(jnp.mean, stats | {'tot_loss': losses})
        metrics['t'] = metrics['t'].at[i].set((t + 1) * n_envs)
        metrics['ep_rew'] = metrics['ep_rew'].at[i].set(
            jnp.where(buf.d, Rs, 0).sum() / n_eps)
        metrics['ep_len'] = metrics['ep_len'].at[i].set(
            jnp.where(buf.d, Ls, 0).sum() / n_eps)
        metrics['policy_loss'] = metrics['policy_loss'].at[i].set(
            stats['policy_loss'])
        metrics['value_loss'] = metrics['value_loss'].at[i].set(
            stats['value_loss'])
        jax.debug.callback(monitor_cb, i, metrics, stats)
        return metrics, R, L

    def update(metrics, t, buf, agent, opt, rngs):
        # Compute lambda-returns and normalized advantages
        v = agent.v(buf.y_)
        R = jax.lax.scan(lambda R, t: (R := t[0].r + (1 - t[0].t) * gam * (
                (1 - lam * (1 - t[0].d)) * t[1] + lam * (1 - t[0].d) * R
            ), R), v[-1], (buf, v), reverse=True)[1]
        A = R - agent.v(buf.y)
        buf = buf._replace(R=R)
        buf = buf._replace(A=(A - A.mean()) / (A.std() + 1e-8))

        # Update parameters and process metrics
        _, (losses, stats) = update_epoch((agent, opt, rngs),
            jax.tree.map(lambda x: rearrange(x, 't n ... -> (t n) ...'), buf))
        return monitor(metrics, t, buf, stats, losses)

    @pbar
    @nnx.jit
    def step(t, x):
        metrics, buf, x, y, agent, opt, rngs = x

        # Sample action and step environment
        u, lp = agent(y).sample_and_log_prob(seed=rngs())
        y_, x, r, term, trunk, _ = jax.vmap(env.step)(
            jax.random.split(rngs(), n_envs), x, u)

        # Save transition to buffer and reset environment if done
        buf = jax.tree.map(lambda b, x: b.at[t % n_steps].set(x), buf,
            Transition(y, u, y_, r, term, term | trunk, 0., 0., lp))
        y, x = jax.vmap(lambda d, x, y, rng: 
            jax.lax.cond(d, lambda: env.reset(rng), lambda: (y, x)))(
            term | trunk, x, y_, jax.random.split(rngs(), n_envs))
        
        # Update parameters and collect statistics every n_steps
        metrics = nnx.cond((t + 1) % n_steps == 0, update,
            lambda m, *_: m, metrics, t, buf, agent, opt, rngs)
        return metrics, buf, x, y, agent, opt, rngs

    # Initialize agent and optimizer
    rngs = nnx.Rngs(rng)
    agent = ActorCritic(env, rngs)
    opt = nnx.Optimizer(agent, optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(optax.linear_schedule(
            lr, 0, n_updates * n_epochs * n_minibatches), eps=1e-5)),
        wrt=nnx.Param)

    # Initialize metrics, buffer, and environment
    metrics = {'t': jnp.zeros(n_updates, int)} | {k: jnp.zeros(n_updates)
        for k in ['ep_rew', 'ep_len', 'policy_loss', 'value_loss']}
    buf = jax.tree.map(lambda x: jax.lax.broadcast(x, (n_steps, n_envs)),
        Transition(jnp.zeros(env.obs_shape), 0,
            jnp.zeros(env.obs_shape), 0., False, False, 0., 0., 0.))
    y, x = jax.vmap(env.reset)(jax.random.split(rngs(), n_envs))

    # Start training
    (metrics, *_), *_ = nnx.fori_loop(0, total_steps // n_envs, step, (
        (metrics, jnp.zeros(n_envs), jnp.zeros(n_envs)), buf, x, y,
        agent, opt, rngs))
    params = nnx.state(agent, nnx.Param)
    return metrics, params


if __name__ == "__main__":
    import gymnax
    env = gymnax.environments.CartPole()
    rng = jax.random.key(42)
    ppo(env, rng)

