"""Policy Gradient algorithms"""

# REINFORCE
# 1. sample trajectories
# 2. compute rewards to go
# 3. compute policy gradient estimate
# 4. update policy

import jax
import jax.numpy as jnp
import numpy as np
# from rlax import policy_gradient_loss
# from ajents.base import Agent
from tqdm import trange

# First: no neural network, just a linear function.
class REINFORCE():
    # Class holds only hyperparameters.. ?
    # Q / policy states are taken as input/output.. (maybe??)
    # only if this is necessary. I think the only good reason for a functional approach is that  JAX transformations are possible, but if the environment is slow anyway.... is this necessary?
    # __init__ needs a gen_env function for logging (optinal)
    def collect_trajectory(self, key, env, log_policy, params):
        observations = []
        actions = []
        rewards = []
        
        key, subkey = jax.random.split(key)
        obs = env.reset(seed=subkey[0].item())
        env.render()
        observations.append(obs)

        done = False
        while not done:
            key, subkey = jax.random.split(key)
            action = jax.random.categorical(subkey, log_policy(params, obs))
            obs, reward, done, _ = env.step(np.asarray(action))
            env.render()
            actions.append(action)
            rewards.append(reward)
            observations.append(obs)
            #returns = returns.at[i].add(reward)
            # breakpoint()
            #log_pi_grad = jax.tree_map(lambda x, y: x + y[action.item()], log_pi_grad, lpg(params, obs))
        return observations, actions, rewards

    # def lpg_trajectory(self, )
                # jax.grad(lambda p, o: log_policy(p, o)[action])(params, obs))
    def learn(self, log_policy, params, env, key, N=20, lr=0.01):
        log_policy_grad = jax.jit(jax.vmap(lambda params, obs, action: 
            jax.tree_map(lambda leaf: leaf[action], jax.jacrev(log_policy)(params, obs)), in_axes=(None, 0, 0)))

        for j in range(50):    # 100 optimizer steps
            returns = jnp.empty(N)
            grad = jax.tree_map(lambda x: jnp.zeros_like(x), params)
            for i in trange(N):
                key, subkey = jax.random.split(key)
                observations, actions, rewards = self.collect_trajectory(subkey, env, log_policy, params)
                returns = returns.at[i].set(sum(rewards))
                traj_grads = log_policy_grad(params, jnp.array(observations[:-1]), jnp.array(actions))
                grad = jax.tree_map(lambda grad, traj_grads: grad + traj_grads.sum(0)*returns[i], grad, traj_grads)
                # breakpoint()
                # g = jax.tree_util.tree_reduce(jnp.sum, g)
                # breakpoint()    # tree? shape?
                # log_pi_grad = 
                # log_pi_grad = jax.tree_map(lambda x, y: x + y[action.item()], log_pi_grad, lpg(params, obs))

                # policy gradient of this trajectory
                # breakpoint()
                # traj_grad = jax.tree_map(lambda x: x * returns[i], log_pi_grad)    # weighting
                # grad = jax.tree_map(lambda x, y: x + y, grad, traj_grad)
            print(f"Step {j}. Average return = {returns.mean()}")

            # Update policy
            params = jax.tree_map(lambda p, g: p + lr*g, params, grad)
        
    # def learn2(self, log_policy, params, env, key, N=100, lr=0.01):
    #     # gradient of log-policy
    #     glp = jax.jit(jax.jacobian(log_policy))

    #     for j in range(5):
    #         returns = []    # return of each rollout
    #         lpgrads = []    # gradient of log-policy of each rollout
    #         grad = jax.tree_map(lambda x: jnp.zeros_like(x), params)    # Policy gradient

    #         # Collect N trajectories
    #         for i in trange(N):
    #             # Collect rollout
    #             key, subkey = jax.random.split(key)
    #             observations, actions, rewards = self.collect_trajectory(subkey, env, log_policy, params)
                
    #             # Calculate rollout return and gradient of log-policy
    #             returns.append(sum(rewards))
    #             lpgrad = [jax.tree_map(lambda w: w[a], glp(params, o)) for o, a in zip(observations, actions)]
    #             lpgrad = jax.tree_map(lambda *x: sum(x), *lpgrad)
    #             lpgrads.append(lpgrad)

    #             # breakpoint()
    #             # lpgrads.append(jax.tree_map(jnp.sum, *(glp(params, o)[a] for o, a in zip(observations, actions))))

    #             # jax.tree_map(lambda *x: sum(x), *(glp(params, o) for o, a in zip(observations, actions)))

    #             # # policy gradient
    #             # log_pi_grad = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    #             # for obs, action in zip(observations[:-1], actions):
    #             #     g = jax.grad(lambda p, o: log_policy(p, o)[action])(params, obs)
    #             #     log_pi_grad = jax.tree_map(lambda x, y: x + y, log_pi_grad, g)
                
    #             # grad = jax.tree_map(lambda g, lpg: g + lpg*returns[i], grad, log_pi_grad)

    #         print(f"Step {j}. Average return = {jnp.array(returns).mean()}")

    #         # Policy gradient
    #         grad = [jax.tree_map(lambda lpg: lpg*r, lpg) for lpg, r in zip(lpgrads, returns)]
    #         grad = jax.tree_map(lambda *x: sum(x), *grad)
    #         # grad = jax.tree_reduce(jnp.sum, grad)

    #         # Update policy
    #         params = jax.tree_map(lambda p, g: p + lr*g, params, grad)
        
    # def learn3(self, log_policy, params, env, key, N=100, lr=0.01):
    #     for j in range(5):
    #         returns = jnp.empty(N)
    #         grad = jax.tree_map(lambda x: jnp.zeros_like(x), params)

    #         for i in trange(N):
    #             key, subkey = jax.random.split(key)
    #             observations, actions, rewards = self.collect_trajectory(subkey, env, log_policy, params)
    #             returns = returns.at[i].set(sum(rewards))

    #             def pg_loss(params, observations, actions, rewards):
    #                 logits = jax.vmap(log_policy, in_axes=(None, 0))(params, observations)
    #                 return policy_gradient_loss(logits, actions, rewards, jnp.ones_like(rewards)/len(rewards))

    #             # trajectory gradient
    #             g = jax.grad(pg_loss)(params, jnp.array(observations[:-1]), jnp.array(actions), jnp.array(rewards))

    #             # policy gradient
    #             # log_pi_grad = jax.tree_map(lambda x: jnp.zeros_like(x), params)
    #             # for obs, action in zip(observations[:-1], actions):
    #             #     g = jax.grad(lambda p, o: log_policy(p, o)[action])(params, obs)
    #             #     log_pi_grad = jax.tree_map(lambda x, y: x + y, log_pi_grad, g)
                
    #             grad = jax.tree_map(lambda grad, g: grad + g, grad, g)

    #         print(f"Step {j}. Average return = {returns.mean()}")

    #         # Update policy
    #         params = jax.tree_map(lambda p, g: p - lr*g, params, grad)

            
        # this function takes in the policy function (would be great to be agnostic: take in flax network, fine, but also just take in linear python / JAX function.)
        # also training parameters.
        # this function should take as little positional arguments as possible.
        # returns trained policy, does everything in between.
        

# Online actor critic


# Off-policy actor critic


# natural policy gradient?


# also: trpo (but separately)

def add(a, b):
    return a + b

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')

    @jax.jit
    def log_policy(params, obs):
        return params['weights'] @ obs + params['bias']
    
    key = jax.random.PRNGKey(42)
    key, sk1, sk2 = jax.random.split(key, 3)
    params = {'weights': 0.01 * jax.random.normal(sk1, (env.action_space.n, env.observation_space.shape[0])), 
              'bias': 0.01 * jax.random.normal(sk2, (env.action_space.n,))}
    
    
    agent = REINFORCE()
    # with jax.profiler.trace("/tmp/tensorboard"):
    key, subkey = jax.random.split(key)
    import datetime
    # start = datetime.datetime.now()
    # agent.learn(log_policy, params, env, subkey)
    # end = datetime.datetime.now()
    # print(end - start)

    start = datetime.datetime.now()
    agent.learn(log_policy, params, env, subkey)
    end = datetime.datetime.now()
    print(end - start)

    # start = datetime.datetime.now()
    # agent.learn3(log_policy, params, env, subkey)
    # end = datetime.datetime.now()
    # print(end - start)
    # print(end - start)
    # start = datetime.datetime.now()
    # agent.learn3(log_policy, params, env, subkey)
    # end = datetime.datetime.now()
    # print(end - start)
