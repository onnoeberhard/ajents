import pickle

import gymnax
import jax
import wandb

from ajents import ppo

wandb.login(key='<WANDB-KEY>')
wandb.init(project='ajents', group='ppo-test',
    config={'seed': 42, 'env': 'CartPole-v1'})

rng = jax.random.key(42)
env = gymnax.environments.CartPole()
metrics, actor = ppo(env, rng, wablog=True)

with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
with open('actor.pkl', 'wb') as f:
    pickle.dump(actor, f)

wandb.finish()
