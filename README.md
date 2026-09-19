# Ajents: RL agents in JAX
This project contains minimal `flax.nnx`-based implementations of Proximal Policy Optimization (for categorical action spaces) and Soft Actor Critic (for continuous action spaces). Similarly to [CleanRL](https://github.com/vwxyzjn/cleanrl), the implementations are contained to single files for easy adaptation. The implementation expects environments to adhere to the [Gymnax](https://github.com/RobertTLange/gymnax) API.

As a starting point, `python ajents/ppo.py` will train a PPO agent on `Cartpole-v1` (in a few seconds on CPU) and `python ajents/sac.py` will train an SAC agent on `Pendulum-v1` (this takes considerably longer since the implementation is purely sequential).
It is straightforward to add Weights and Biases monitoring to the training as follows:

```python
import pickle

import gymnax
import jax
import wandb

from ajents import ppo

wandb.login(key='<WANDB_KEY>')
wandb.init(project='project-name', group='group-name',
    config={'seed': 42, 'env': 'CartPole-v1'})

rng = jax.random.key(42)
env = gymnax.environments.CartPole()
metrics, actor = ppo(env, rng, wablog=True)

with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
with open('actor.pkl', 'wb') as f:
    pickle.dump(actor, f)

wandb.finish()
```

If there are any problems, or if you have a question, don't hesitate to open an issue here on GitHub.
