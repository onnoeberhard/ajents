"""Do JAX profiling for reinforce_linear.py example"""
import jax

from examples.reinforce_linear import main

# print("Starting profiler...")
# jax.profiler.start_trace('tmp/tensorboard')

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    main(test=False, view=False)

# print("Stopping profiler...")
# jax.profiler.stop_trace()
# print("Profiling results in TensorBoard (logdir=tmp/tensorboard).")
