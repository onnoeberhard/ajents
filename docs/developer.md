# Profiling
python -m cProfile -o reinforce.prof examples/reinforce_linear.py
snakeviz reinforce.prof
