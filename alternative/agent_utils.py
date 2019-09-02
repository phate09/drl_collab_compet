import sys
sys.path.append("../")

try:
    from alternative.models import Actor, Critic
    from alternative import param_table
    from alternative.make_env import make
except ModuleNotFoundError:
    from .models import Actor, Critic
    from alternative import param_table
    from alternative.make_env import make
