import torch.nn as nn
from typing import List, Union


def params(mod: Union[List, nn.Module]):
    """
    Recursively get parameters from a single / list of `nn.Modules`
    Useful when you want to grab multiple parts of the model separately
      and put them all into a single parameter group
    """
    if isinstance(mod, nn.Module):
        return list(mod.parameters())
    elif isinstance(mod, list):
        for m in mod:
            return params(m)
