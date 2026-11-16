"""OmegaConf <-> plain-dict helpers.

Self-contained config-reformatting helpers.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def omegaconf_to_dict(cfg: DictConfig) -> dict:
    """Resolve interpolations and convert nested dictionaries and lists to plain containers."""
    return OmegaConf.to_container(cfg, resolve=True)


def print_dict(val, nesting: int = -4, start: bool = True):
    """Pretty-print a nested dict for debugging."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        print(val)
