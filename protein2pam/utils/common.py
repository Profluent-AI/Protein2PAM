"""
This module contains common utility functions used throughout the protein2pam package.
"""

import copy
import json
import copy
from typing import Any, Mapping

import numpy as np
import torch


def model_info():
    """Print a formatted table of CRISPR model metadata."""    
    table = [
        ["Model Name", "Input Protein/Domain", "CRISPR Type", "Samples", "Includes lit PAMs", "Used in Webserver"],
        ["cas8", "Cas8 or Cas10d", "Type I", "28410", "No", "Yes"],
        ["cas9", "Cas9 PI-Domain", "Type II", "15843", "Yes", "Yes"],
        ["cas12", "Cas12 protein", "Type V", "1720", "Yes", "Yes"],
        ["cas9_full", "Cas9 protein", "Type II", "15843", "Yes", "No"],
        ["cas9_full_nolit", "Cas9 protein", "Type II", "15731", "No", "No"],
        ["cas9_pid_nme", "Cas9 PI-Domain", "Type II", "15843", "No", "No"],
        ["cas9_pid_nolit", "Cas9 PI-Domain", "Type II", "15731", "No", "No"],
        ["cas12_nolit", "Cas12", "Type V", "1675", "", "No"],
    ]

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    header = table[0]
    divider = ["-" * width for width in col_widths]

    print(" | ".join(f"{str(cell).ljust(width)}" for cell, width in zip(header, col_widths)))
    print(" | ".join(divider))

    for row in table[1:]:
        print(" | ".join(f"\033[1m{str(cell).ljust(width)}\033[0m" if i == 0 else f"{str(cell).ljust(width)}" for i, (cell, width) in enumerate(zip(row, col_widths))))

        
def exists(x):
    """Return True if x is not None."""   
    return x is not None


def parse_pam_logo(logo, return_info=False):
    """Parse a PAM logo into a numpy array."""
    logo = json.loads(logo) if isinstance(logo, str) else logo
    info_matrix = np.asarray([[v for _, v in sorted(m)] for _, m in logo])
    if return_info:
        return info_matrix
    info_matrix = np.maximum(info_matrix, 1e-8)
    return info_matrix / info_matrix.sum(axis=1).reshape(-1, 1)


def prob_to_info(p: torch.Tensor = None, logp: torch.Tensor = None) -> torch.Tensor:
    """Convert probability or log probability to information."""
    if logp is None and p is not None:
        logp = p.log()
    elif p is None and logp is not None:
        p = logp.exp()
    elif p is None and logp is None:
        raise ValueError("Cannot have both p and logp be None")
    return (
        p
        * torch.nansum(p * (logp + np.log(logp.shape[-1])), dim=-1, keepdims=True)
        / np.log(2)
    )


def process_config(config: Mapping[str, Any]):
    """Process a configuration dictionary or list of configurations."""
    if isinstance(config, list):
        return sum([process_config(c) for c in config], [])

    configs = [config]
    if isinstance(config, dict):
        for k, v in config.items():
            v = process_config(v)
            new_configs = []
            for x in v:
                for config in configs:
                    new_config = copy.deepcopy(config)
                    new_config[k] = x
                    new_configs.append(new_config)
            configs = new_configs
    return configs

