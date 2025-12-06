"""
This module contains the getter function, which is used to get a model from a configuration.
"""

from typing import Dict, Union
from ruamel.yaml import YAML
from protein2pam.models.ensemble import ensemble_from_config


def get_model(cfg: Union[str, Dict], path=None):
    """Get a model from a configuration."""
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = YAML(typ="safe").load(f)
    return ensemble_from_config(cfg, model_path=path)
