"""
This module contains the Ensemble class, which is used to combine multiple models into a single ensemble.
"""

import copy
import json
from typing import Any, Dict, List, Mapping, Union

from ruamel.yaml import YAML
import torch
import torch.nn as nn

from protein2pam.models.esm2 import ESMPlusHead, get_esm
from protein2pam.utils.common import process_config


class Ensemble(nn.Module):
    """Ensemble of models."""
    def __init__(self, models: List[nn.Module]):
        """Initialize the ensemble."""
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, embeddings, *args, **kwargs):
        """Forward pass of the ensemble."""
        return torch.stack(
            [model(embeddings, *args, **kwargs) for model in self.models]
        ).mean(dim=0)


def get_base_model(cfg: Dict, headless=False) -> nn.Module:
    """Get the base model from the configuration."""
    model_cfg = copy.deepcopy(cfg["model"])
    if model_cfg["name"] in ["esm", "head"]:
        head_cfg = (
            model_cfg if model_cfg["name"] == "head" else model_cfg.get("head", {})
        )
        data_kwargs = cfg.get("data", {}).get("data_kwargs", {})
        emb = data_kwargs.get("embeddings", None)
        pool = data_kwargs.get("pool", None)
        if "MarginalPAM" in cfg.get("data", {}).get("name", ""):
            head_cfg["output_dim"] = [10, 4]
        train_kwargs = cfg.get("train", {}).get("trainer_kwargs", {})
        if cfg.get("train", {}).get("loss", None) in ["quantile"]:
            quantiles = json.loads(train_kwargs.get("quantiles_json", "[0.1, 0.9]"))
            head_cfg["output_dim"] = (
                len(quantiles) if isinstance(quantiles, list) else 1
            )
        if train_kwargs.get("use_cts_feats"):
            head_cfg["ple_input_dim"] = 10
        return get_esm(model_cfg, embeddings=emb, pool=pool, headless=headless)
    else:
        raise ValueError(f"Invalid model.name {model_cfg['name']}")


def construct_ensemble(configs: List[Mapping[str, Any]]):
    """Construct an ensemble of models from a list of configurations."""
    full_models, base_heads = [], {}
    for config in configs:
        model = get_base_model(config)
        if isinstance(model, (ESMPlusHead)):
            full_models.append(model)
        else:
            esm_chkpt = config["model"].get("checkpoint", None)
            pool = config["data"].get("data_kwargs", {}).get("pool", None)
            k = (esm_chkpt, pool)
            if k not in base_heads:
                base = get_base_model(config, headless=True)
                base_heads[k] = [base]
            base_heads[k].append(model)

    for (_, pool), b_h in base_heads.items():
        base, heads = b_h[0], b_h[1:]
        head = Ensemble(heads) if len(heads) > 1 else heads[0]
        model = ESMPlusHead(esm=base, head=head, pool=pool)
        full_models.append(model)

    return Ensemble(full_models) if len(full_models) > 1 else full_models[0]


def ensemble_from_config(config: Union[str, Dict], model_path: str = None, no_grad=True):
    """
    Parses a config file specifiying multiple models, and creates an ensemble of those models.
    Optionally loads model state from the specified path.
    """
    yaml = YAML(typ="safe")
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.load(f)
    model = construct_ensemble(process_config(config))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    if no_grad:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model.to(device="cuda" if torch.cuda.is_available() else "cpu")
