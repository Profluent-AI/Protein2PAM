"""
This module contains the ESM model class, which is used to load and use the ESM2 model.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from protein2pam.models.esm_model.ESM2 import ESM2
from protein2pam.models.embedding import PLE


def get_esm(model_cfg, embeddings=None, pool=None, headless=False):
    """Get the ESM model."""
    size2name = {
        "xsmall": "esm2_t6_8M_UR50D",
        "small": "esm2_t30_150M_UR50D",
        "medium": "esm2_t33_650M_UR50D",
        "large": "esm2_t36_3B_UR50D",
        "xlarge": "esm2_t48_15B_UR50D",
    }
    size2dim = {"medium": 1280}
    size = model_cfg.get("size", "medium")
    assert (
        size in size2name
    ), f"Expected model.size to be one of {size2name.keys()}, but got {size}"
    checkpoint = model_cfg.get("checkpoint", None)
    if checkpoint is not None:
        if checkpoint.startswith("gs://"):
            path = os.path.join(str(Path.home()), ".cache", checkpoint[len("gs://") :])
            if not os.path.isfile(path + ".success"):
                subprocess.run(["gcloud", "storage", "cp", checkpoint, path])
                Path(path + ".success").touch()
            checkpoint = path

    head_cfg = model_cfg if model_cfg["name"] == "head" else model_cfg.get("head", {})
    head_name = head_cfg.pop("name", None)
    headless = (headless or head_name is None) and model_cfg["name"] != "head"
    include_base = (headless or embeddings is None) and model_cfg["name"] != "head"

    # Get base model if needed, and compute embedding dimension
    if model_cfg["name"] != "head" and (include_base or size not in size2dim):
        base_model = ESM2(
            model=size2name[size],
            checkpoint=checkpoint,
            device="cpu",
            trainable_heads=headless,
        )
        for param in base_model.model.contact_head.parameters():
            param.requires_grad = False
        if headless:
            return base_model
        embed_dim = base_model.model.embed_dim
    elif not headless:
        embed_dim = size2dim[size]

    # Get head
    if include_base:
        checkpoint = head_cfg.pop("checkpoint", None)
    else:
        checkpoint = None
    input_dim = 0 if model_cfg["name"] == "head" else embed_dim
    if head_name == "Linear":
        head_cfg["hidden_layers"] = 0
        head_cfg["activation"] = "identity"
        head_cfg["embed_dim"] = head_cfg["output_dim"]
        head = MLPHead(input_dim=input_dim, **head_cfg)
    elif head_name == "MLP":
        head = MLPHead(input_dim=input_dim, **head_cfg)
    else:
        raise ValueError(f"Invalid model.head.name {head_name}")

    # Return just head if using embeddings, full model otherwise
    if not include_base:
        return head
    freeze_esm = model_cfg.get("freeze_esm", False)
    return ESMPlusHead(
        esm=base_model,
        head=head,
        pool=pool,
        freeze_esm=freeze_esm,
        checkpoint=checkpoint,
    )


class MLPHead(nn.Module):
    """Multi-layer perceptron head."""
    def __init__(
        self,
        input_dim,
        embed_dim=None,
        ple_input_dim=None,
        ple_minval=0.0,
        ple_maxval=1.0,
        ple_n_bins=20,
        dropout=0.2,
        output_dim=1,
        hidden_layers=0,
        activation="tanh",
        output_activation="identity",
        checkpoint=None,
    ):
        super().__init__()
        if input_dim == 0:
            assert embed_dim is not None
        self.input_dim = input_dim
        self.output_dim = (
            [output_dim] if not isinstance(output_dim, (list, tuple)) else output_dim
        )
        self.dropout = nn.Dropout(dropout)
        embed_dim = embed_dim or input_dim

        if ple_input_dim is not None and ple_n_bins is not None:
            self.ple = nn.Sequential(
                PLE(
                    minval=ple_minval,
                    maxval=ple_maxval,
                    nbins=ple_n_bins,
                    flatten_bin_dim=True,
                ),
                nn.Linear(ple_input_dim * ple_n_bins, embed_dim),
            )
        else:
            self.ple = None

        self.activation = self.get_activation(activation)
        self.dense = nn.Linear(input_dim, embed_dim)
        layers = sum(
            [
                [self.dropout, nn.Linear(embed_dim, embed_dim), self.activation]
                for _ in range(hidden_layers)
            ],
            [],
        )
        self.layers = None if len(layers) == 0 else nn.Sequential(*layers)
        self.out_proj = nn.Linear(embed_dim, np.prod(self.output_dim))
        self.output_activation = self.get_activation(output_activation)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.load_state_dict(state_dict)

    @staticmethod
    def get_activation(activation):
        if activation == "identity":
            return nn.Identity()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Invalid activation {activation}")

    def forward(self, embeddings, continuous_input=None):
        """Forward pass of the MLP head."""
        if self.input_dim != 0:
            x = self.activation(self.dense(self.dropout(embeddings)))
        else:
            x = None
        if self.ple is not None:
            if continuous_input is None:
                raise ValueError("`continuous_input` is required if PLE is initialized")
            continuous_input = continuous_input.to(x) if x is not None else x
            ple = self.ple(continuous_input)
            x = ple if x is None else ple + x
        x = x if self.layers is None else self.layers(x)
        return self.output_activation(self.out_proj(self.dropout(x))).reshape(
            -1, *self.output_dim
        )


class ESMPlusHead(nn.Module):
    """ESM model with a head."""
    def __init__(
        self,
        esm: ESM2,
        head: nn.Module,
        pool: str = "cls",
        freeze_esm=False,
        checkpoint=None,
    ):
        super().__init__()
        self.esm = esm
        self.head = head
        self.pool = pool

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location="cpu")
            self.load_state_dict(state_dict)

        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False

    def forward(self, seqs, *args, return_inputs=False, **kwargs):
        """Forward pass of the ESMPlusHead."""
        seqs = [seqs] if isinstance(seqs, str) else seqs
        _, _, batch_tokens = self.esm.batch_converter(
            [(str(i), seq) for i, seq in enumerate(seqs)]
        )
        batch_tokens = batch_tokens.to(next(self.parameters()).device)
        esm_output = self.esm(
            batch_tokens, keep_cls_eos=True, make_inputs_one_hot=return_inputs
        )
        embeddings = esm_output["embeddings"]
        embeddings = embeddings.reshape(len(seqs), -1, embeddings.shape[-1])
        if self.pool == "cls":
            embeddings = embeddings[:, 0]
        elif self.pool == "mean":
            embeddings = torch.stack(
                [e[: len(seq)].mean(dim=0) for e, seq in zip(embeddings, seqs)]
            )
        elif self.pool == "max":
            embeddings = torch.stack(
                [e[: len(seq)].mean(dim=0) for e, seq in zip(embeddings, seqs)]
            )
        output = self.head(embeddings, *args, **kwargs)
        return (output, esm_output["inputs"]) if return_inputs else output
