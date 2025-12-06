"""
This module contains the PLE (Piecewise Linear Embedding) class, which is used to embed continuous values into a discrete space.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PLE(nn.Module):
    """
    Piecewise linear embedding of continuous values. From Gorishniy et al.,
    "On Embeddings for Numerical Features in Tabular Deep Learning," 2022.
    """

    def __init__(self, minval=0.0, maxval=1.0, nbins=20, flatten_bin_dim=True):
        """Initialize the PLE."""
        super().__init__()
        self.minval = minval
        self.maxval = maxval
        self.nbins = nbins
        self.flatten_bin_dim = flatten_bin_dim
        self.bins = torch.linspace(minval, maxval, nbins)
        self.bin_prefix = torch.zeros(nbins, nbins)
        for i in range(nbins):
            self.bin_prefix[i, : i - 1] = 1

    def forward(self, values: torch.Tensor):
        """Forward pass of the PLE."""
        if (values > self.maxval).any() or (values < self.minval).any():
            logger.warning(
                f"Expected values between {self.minval} and {self.maxval} for PLE, "
                f"but got values between {values.min().item()} and {values.max().item()}."
            )
        bins = self.bins.to(values)
        bin_prefix = self.bin_prefix.to(values)
        js = torch.searchsorted(bins, values.flatten())
        js = torch.maximum(
            torch.minimum(js, torch.tensor(self.nbins) - 1), torch.tensor(1)
        )
        output = bin_prefix[js - 1]
        output[range(len(output)), js - 1] = (values.flatten() - bins[js - 1]) / (
            bins[js] - bins[js - 1]
        )
        if self.flatten_bin_dim:
            return output.reshape(*values.shape[:-1], -1)
        return output.reshape(*values.shape, self.nbins)
