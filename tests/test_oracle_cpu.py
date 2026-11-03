"""The oracles should run on CPU-only machines (issue #1)."""

import torch
import torch.nn as nn

from protein2pam.models.oracle import MarginalPAMInfoSimOracle, MarginalPAMOracle


class TinyPAMModel(nn.Module):
    """Stands in for ESMPlusHead: list of sequences -> (batch, positions, 4) logits."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 8 * 4)

    def forward(self, seqs):
        x = torch.tensor([[float(len(s))] for s in seqs])
        return self.lin(x).view(len(seqs), 8, 4)


class TinyConfidenceModel(nn.Module):
    """Stands in for the confidence model: (sequences, ten_nn_pct_id) -> (batch, 1)."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, seqs, ten_nn_pct_id):
        return torch.sigmoid(self.lin(ten_nn_pct_id.float()))


def test_marginal_pam_oracle_runs_on_cpu():
    oracle = MarginalPAMOracle.__new__(MarginalPAMOracle)
    oracle.model = TinyPAMModel()
    oracle.model_name = "tiny"

    result = oracle.evaluate(["MKTAYIAK", "MKTAY"])

    assert result["predictions"].shape == (2, 8, 4)
    assert result["model_name"] == "tiny"


def test_info_sim_oracle_runs_on_cpu():
    oracle = MarginalPAMInfoSimOracle.__new__(MarginalPAMInfoSimOracle)
    oracle.model = TinyConfidenceModel()
    # evaluate() keeps the 10 nearest training sequences, so give it at least 10
    oracle.train_seqs = ["MKTAYIAK" + "A" * i for i in range(12)]

    result = oracle.evaluate(["MKTAYIAK", "MKTAY"], ten_nn_pct_id=None, train_seqs=None)

    assert result["confidence_scores"].shape == (2,)
    assert result["percent_identities"].shape == (2, 10)
