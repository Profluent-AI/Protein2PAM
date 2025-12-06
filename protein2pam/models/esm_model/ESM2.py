"""
This module contains the ESM2 model class, which is used to load and use the ESM2 model.
"""

from types import MethodType
from typing import List, Tuple, Union

import esm
import torch
import torch.nn as nn
import torch.nn.functional as F

from protein2pam.models.esm_model.esm_utils import override_forward

MODEL_CREATION_DICT = {
    "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
}


class ESM2(nn.Module):
    """ESM2 model."""
    def __init__(
        self,
        model="esm2_t33_650M_UR50D",
        checkpoint=None,
        device: str = "cpu",
        eval=True,
        trainable_heads=True,
    ):
        super().__init__()
        # check if cuda is available
        if device != "cpu" and not torch.cuda.is_available():
            print("CUDA is not available. Switching to CPU.")
            device = "cpu"

        self.esm_model_name = model

        self.model, self.alphabet = MODEL_CREATION_DICT[model]()
        self.model.forward = MethodType(override_forward, self.model)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.idx_to_tok = {v: k for k, v in self.alphabet.tok_to_idx.items()}

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.load_state_dict(checkpoint)

        self.to(device)
        if eval:
            self.eval()

        if not trainable_heads:
            for param in self.model.lm_head.parameters():
                param.requires_grad = False
            for param in self.model.contact_head.parameters():
                param.requires_grad = False

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def forward(
        self,
        batch_tokens,
        keep_cls_eos=False,
        squeeze_batchdim=True,
        make_inputs_one_hot=False,
    ):
        """Forward pass of the ESM2 model."""
        repr_layer = self.model.num_layers
        if make_inputs_one_hot:
            batch_tokens = F.one_hot(
                batch_tokens, self.model.embed_tokens.num_embeddings
            ).to(self.model.embed_tokens.weight)
            batch_tokens.requires_grad = torch.is_grad_enabled()
        results = self.model(
            batch_tokens, repr_layers=[repr_layer], return_contacts=False
        )
        embeddings = results["representations"][repr_layer]
        if not keep_cls_eos:
            embeddings = embeddings[:, 1:-1]
        if squeeze_batchdim:
            embeddings = embeddings.squeeze_(0)
        return {
            "logits": results["logits"],
            "embeddings": embeddings,
            "inputs": batch_tokens,
        }

    def embed(self, sequence: str, average=False):
        """Embed a sequence."""
        input = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(input)
        batch_tokens = batch_tokens.to(self.device)

        # Extract per-residue representations (on CPU)
        embeddings = self(batch_tokens)["embeddings"]
        return embeddings.mean(dim=0) if average else embeddings

    def mask_sequence_at_positions(
        self,
        sequence: str,
        positions: List[int],
    ):
        """Mask a sequence at given positions."""
        idx_to_tok = {v: k for k, v in self.alphabet.tok_to_idx.items()}
        mask_token = idx_to_tok[self.alphabet.mask_idx]

        # Mask the sequence at the given positions
        masked_sequence = list(sequence)
        for position in positions:
            masked_sequence[position] = mask_token

        masked_sequence = "".join(masked_sequence)

        return masked_sequence

    def get_logits(self, sequences: List[Union[Tuple[str, str], str]]) -> torch.tensor:
        """Gets the esm logits for a given list of sequences.

        Args:
            sequences List[Union[Tuple[str, str], str]]): Either a list of tuples(seq_id, seq) or a list of sequences

        Returns:
            torch.tensor : tensor of logits for the provided sequences. The first and last logits are for the cls and eos tokens respectively
        """

        if isinstance(sequences[0], str):
            sequences = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            return self(batch_tokens)["logits"]

    def get_masked_residue_logits(
        self,
        sequence,
        batch_size=None,
    ):
        """Get the masked residue logits for a given sequence."""
        masked_sequences = []
        for i in range(len(sequence)):
            masked_sequences.append(
                (f"mask_{i}", self.mask_sequence_at_positions(sequence, [i]))
            )

        if exists(batch_size):
            residue_logits = []
            for i in range(0, len(masked_sequences), batch_size):
                batch = masked_sequences[i : i + batch_size]
                residue_logits_ = self.get_logits(batch)
                residue_logits.append(residue_logits_)

            residue_logits = torch.cat(residue_logits, dim=0)
        else:
            residue_logits = self.get_logits(masked_sequences)

        residue_logits = residue_logits[:, 1:-1]  # Remove start and end tokens

        # Get logits for masked residues
        residue_logits = torch.diagonal(residue_logits).transpose(-1, -2)

        return residue_logits

    def get_masked_residue_ll(
        self,
        sequence,
        batch_size=None,
        return_entropy=False,
    ):
        """Get the masked residue log likelihood for a given sequence."""
        logits = self.get_masked_residue_logits(
            sequence,
            batch_size=batch_size,
        )

        _, _, tokens = self.batch_converter([["seq", sequence]])
        tokens = tokens[0, 1:-1]
        tokens = tokens.to(self.device)
        ll = -1 * torch.nn.functional.cross_entropy(
            input=logits,
            target=tokens.T,
            weight=None,
            size_average=None,
            reduction="none",
        )

        if return_entropy:
            probs = torch.softmax(logits, dim=-1)
            entropy = -1 * (probs * torch.log(probs)).sum(dim=-1)

            return ll, entropy

        return ll

    def get_unmasked_residue_ll(
        self,
        sequences: Union[str, List[str]],
        batch_size=None,
        return_entropy=False,
    ):
        """Get the unmasked residue log likelihood for a given sequence."""
        if isinstance(sequences, str):
            sequences = [sequences]

        sequences_batch = [
            [f"seq{i}", sequence] for i, sequence in enumerate(sequences)
        ]
        if exists(batch_size):
            logits = []
            for i in range(0, len(sequences_batch), batch_size):
                batch = sequences_batch[i : i + batch_size]
                logits_ = self.get_logits(batch)
                logits.append(logits_)

            logits = padcat(logits, val=0, dim=0)
            # logits = torch.cat(logits, dim=0)
        else:
            logits = self.get_logits(sequences_batch)

        logits = logits[:, 1:-1]  # Remove start and end tokens
        logits = logits.transpose(-1, -2)  # Reshape to (batch, vocab, seq_len)

        _, _, tokens = self.batch_converter(sequences_batch)
        tokens = tokens[:, 1:-1]
        tokens = tokens.to(self.device)
        ll = -1 * torch.nn.functional.cross_entropy(
            input=logits,
            target=tokens,
            weight=None,
            size_average=None,
            reduction="none",
        )

        if return_entropy:
            probs = torch.softmax(logits, dim=-1)
            entropy = -1 * (probs * torch.log(probs)).sum(dim=-1)

            return ll, entropy

        return ll

    def masked_fill(
        self,
        sequence: str,
        positions: List[int],
    ):
        """Masked fill a sequence."""
        masked_sequence = self.mask_sequence_at_positions(sequence, positions)

        logits = self.get_logits([("", masked_sequence)])[0, 1:-1]

        idx_to_tok = {v: k for k, v in self.alphabet.tok_to_idx.items()}
        predicted_sequence = [idx_to_tok[i] for i in logits.argmax(dim=-1).tolist()]
        predicted_sequence = "".join(predicted_sequence)

        filled_sequence = [
            predicted_sequence[i] if i in positions else sequence[i]
            for i in range(len(sequence))
        ]
        filled_sequence = "".join(filled_sequence)

        return filled_sequence

    def iterative_masked_fill(
        self,
        sequence: str,
        positions: List[int],
    ):
        """Iteratively masked fill a sequence."""
        positions = set(positions)
        while len(positions) > 0:
            position_list = list(positions)
            masked_sequence = self.mask_sequence_at_positions(sequence, position_list)
            logits = self.get_logits([("", masked_sequence)])[0, 1:-1]

            # set to neg inf
            unmasked_pos = torch.ones(len(sequence), device=self.device) * -1e9
            unmasked_pos[position_list] = 1
            logits = logits + unmasked_pos.unsqueeze(-1)

            # get the position with the highest logits value
            sample_pos = logits.max(dim=-1)[0].argmax().item()
            positions.remove(sample_pos)
            sample_res = self.idx_to_tok[logits[sample_pos].argmax().item()]

            # fill in the position
            sequence = sequence[:sample_pos] + sample_res + sequence[sample_pos + 1 :]

        return sequence
