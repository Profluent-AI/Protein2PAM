"""
This module contains the ESM model utils.
"""

from typing import List

import esm
import torch

from protein2pam.utils.common import exists

MODEL_CREATION_DICT = {
    "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
}


def load_model_components(model="esm2_t33_650M_UR50D", device=None, eval=True):
    """Load the ESM-2 model components."""
    # Load ESM-2 model
    model, alphabet = MODEL_CREATION_DICT[model]()
    batch_converter = alphabet.get_batch_converter()

    if exists(device):
        model = model.to(device)

    if eval:
        model.eval()  # disables dropout for deterministic results

    return model, alphabet, batch_converter


def mask_sequence_at_positions(
    sequence: str, positions: List[int], alphabet: esm.Alphabet
):
    """Mask a sequence at given positions."""
    idx_to_tok = {v: k for k, v in alphabet.tok_to_idx.items()}
    mask_token = idx_to_tok[alphabet.mask_idx]

    # Mask the sequence at the given positions
    masked_sequence = list(sequence)
    for position in positions:
        masked_sequence[position] = mask_token

    masked_sequence = "".join(masked_sequence)

    return masked_sequence


def get_logits(model, batch_converter, sequences, device=None):
    """Get the logits for a given sequence."""
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

    if exists(device):
        batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    return results["logits"]


def override_forward(
    self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
):
    """Override the forward method of the ESM-2 model."""
    if return_contacts:
        need_head_weights = True

    if tokens.ndim == 2 and not torch.is_floating_point(tokens):
        padding_mask = tokens.eq(self.padding_idx)  # B, T
        x = self.embed_scale * self.embed_tokens(tokens)
    elif tokens.ndim == 3:
        padding_mask = tokens[:, :, self.padding_idx] != 0  # B, T, V
        x = self.embed_scale * (tokens @ self.embed_tokens.weight)
    else:
        raise ValueError("Expected tokens to be either integers or one-hot.")

    if self.token_dropout:
        if tokens.ndim == 2:
            mask = tokens == self.mask_idx
        else:
            mask = tokens[:, :, self.mask_idx] != 0
        x.masked_fill_(mask.unsqueeze(-1), 0.0)
        # x: B x T x C
        mask_ratio_train = 0.15 * 0.8
        src_lengths = (~padding_mask).sum(-1)
        mask_ratio_observed = mask.sum(-1).to(x.dtype) / src_lengths
        x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            # (H, B, T, T) => (B, H, T, T)
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        # attentions: B x L x H x T x T
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result
