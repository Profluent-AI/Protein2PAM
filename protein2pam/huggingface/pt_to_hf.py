"""Utilities to convert a PyTorch state dict for the fair-esm format to a Huggingface model."""
import os
import warnings

import torch

from protein2pam.huggingface.configuration_esm import EsmConfig
from protein2pam.huggingface.modeling_esm import EsmForSequenceClassification

_config_path = os.path.join(os.path.dirname(__file__), "config.json")

def state_dict_key_transform(key: str) -> str:
    if key.startswith("esm.model"):
        key = key.replace("esm.model", "esm.encoder")
        key = key.replace("layers.", "layer.")
        key = key.replace("self_attn.", "attention.")
        key = key.replace("q_proj", "self.query")
        key = key.replace("k_proj", "self.key")
        key = key.replace("v_proj", "self.value")
        key = key.replace("rot_emb", "self.rotary_embeddings")
        key = key.replace("attention.out_proj", "attention.output.dense")
        key = key.replace("fc1", "intermediate.dense")
        key = key.replace("fc2", "output.dense")
        key = key.replace("self_attn_layer_norm", "attention.LayerNorm")
        key = key.replace("final_layer_norm", "LayerNorm")
    key = key.replace("esm.encoder.embed_tokens", "esm.embeddings.word_embeddings")
    key = key.replace("esm.encoder.contact_head", "esm.contact_head")
    if key.startswith("head."):
        key = "classifier." + key[len("head."):]
        if key.startswith("classifier.layers."):
            key_split = key.split(".")
            key_split[2] = str(int(key_split[2]) + 3)
            key = ".".join(key_split)
        key = key.replace("dense", "layers.1")
    return key


def pt_to_hf(pt_path: str) -> EsmForSequenceClassification:
    state_dict = {state_dict_key_transform(k): v for k, v in torch.load(pt_path).items()}
    model = EsmForSequenceClassification.from_pretrained(
        "facebook/esm2_t33_650M_UR50D", config=EsmConfig.from_pretrained(_config_path)
    )
    ret = model.load_state_dict(state_dict, strict=False)
    warnings.warn(f"Missing keys: {ret.missing_keys}\n\nUnexpected keys:{ret.unexpected_keys}\n")
    return model
