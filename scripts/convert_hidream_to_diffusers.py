import argparse
from contextlib import nullcontext

import safetensors.torch
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download

from diffusers import HiDreamImageTransformer2DModel
from diffusers.utils.import_utils import is_accelerate_available


"""
# Transformer

python scripts/convert_hidream_to_diffusers.py  \
--original_state_dict_repo_id "HiDream-ai/HiDream-I1-Full"
"""

CTX = init_empty_weights if is_accelerate_available() else nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--original_state_dict_repo_id", default=None, type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument("--dtype", type=str, default="bf16")

args = parser.parse_args()
dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32


def load_original_checkpoint(args):
    state_dict = {}
    files = [
        "diffusion_pytorch_model-00001-of-00007.safetensors",
        "diffusion_pytorch_model-00002-of-00007.safetensors",
        "diffusion_pytorch_model-00003-of-00007.safetensors",
        "diffusion_pytorch_model-00004-of-00007.safetensors",
        "diffusion_pytorch_model-00005-of-00007.safetensors",
        "diffusion_pytorch_model-00006-of-00007.safetensors",
        "diffusion_pytorch_model-00007-of-00007.safetensors",
    ]
    for file in files:
        if args.original_state_dict_repo_id is not None:
            ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=file, subfolder="transformer")
        elif args.checkpoint_path is not None:
            ckpt_path = args.checkpoint_path + f"/{file}"
        else:
            raise ValueError(" please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")
        original_state_dict = safetensors.torch.load_file(ckpt_path)
        state_dict.update(original_state_dict)

    print(f"got {state_dict.keys()}")

    return state_dict

def convert_hidream_transformer_checkpoint_to_diffusers(
    original_state_dict, num_layers, num_single_layers, num_experts
):
    converted_state_dict = {}

    for k in [
        "t_embedder.timestep_embedder.linear_1.weight",
        "t_embedder.timestep_embedder.linear_1.bias",
        "t_embedder.timestep_embedder.linear_2.weight",
        "t_embedder.timestep_embedder.linear_2.bias",
    ]:
        converted_state_dict[k] = original_state_dict.pop(k)

    for suffix in [
        "pooled_embedder.linear_1.weight",
        "pooled_embedder.linear_1.bias",
        "pooled_embedder.linear_2.weight",
        "pooled_embedder.linear_2.bias",
    ]:
        orig_key = "p_embedder." + suffix
        new_key = "t_embedder." + suffix
        converted_state_dict[new_key] = original_state_dict.pop(orig_key)

    for attr in ["weight", "bias"]:
        orig_key = "x_embedder.proj." + attr
        new_key = "x_embedder." + attr
        converted_state_dict[new_key] = original_state_dict.pop(orig_key)

    for i in range(num_layers):
        old_prefix = f"double_stream_blocks.{i}.block."
        new_prefix = f"transformer_blocks.{i}."

        for attr in ["weight", "bias"]:
            key_old = old_prefix + "adaLN_modulation.1." + attr
            key_new = new_prefix + "adaLN_modulation.1." + attr
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["to_q", "to_k", "to_v", "to_out"]:
            for attr in ["weight", "bias"]:
                key_old = old_prefix + "attn1." + sub + "." + attr
                key_new = new_prefix + "attn." + sub + "." + attr
                converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["q_rms_norm", "k_rms_norm"]:
            key_old = old_prefix + "attn1." + sub + ".weight"
            key_new = new_prefix + "attn." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["to_q_t", "to_k_t", "to_v_t", "to_out_t"]:
            for attr in ["weight", "bias"]:
                key_old = old_prefix + "attn1." + sub + "." + attr
                key_new = new_prefix + "attn." + sub + "." + attr
                converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["q_rms_norm_t", "k_rms_norm_t"]:
            key_old = old_prefix + "attn1." + sub + ".weight"
            key_new = new_prefix + "attn." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["shared_experts.w1", "shared_experts.w2", "shared_experts.w3"]:
            key_old = old_prefix + "ff_i." + sub + ".weight"
            key_new = new_prefix + "ff_image." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for expert in range(num_experts):
            for sub in ["w1", "w2", "w3"]:
                key_old = old_prefix + f"ff_i.experts.{expert}." + sub + ".weight"
                key_new = new_prefix + f"ff_image.experts.{expert}." + sub + ".weight"
                converted_state_dict[key_new] = original_state_dict.pop(key_old)

        key_old = old_prefix + "ff_i.gate.weight"
        key_new = new_prefix + "ff_image.gate.weight"
        converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["w1", "w2", "w3"]:
            key_old = old_prefix + "ff_t." + sub + ".weight"
            key_new = new_prefix + "ff_text." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

    for i in range(num_single_layers):
        old_prefix = f"single_stream_blocks.{i}.block."
        new_prefix = f"single_transformer_blocks.{i}."

        for attr in ["weight", "bias"]:
            key_old = old_prefix + "adaLN_modulation.1." + attr
            key_new = new_prefix + "adaLN_modulation.1." + attr
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["to_q", "to_k", "to_v", "to_out"]:
            for attr in ["weight", "bias"]:
                key_old = old_prefix + "attn1." + sub + "." + attr
                key_new = new_prefix + "attn." + sub + "." + attr
                converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["q_rms_norm", "k_rms_norm"]:
            key_old = old_prefix + "attn1." + sub + ".weight"
            key_new = new_prefix + "attn." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for sub in ["shared_experts.w1", "shared_experts.w2", "shared_experts.w3"]:
            key_old = old_prefix + "ff_i." + sub + ".weight"
            key_new = new_prefix + "ff." + sub + ".weight"
            converted_state_dict[key_new] = original_state_dict.pop(key_old)

        for expert in range(4):
            for sub in ["w1", "w2", "w3"]:
                key_old = old_prefix + f"ff_i.experts.{expert}." + sub + ".weight"
                key_new = new_prefix + "ff.experts." + f"{expert}" + "." + sub + ".weight"
                converted_state_dict[key_new] = original_state_dict.pop(key_old)

        key_old = old_prefix + "ff_i.gate.weight"
        key_new = new_prefix + "ff.gate.weight"
        converted_state_dict[key_new] = original_state_dict.pop(key_old)

    for key in [
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias",
    ]:
        converted_state_dict[key] = original_state_dict.pop(key)

    for key in list(original_state_dict.keys()):
        if key.startswith("caption_projection."):
            converted_state_dict[key] = original_state_dict.pop(key)

    print(f"{original_state_dict.keys()} remaining")

    return converted_state_dict



def main(args):
    original_ckpt = load_original_checkpoint(args)

    config = {
    "_class_name": "HiImageTransformer2DModel",
    "_diffusers_version": "0.32.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [
        64,
        32,
        32
    ],
    "caption_channels": [
        4096,
        4096
    ],
    "max_resolution": [
        128,
        128
    ],
    "in_channels": 16,
    "llama_layers": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31,
        31
    ],
    "num_attention_heads": 20,
    "num_routed_experts": 4,
    "num_activated_experts": 2,
    "num_layers": 16,
    "num_single_layers": 32,
    "out_channels": 16,
    "patch_size": 2,
    "text_emb_dim": 2048
    }

    converted_transformer_state_dict = convert_hidream_transformer_checkpoint_to_diffusers(
        original_ckpt, config["num_layers"], config["num_single_layers"], config["num_routed_experts"]
    )
    with CTX():
        transformer = HiDreamImageTransformer2DModel(**config)
    transformer.load_state_dict(converted_transformer_state_dict, assign=True, strict=True)

    transformer.to(dtype).save_pretrained(f"{args.output_path}/transformer", max_shard_size="5GB")

if __name__ == "__main__":
    main(args)
