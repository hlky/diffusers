import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.modeling_outputs import Transformer2DModelOutput
from ...models.modeling_utils import ModelMixin
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, HiDreamImageFeedForwardSwiGLU
from ..embeddings import (
    FluxPosEmbed,
    HiDreamImageOutEmbed,
    HiDreamImagePooledTimestepEmbed,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class HiDreamAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        self.to_q = nn.Linear(query_dim, self.inner_dim)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim)
        self.to_out = nn.Linear(self.inner_dim, self.out_dim)
        self.q_rms_norm = nn.RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = nn.RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = nn.Linear(query_dim, self.inner_dim)
            self.to_k_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_v_t = nn.Linear(self.inner_dim, self.inner_dim)
            self.to_out_t = nn.Linear(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = nn.RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        norm_image_tokens: torch.FloatTensor,
        image_tokens_masks: torch.FloatTensor = None,
        norm_text_tokens: torch.FloatTensor = None,
        image_rotary_emb: torch.FloatTensor = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states=norm_image_tokens,
            hidden_states_mask=image_tokens_masks,
            encoder_hidden_states=norm_text_tokens,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: HiDreamAttention,
        hidden_states: torch.FloatTensor,
        hidden_states_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        image_rotary_emb: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        query = attn.q_rms_norm(attn.to_q(hidden_states)).to(dtype=dtype)
        key = attn.k_rms_norm(attn.to_k(hidden_states)).to(dtype=dtype)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if hidden_states_mask is not None:
            key = key * hidden_states_mask.view(batch_size, -1, 1, 1)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.q_rms_norm_t(attn.to_q_t(encoder_hidden_states)).to(dtype=dtype)
            encoder_hidden_states_key_proj = attn.k_rms_norm_t(attn.to_k_t(encoder_hidden_states)).to(dtype=dtype)
            encoder_hidden_states_value_proj = attn.to_v_t(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            num_image_tokens = query.shape[2]
            num_text_tokens = encoder_hidden_states_query_proj.shape[2]
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            query = query
            key = key
            value = value

        if image_rotary_emb is not None:
            from ..embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = torch.split(
                hidden_states, [num_image_tokens, num_text_tokens], dim=1
            )
            hidden_states = attn.to_out(hidden_states)
            encoder_hidden_states = attn.to_out_t(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # print(bsz, seq_len, h)
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
    ):
        super().__init__()
        self.shared_experts = HiDreamImageFeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            [HiDreamImageFeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)]
        )
        self.gate = MoEGate(
            embed_dim=dim, num_routed_experts=num_routed_experts, num_activated_experts=num_activated_experts
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype)
            # y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


@maybe_allow_in_graph
class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=True,
        )

        # 3. Feed-forward
        self.norm_out = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
            )
        else:
            self.ff = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        image_rotary_emb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input)[
            :, None
        ].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa) + shift_msa
        attn_output = self.attn(
            norm_image_tokens,
            image_tokens_masks,
            image_rotary_emb=image_rotary_emb,
        )
        image_tokens = gate_msa * attn_output + image_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm_out(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp) + shift_mlp
        ff_output = gate_mlp * self.ff(norm_image_tokens.to(dtype=wtype))
        image_tokens = ff_output + image_tokens
        return image_tokens


@maybe_allow_in_graph
class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # 1. Attention
        self.norm_image = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm_text = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=False,
        )

        # 3. Feed-forward
        self.norm_out = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_image = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
            )
        else:
            self.ff_image = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)
        self.norm_out_text = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_text = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        adaln_input: Optional[torch.FloatTensor] = None,
        image_rotary_emb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        wtype = image_tokens.dtype
        (
            shift_msa_image,
            scale_msa_image,
            gate_msa_image,
            shift_mlp_image,
            scale_mlp_image,
            gate_mlp_image,
            shift_msa_text,
            scale_msa_text,
            gate_msa_text,
            shift_mlp_text,
            scale_mlp_text,
            gate_mlp_text,
        ) = self.adaLN_modulation(adaln_input)[:, None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_image_tokens = self.norm_image(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_msa_image) + shift_msa_image
        norm_text_tokens = self.norm_text(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_msa_text) + shift_msa_text

        attn_output_image, attn_output_text = self.attn(
            norm_image_tokens,
            image_tokens_masks,
            norm_text_tokens,
            image_rotary_emb=image_rotary_emb,
        )

        image_tokens = gate_msa_image * attn_output_image + image_tokens
        text_tokens = gate_msa_text * attn_output_text + text_tokens

        # 2. Feed-forward
        norm_image_tokens = self.norm_out(image_tokens).to(dtype=wtype)
        norm_image_tokens = norm_image_tokens * (1 + scale_mlp_image) + shift_mlp_image
        norm_text_tokens = self.norm_out_text(text_tokens).to(dtype=wtype)
        norm_text_tokens = norm_text_tokens * (1 + scale_mlp_text) + shift_mlp_text

        ff_output_image = gate_mlp_image * self.ff_image(norm_image_tokens)
        ff_output_text = gate_mlp_text * self.ff_text(norm_text_tokens)
        image_tokens = ff_output_image + image_tokens
        text_tokens = ff_output_text + text_tokens
        return image_tokens, text_tokens


class HiDreamImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageTransformerBlock", "HiDreamImageSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.llama_layers = llama_layers

        self.t_embedder = HiDreamImagePooledTimestepEmbed(self.inner_dim, text_emb_dim)
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, self.inner_dim, bias=True)
        self.pe_embedder = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.transformer_blocks = nn.ModuleList(
            [
                HiDreamImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    num_routed_experts=num_routed_experts,
                    num_activated_experts=num_activated_experts,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                HiDreamImageSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    num_routed_experts=num_routed_experts,
                    num_activated_experts=num_activated_experts,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        self.final_layer = HiDreamImageOutEmbed(self.inner_dim, patch_size, self.out_channels)

        caption_channels = [
            caption_channels[1],
        ] * (num_layers + num_single_layers) + [
            caption_channels[0],
        ]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def expand_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            is_mps = device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(batch_size)
        return timesteps

    def patchify(self, x: Union[torch.Tensor, List[torch.Tensor]], max_seq: int, image_sizes=None):
        pz = self.config.patch_size
        pz2 = pz * pz

        if isinstance(x, torch.Tensor):
            B, C, H, W = x.shape
            device = x.device
            dtype = x.dtype
        else:
            B = len(x)
            C, H, W = x[0].shape
            device = x[0].device
            dtype = x[0].dtype
            x = torch.stack(x, dim=0)

        if image_sizes is not None:
            x_masks = torch.zeros((B, max_seq), dtype=dtype, device=device)
            for i, (ph, pw) in enumerate(image_sizes):
                x_masks[i, : ph * pw] = 1

            B, C, S, _ = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, S, pz2 * C)
        else:
            assert H % pz == 0 and W % pz == 0, "Image dimensions must be divisible by patch size"
            H_patches = H // pz
            W_patches = W // pz
            S = H_patches * W_patches

            x = x.unfold(2, pz, pz).unfold(3, pz, pz)  # (B, C, H_p, W_p, pz, pz)
            x = x.permute(0, 2, 3, 4, 5, 1)  # (B, H_p, W_p, pz, pz, C)
            x = x.reshape(B, S, pz2 * C)  # (B, S, pz2*C)
            x_masks = None
            image_sizes = [(H_patches, W_patches)] * B

        return x, x_masks, image_sizes

    def unpatchify(self, x: torch.Tensor, image_sizes: List[Tuple[int, int]], is_training: bool) -> torch.Tensor:
        pz = self.config.patch_size
        pz2 = pz * pz
        C = x.shape[-1] // pz2

        if is_training:
            # x: (B, S, pz2*C) -> (B, C, S, pz2)
            x = x.reshape(x.shape[0], x.shape[1], pz2, C).permute(0, 3, 1, 2)
        else:
            x_arr = []
            for i, (ph, pw) in enumerate(image_sizes):
                S = ph * pw
                xi = x[i, :S].reshape(ph, pw, pz, pz, C)  # (ph, pw, pz, pz, C)
                xi = xi.permute(4, 0, 2, 1, 3).reshape(C, ph * pz, pw * pz)  # (C, H, W)
                x_arr.append(xi.unsqueeze(0))  # (1, C, H, W)

            x = torch.cat(x_arr, dim=0)  # (B, C, H, W)

        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        img_ids: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        adaln_input = self.t_embedder(timesteps, pooled_embeds, hidden_states_type)

        hidden_states, image_tokens_masks, image_sizes = self.patchify(hidden_states, self.max_seq, image_sizes)
        if image_tokens_masks is None:
            patch_H, patch_W = image_sizes[0]
            img_ids = torch.zeros(patch_H, patch_W, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(patch_H)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(patch_W)[None, :]

            img_ids_height, img_ids_width, img_ids_channels = img_ids.shape

            img_ids = img_ids.reshape(img_ids_height * img_ids_width, img_ids_channels)
        hidden_states = self.x_embedder(hidden_states)

        T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states[-1]
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            encoder_hidden_states[-1].shape[1]
            + encoder_hidden_states[-2].shape[1]
            + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device,
            dtype=img_ids.dtype,
        )
        ids = torch.cat((img_ids, txt_ids), dim=0)
        image_rotary_emb = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.transformer_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat(
                [initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1
            )
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, initial_encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    image_tokens_masks,
                    cur_encoder_hidden_states,
                    adaln_input,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, initial_encoder_hidden_states = block(
                    image_tokens=hidden_states,
                    image_tokens_masks=image_tokens_masks,
                    text_tokens=cur_encoder_hidden_states,
                    adaln_input=adaln_input,
                    image_rotary_emb=image_rotary_emb,
                )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if image_tokens_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=image_tokens_masks.device,
                dtype=image_tokens_masks.dtype,
            )
            image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_transformer_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    image_tokens_masks,
                    None,
                    adaln_input,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    image_tokens=hidden_states,
                    image_tokens_masks=image_tokens_masks,
                    text_tokens=None,
                    adaln_input=adaln_input,
                    image_rotary_emb=image_rotary_emb,
                )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(output, image_sizes, self.training)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
