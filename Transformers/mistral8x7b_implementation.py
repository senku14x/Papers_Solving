import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MixtralConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    sliding_window: int = 8192
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    router_aux_loss_coef: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
    B, H, T, D = x.shape
    assert D % 2 == 0
    cos_t = cos[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[pos_offset:pos_offset + T].unsqueeze(0).unsqueeze(0)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos_t - x2 * sin_t
    y2 = x1 * sin_t + x2 * cos_t
    y = torch.empty_like(x)
    y[..., 0::2] = y1
    y[..., 1::2] = y2
    return y


class MixtralAttention(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.repeat_interleave(self.n_rep, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        sliding_window: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        D = self.head_dim

        q = self.q_proj(x).view(B, T, self.n_heads, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, D).transpose(1, 2)

        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        q = apply_rope(q, cos, sin, pos_offset=past_len)
        k = apply_rope(k, cos, sin, pos_offset=past_len)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (k, v) if use_cache else None

        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)

        Ttot = k_full.shape[2]
        scale = 1.0 / math.sqrt(D)

        att = torch.matmul(q, k_full.transpose(-2, -1)) * scale

        device = x.device
        i = torch.arange(T, device=device).unsqueeze(-1)
        j = torch.arange(Ttot, device=device).unsqueeze(0)
        causal = j <= (past_len + i)

        win = sliding_window if sliding_window is not None else self.cfg.sliding_window
        if win is not None and win > 0:
            lower = (past_len + i) - (win - 1)
            window_ok = j >= lower
            mask = causal & window_ok
        else:
            mask = causal

        att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(att.dtype).min)
        probs = F.softmax(att, dim=-1)
        out = torch.matmul(probs, v_full)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return out, present


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w2(x)) * self.w1(x))


class MixtralMoE(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.cfg = cfg
        self.E = cfg.num_local_experts
        self.K = cfg.num_experts_per_tok

        self.router = nn.Linear(cfg.hidden_size, self.E, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(cfg.hidden_size, cfg.intermediate_size) for _ in range(self.E)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        N = x_flat.shape[0]

        logits = self.router(x_flat)
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(probs, k=self.K, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        y_flat = x_flat.new_zeros((N, C))

        for e in range(self.E):
            mask = (topk_idx == e)
            if not mask.any():
                continue
            tok_ids, k_ids = torch.where(mask)
            x_e = x_flat[tok_ids]
            y_e = self.experts[e](x_e)
            w = topk_probs[tok_ids, k_ids].unsqueeze(-1)
            y_flat.index_add_(0, tok_ids, y_e * w)

        y = y_flat.view(B, T, C)

        one_hot = F.one_hot(topk_idx, num_classes=self.E).float()
        f = one_hot.sum(dim=(0, 1)) / (N * self.K)
        p = probs.mean(dim=0)
        aux = self.E * torch.sum(f * p)

        return y, aux


class MixtralBlock(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = MixtralAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.moe = MixtralMoE(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        a, present = self.attn(self.ln1(x), cos, sin, past_kv=past_kv, use_cache=use_cache, sliding_window=self.cfg.sliding_window)
        x = x + a

        m, aux = self.moe(self.ln2(x))
        x = x + m
        return x, present, aux


class MixtralForCausalLM(nn.Module):
    def __init__(self, cfg: MixtralConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([MixtralBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        self._rope_cache: Dict[Tuple[torch.device, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_rope(self, seq_len: int, device, dtype):
        key = (device, dtype, seq_len)
        if key not in self._rope_cache:
            cos, sin = build_rope_cache(seq_len, self.cfg.hidden_size // self.cfg.num_attention_heads, self.cfg.rope_theta, device, dtype)
            self._rope_cache[key] = (cos, sin)
        return self._rope_cache[key]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ):
        B, T = input_ids.shape
        device = input_ids.device
        dtype = self.embed.weight.dtype

        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        cos, sin = self._get_rope(past_len + T, device, dtype)

        x = self.embed(input_ids)

        new_past = [] if use_cache else None
        aux_losses = []

        for i, layer in enumerate(self.layers):
            pkv = None if past_key_values is None else past_key_values[i]
            x, present, aux = layer(x, cos, sin, past_kv=pkv, use_cache=use_cache)
            aux_losses.append(aux)
            if use_cache:
                new_past.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        aux_loss = torch.stack(aux_losses).mean() * self.cfg.router_aux_loss_coef

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = ce + aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": aux_loss,
            "past_key_values": tuple(new_past) if use_cache else None,
        }
