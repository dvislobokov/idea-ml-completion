"""Our own small decoder-only code model (LLaMA-style), mirrored by the Kotlin inference in ml-core (`core.nn`).

Architecture (must stay in sync with ml-core NnConfig / NnModel):
  * pre-norm RMSNorm (x * rsqrt(mean(x^2) + eps) * w), eps 1e-5
  * RoPE, "rotate half" (GPT-NeoX / HF-LLaMA) pairing (i, i + headDim/2), theta 10000, float32 inv_freq
  * causal self-attention with GQA (nKvHeads | nHeads), no biases, no QK-norm
  * SwiGLU MLP: w2(silu(w1 x) * w3 x), no biases
  * tied input/output embeddings (lm_head = tok_emb^T), final RMSNorm
Nothing exotic: inference runs on CPU from int8 weights with one scale per output row.
"""
from dataclasses import dataclass, asdict, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_dim: int = 1408
    max_context: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True
    name: str = "custom"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def kv_dim(self):
        return self.n_kv_heads * self.head_dim

    def param_count(self):
        """Exact number of trainable parameters (tied embeddings counted once). Same formula as NnConfig.paramCount."""
        d = self.d_model
        per_layer = d * d + 2 * d * self.kv_dim + d * d + 3 * d * self.ffn_dim + 2 * d
        emb = self.vocab_size * d
        return emb + self.n_layers * per_layer + d + (0 if self.tied_embeddings else emb)

    def non_embedding_params(self):
        return self.param_count() - self.vocab_size * self.d_model * (1 if self.tied_embeddings else 2)

    def flops_per_token(self, seq_len=None):
        """Training FLOPs per token (fwd+bwd ~ 6*N, plus attention 12*L*d*T)."""
        t = seq_len or self.max_context
        return 6 * self.param_count() + 12 * self.n_layers * self.d_model * t

    def int8_bytes(self):
        """Size of the exported .cml (int8 matrices + f32 scales/norms), approximately."""
        d = self.d_model
        mats = self.param_count() - (2 * self.n_layers + 1) * d
        scales = self.vocab_size + self.n_layers * (2 * d + 2 * self.kv_dim + 2 * self.ffn_dim + d)
        return mats + 4 * scales + 4 * (2 * self.n_layers + 1) * d

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


# Named presets. head_dim is 64 everywhere (good for CPU kernels and SDPA); GQA with 2-4 kv heads keeps the KV cache
# small on the CPU side. Parameter counts (tied embeddings, vocab 16384) are printed by `python model.py`.
PRESETS = {
    "go5m":   dict(d_model=256, n_layers=6,  n_heads=4,  n_kv_heads=2, ffn_dim=704),    # ~8 M incl. 4.2 M embeddings: proxy for data/format hypotheses
    "go19m":  dict(d_model=384, n_layers=8,  n_heads=6,  n_kv_heads=2, ffn_dim=1024),   # 18.9 M
    "go31m":  dict(d_model=512, n_layers=8,  n_heads=8,  n_kv_heads=2, ffn_dim=1408),   # 30.9 M
    "go50m":  dict(d_model=640, n_layers=10, n_heads=10, n_kv_heads=2, ffn_dim=1536),   # 49.8 M
    "go102m": dict(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, ffn_dim=2560),   # 102.3 M
}


def preset(name, **overrides):
    kw = dict(PRESETS[name]); kw.update(overrides)
    return ModelConfig(name=name, **kw)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).type_as(x)


def rope_cos_sin(head_dim, max_len, theta, device):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    ang = torch.outer(pos, inv_freq)                     # [T, hd/2]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, hd]; cos/sin: [T, hd/2]. Rotate-half convention: pairs (i, i + hd/2)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[None, None, :, :].to(x.dtype)
    s = sin[None, None, :, :].to(x.dtype)
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.wq = nn.Linear(c.d_model, c.d_model, bias=False)
        self.wk = nn.Linear(c.d_model, c.kv_dim, bias=False)
        self.wv = nn.Linear(c.d_model, c.kv_dim, bias=False)
        self.wo = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_kv != self.n_heads)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(c.d_model, c.ffn_dim, bias=False)   # gate
        self.w3 = nn.Linear(c.d_model, c.ffn_dim, bias=False)   # up
        self.w2 = nn.Linear(c.ffn_dim, c.d_model, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.attn = Attention(c)
        self.ffn_norm = RMSNorm(c.d_model, c.norm_eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class CodeLM(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final_norm = RMSNorm(c.d_model, c.norm_eps)
        if c.tied_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        cos, sin = rope_cos_sin(c.head_dim, c.max_context, c.rope_theta, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # GPT-2 style: scale residual-output projections by 1/sqrt(2*n_layers)
        for blk in self.layers:
            for w in (blk.attn.wo.weight, blk.mlp.w2.weight):
                nn.init.normal_(w, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def logits(self, h):
        if self.lm_head is None:
            return F.linear(h, self.tok_emb.weight)
        return self.lm_head(h)

    def forward(self, idx, targets=None):
        """idx: [B, T] int64. Returns (logits [B, T, V], loss or None). Loss = mean CE over all positions."""
        B, T = idx.shape
        assert T <= self.config.max_context
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for blk in self.layers:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        logits = self.logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_ids=(), temperature=0.0):
        """Greedy (temperature 0) or sampled continuation of idx [1, T]. Simple full-recompute loop (no KV cache)."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.max_context:]
            logits, _ = self(ctx)
            lg = logits[:, -1, :].float()
            if temperature > 0:
                probs = F.softmax(lg / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = lg.argmax(-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
            if int(nxt) in stop_ids:
                break
        return idx

    def param_groups(self, weight_decay):
        decay = [p for n, p in self.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.dim() < 2]
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


if __name__ == "__main__":
    print(f"{'preset':8} {'d':>5} {'L':>3} {'H':>3} {'KV':>3} {'ffn':>5} {'params':>12} {'non-emb':>12} {'int8 MB':>8} {'GFLOP/tok':>9}")
    for name in PRESETS:
        c = preset(name)
        print(f"{name:8} {c.d_model:5} {c.n_layers:3} {c.n_heads:3} {c.n_kv_heads:3} {c.ffn_dim:5} "
              f"{c.param_count():12,} {c.non_embedding_params():12,} {c.int8_bytes()/1e6:8.1f} {c.flops_per_token()/1e9:9.3f}")
    m = CodeLM(preset("go31m"))
    n = sum(p.numel() for p in m.parameters())
    assert n == m.config.param_count(), (n, m.config.param_count())
    print("go31m: live parameter count matches formula:", n)
