"""Export a training checkpoint to the neural `.cml` file read by ml-core (io.github.completionml.core.nn.NnFormat).

Format "CMLN" v1 (little-endian, uncompressed, 64-byte aligned tensors so that Kotlin can mmap it):
  "CMLN" | int32 version=1 | int32 metaLen | meta ("key=value\n" lines, UTF-8) | int32 nTensors |
  per tensor: int16 nameLen | name | int8 dtype (0=f32, 1=q8) | int8 rank | int32 dims[rank] | int64 dataOff | int64 scaleOff
  then the tensor payloads at their offsets.
Tensors: 2-D weights are stored as [in, out] (= PyTorch weight.T) int8 with one float32 scale per OUTPUT column
(= per PyTorch output row): W[i, o] = q[i, o] * scale[o], q = round(W / scale) clipped to [-127, 127], scale = max|W_o| / 127.
tok_emb is [dModel, vocab] (token v is column v, so the scale is per token), norms are float32 vectors.
Names: tok_emb, final_norm, layers.<i>.{attn_norm, wq, wk, wv, wo, ffn_norm, w1, w3, w2}, optional lm_head.

The byte packing is isolated in `write_cmln()`; everything else is tensor naming/quantisation.
`--check N` evaluates the fake-quantised model (quantise -> dequantise in PyTorch) on N held-out windows and reports the
loss/ppl change against the float model.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CodeLM, ModelConfig  # noqa: E402

MAGIC = b"CMLN"
VERSION = 1
DTYPE_F32, DTYPE_Q8 = 0, 1
ALIGN = 64


def align(x):
    return (x + ALIGN - 1) // ALIGN * ALIGN


def quantize_rows(w: torch.Tensor):
    """w: PyTorch [out, in] float. Returns (q int8 [in, out], scales f32 [out])."""
    w = w.detach().float().cpu()
    scales = w.abs().amax(dim=1) / 127.0
    inv = torch.where(scales > 0, 1.0 / scales, torch.zeros_like(scales))
    q = torch.round(w * inv[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.t().contiguous().numpy(), scales.numpy().astype(np.float32)


def model_tensors(model: CodeLM):
    """Ordered list of (name, kind, payload): kind 'q8' -> (q [in,out] int8, scales), 'f32' -> np.float32 vector."""
    c = model.config
    out = [("tok_emb", "q8", quantize_rows(model.tok_emb.weight))]
    for i, blk in enumerate(model.layers):
        p = f"layers.{i}."
        out.append((p + "attn_norm", "f32", blk.attn_norm.weight.detach().float().cpu().numpy()))
        for n in ("wq", "wk", "wv", "wo"):
            out.append((p + n, "q8", quantize_rows(getattr(blk.attn, n).weight)))
        out.append((p + "ffn_norm", "f32", blk.ffn_norm.weight.detach().float().cpu().numpy()))
        for n in ("w1", "w3", "w2"):
            out.append((p + n, "q8", quantize_rows(getattr(blk.mlp, n).weight)))
    out.append(("final_norm", "f32", model.final_norm.weight.detach().float().cpu().numpy()))
    if not c.tied_embeddings:
        out.append(("lm_head", "q8", quantize_rows(model.lm_head.weight)))
    return out


def config_meta(c: ModelConfig):
    """Keys exactly as NnConfig.toMeta() writes them."""
    return {"vocabSize": str(c.vocab_size), "dModel": str(c.d_model), "nLayers": str(c.n_layers), "nHeads": str(c.n_heads),
            "nKvHeads": str(c.n_kv_heads), "ffnDim": str(c.ffn_dim), "maxContext": str(c.max_context),
            "ropeTheta": repr(float(c.rope_theta)), "normEps": repr(float(c.norm_eps)),
            "tiedEmbeddings": "true" if c.tied_embeddings else "false", "activation": "swiglu", "norm": "rmsnorm"}


def write_cmln(path, meta: dict, tensors):
    """The byte-level packing (mirrors NnFormat.write)."""
    for k, v in meta.items():
        assert "=" not in k and "\n" not in k and "\n" not in str(v), (k, v)
    meta_bytes = "".join(f"{k}={v}\n" for k, v in meta.items()).encode("utf-8")
    dir_size = 4
    for name, kind, _ in tensors:
        dir_size += 2 + len(name.encode()) + 2 + 4 * (2 if kind == "q8" else 1) + 16
    pos = align(12 + len(meta_bytes) + dir_size)
    offs = []
    for name, kind, payload in tensors:
        if kind == "f32":
            offs.append((pos, 0)); pos = align(pos + 4 * payload.size)
        else:
            q, sc = payload
            d = pos; pos = align(pos + q.size)
            s = pos; pos = align(pos + 4 * sc.size)
            offs.append((d, s))
    head = bytearray()
    head += MAGIC + struct.pack("<ii", VERSION, len(meta_bytes)) + meta_bytes + struct.pack("<i", len(tensors))
    for (name, kind, payload), (d, s) in zip(tensors, offs):
        nb = name.encode("utf-8")
        head += struct.pack("<h", len(nb)) + nb
        if kind == "f32":
            head += struct.pack("<bbi", DTYPE_F32, 1, payload.size)
        else:
            q, sc = payload
            head += struct.pack("<bbii", DTYPE_Q8, 2, q.shape[0], q.shape[1])
        head += struct.pack("<qq", d, s)
    assert len(head) == 12 + len(meta_bytes) + dir_size
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        for (name, kind, payload), (d, s) in zip(tensors, offs):
            f.seek(d)
            if kind == "f32":
                f.write(np.ascontiguousarray(payload, dtype="<f4").tobytes())
            else:
                q, sc = payload
                f.write(np.ascontiguousarray(q).tobytes())
                f.seek(s)
                f.write(np.ascontiguousarray(sc, dtype="<f4").tobytes())
        f.truncate(pos)
    os.replace(tmp, path)
    return pos


def read_cmln_header(path):
    """Parse the header back (self-check). Returns (meta, [(name, dtype, dims, dataOff, scaleOff)])."""
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC
        ver, ml = struct.unpack("<ii", f.read(8)); assert ver == VERSION
        meta = dict(l.split("=", 1) for l in f.read(ml).decode().split("\n") if l)
        n, = struct.unpack("<i", f.read(4))
        ts = []
        for _ in range(n):
            nl, = struct.unpack("<h", f.read(2)); name = f.read(nl).decode()
            dt, rk = struct.unpack("<bb", f.read(2)); dims = struct.unpack("<%di" % rk, f.read(4 * rk))
            d, s = struct.unpack("<qq", f.read(16))
            ts.append((name, dt, dims, d, s))
    return meta, ts


@torch.no_grad()
def fake_quantize_(model: CodeLM):
    """Replace every 2-D weight with its int8 quantise->dequantise version (in place)."""
    for name, p in model.named_parameters():
        if p.dim() == 2:
            q, sc = quantize_rows(p)
            p.copy_(torch.from_numpy(q.T.astype(np.float32) * sc[:, None]).to(p.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt-*.pt from train.py")
    ap.add_argument("--out", required=True, help="output .cml path")
    ap.add_argument("--vocab", default=os.path.expanduser("~/work/ml-data/tokenizer/go-16384.bpe"))
    ap.add_argument("--lang", default="go")
    ap.add_argument("--check", type=int, default=0, help="eval N held-out windows float vs int8-dequantised")
    ap.add_argument("--data", default=os.path.expanduser("~/work/ml-data/go/bpe16k"))
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_dict(ck["config"])
    model = CodeLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tensors = model_tensors(model)
    meta = {"kind": "nn", "language": a.lang, "tokenizer": os.path.basename(a.vocab),
            "tokenizerSha256": hashlib.sha256(open(a.vocab, "rb").read()).hexdigest(),
            "preset": cfg.name, "paramCount": str(cfg.param_count()), "trainStep": str(ck["step"]),
            "trainTokens": str(ck["tokens"]), "run": str(ck["args"].get("run", "")),
            "docFormat": "file_sep+path;fim=psm/spm;eot", "fimRate": str(ck["args"].get("fim_rate", "")),
            "pathHeader": "false" if ck["args"].get("no_path") else "true"}
    meta.update(config_meta(cfg))
    size = write_cmln(a.out, meta, tensors)
    m2, ts = read_cmln_header(a.out)
    assert m2["dModel"] == str(cfg.d_model) and len(ts) == len(tensors)
    print(f"wrote {a.out}: {size/1e6:.1f} MB, {len(ts)} tensors, params {cfg.param_count():,}")
    for name, dt, dims, d, s in ts[:4] + ts[-2:]:
        print(f"  {name:22} {'q8 ' if dt else 'f32'} {list(dims)} @ {d}" + (f" scales @ {s}" if dt else ""))
    if a.check:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import data as D
        from train import evaluate
        tok = D.Tokenizer(a.vocab)
        sh = D.Shards(a.data, "test")
        win = D.eval_windows(sh, tok, a.check, cfg.max_context, fim=False, with_path=not ck["args"].get("no_path"))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        l0 = evaluate(model, win, 8, dev)
        fake_quantize_(model)
        l1 = evaluate(model, win, 8, dev)
        print(f"int8 check on {a.check} windows: float loss {l0:.4f} (ppl {math.exp(l0):.3f}) -> "
              f"int8 loss {l1:.4f} (ppl {math.exp(l1):.3f}), delta {l1-l0:+.4f}")


if __name__ == "__main__":
    main()
