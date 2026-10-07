#!/usr/bin/env python
"""Charts for docs/talk/slides.md. Every number is copied from CHANGELOG.md / docs / eval files of this repository
(the source is named in a comment next to each dataset). Run: /root/work/nn/.venv/bin/python -I make_charts.py"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
GO, CS, AQUA, YEL, VIO, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#9a9892"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "svg.fonttype": "none", "figure.facecolor": "white"})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), format="svg")
    if os.environ.get("CHART_PNG"): fig.savefig(os.path.join(os.environ["CHART_PNG"], name[:-4] + ".png"), dpi=80)
    plt.close(fig)

def bars(ax, labels, series, colors, names, fmt="{:.1f}", width=None, ylabel=None, ylim=None):
    n = len(series); x = np.arange(len(labels)); w = width or 0.8 / n
    for i, (vals, c, nm) in enumerate(zip(series, colors, names)):
        xs = x + (i - (n - 1) / 2) * w
        b = ax.bar(xs, vals, w * 0.92, color=c, label=nm, linewidth=0)
        for xi, v in zip(xs, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)): continue
            ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    if ylabel: ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    if n > 1: ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)

# ---------- 1. corpus: licences (computed from ~/work/ml-data/catalog/{go,csharp}-20.jsonl, all catalogue rows) ----------
lic_go = {"MIT": 13892, "Apache-2.0": 10750, "none": 4960, "NOASSERTION": 2343, "GPL-3.0": 1923, "BSD-3-Clause": 1094,
          "AGPL-3.0": 1020, "MPL-2.0": 770, "other": 38085 - (13892+10750+4960+2343+1923+1094+1020+770)}
lic_cs = {"MIT": 18755, "none": 10887, "NOASSERTION": 3490, "Apache-2.0": 3213, "GPL-3.0": 3163, "AGPL-3.0": 560,
          "GPL-2.0": 424, "BSD-3-Clause": 420, "other": 42579 - (18755+10887+3490+3213+3163+560+424+420)}
keys = ["MIT", "Apache-2.0", "GPL-3.0", "AGPL-3.0", "BSD-3-Clause", "MPL-2.0", "GPL-2.0", "NOASSERTION", "none", "other"]
fig, ax = plt.subplots(figsize=(10, 4.6))
g = [100 * lic_go.get(k, 0) / 38085 for k in keys]; c = [100 * lic_cs.get(k, 0) / 42579 for k in keys]
bars(ax, keys, [g, c], [GO, CS], ["Go (38 085 репо)", "C# (42 579 репо)"], ylabel="% репозиториев каталога")
ax.set_title("Лицензии в каталоге GitHub ≥20★ (поле license API)", loc="left", color=INK)
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
save(fig, "corpus_licences.svg")

# ---------- 2. prepare funnel (stats.json of `ml-train prepare`, 2026-10-07 rebuild) ----------
go_st = {"kept": 4694559, "dup_exact": 1032298, "dup_near": 552856, "generated_marker": 763043, "generated_name": 69756,
         "too_large": 2422, "long_lines": 23381}
cs_st = {"kept": 6058957, "dup_exact": 1556835, "dup_near": 1269715, "generated_marker": 543748, "generated_name": 769899,
         "too_large": 2584, "long_lines": 26744}
cats = ["kept", "dup_exact", "dup_near", "generated_marker", "generated_name", "long_lines/too_large"]
def row(st): return [st["kept"], st["dup_exact"], st["dup_near"], st["generated_marker"], st["generated_name"], st["long_lines"] + st["too_large"]]
fig, ax = plt.subplots(figsize=(10, 4.6))
bars(ax, cats, [[v / 1e6 for v in row(go_st)], [v / 1e6 for v in row(cs_st)]], [GO, CS], ["Go", "C#"], fmt="{:.2f}", ylabel="млн файлов")
ax.set_title("ml-train prepare: что осталось и что выброшено (без excluded dirs: Go 7.26 млн, C# 0.95 млн)", loc="left", color=INK, fontsize=11)
save(fig, "corpus_funnel.svg")

# ---------- 3. n-gram perplexity by experiment (CHANGELOG e0x–e18; test sets differ between corpus scales) ----------
exps = ["base\n(13 repo)", "e04\ncache", "e05\nrepo split", "e06\nMKN-5", "e11/e12\n580/472 repo", "e14/e15\nfull corpus", "e18\nrebuilt"]
ppl_go = [8.3, 5.0, 5.5, 5.1, 4.7, 4.1, 4.6]; ppl_cs = [9.9, 5.8, 6.9, 6.5, 5.9, 5.9, 5.3]
fig, ax = plt.subplots(figsize=(10, 4.6))
bars(ax, exps, [ppl_go, ppl_cs], [GO, CS], ["Go", "C#"], ylabel="perplexity (ниже — лучше)")
ax.set_title("N-граммная LM: perplexity по экспериментам (cache λ=0.3 с e04; тестовые наборы растут с корпусом)", loc="left", color=INK, fontsize=11)
save(fig, "ngram_ppl.svg")

# ---------- 4. n-gram model size (CHANGELOG e02, e03, e11, e14) ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
labels = ["v1 float\n(13 repo)", "v2 24-bit fp\n+8-bit", "+prune\n1,1,2,2", "580 repo\nno prune", "580 repo\nmin-repos", "22 610 repo\ne14-a", "22 610 repo\ne14-b"]
sizes = [45.1, 18.6, 7.7, 163, 20, 207, 32]
bars(ax, labels, [sizes], [GO], ["Go LM, MB"], fmt="{:.0f}", ylabel="размер файла, MB")
ax.set_title("Размер n-граммной модели: формат и pruning держат 32 MB на полном корпусе", loc="left", color=INK)
save(fig, "ngram_size.svg")

# ---------- 5. ranker e17 on real Go lists (CHANGELOG e17) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["правила плагина", "только n-gram LM", "самый частый\nв файле", "proxy-ranker\n(синтетика)", "e17 ranker\n(реальные списки)"]
mrr = [0.527, 0.499, 0.515, 0.518, 0.808]; top1 = [0.388, 0.396, 0.357, 0.390, 0.710]; top5 = [0.685, 0.608, 0.710, 0.666, 0.934]
bars(ax, labels, [mrr, top1, top5], [GO, AQUA, YEL], ["MRR", "top-1", "top-5"], fmt="{:.3f}", ylim=(0, 1.05))
ax.set_title("Go, 83 904 реальных списка из 295 held-out репо: ranker 0.808 MRR против правил 0.527", loc="left", color=INK, fontsize=11)
save(fig, "ranker_e17.svg")

# ---------- 6. per context kind (CHANGELOG e17) ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
kinds = ["после `.`", "начало\nоператора", "аргумент", "позиция\nтипа", "правая часть\nприсваивания", "прочее"]
bars(ax, kinds, [[0.776, 0.832, 0.837, 0.825, 0.780, 0.786], [0.342, 0.642, 0.653, 0.420, 0.490, 0.500]], [GO, GREY], ["e17 ranker", "правила плагина"], fmt="{:.3f}", ylim=(0, 1.0), ylabel="MRR")
ax.set_title("По видам контекста: правила хуже всего после `.` и в позиции типа", loc="left", color=INK)
save(fig, "ranker_kinds.svg")

# ---------- 7. heaviest standardised weights (CHANGELOG e17) ----------
fig, ax = plt.subplots(figsize=(10, 4.2))
names = ["prefix_case_match", "scope_level", "declared in file", "in_vocab", "needs import", "kind_keyword", "after_dot:lm_global_logprob"]
w = [1.51, -0.95, 0.86, 0.73, -0.73, -0.69, 0.67]
cols = [GO if v > 0 else RED for v in w]
ax.barh(names[::-1], w[::-1], color=cols[::-1], height=0.6)
for i, v in enumerate(w[::-1]): ax.text(v + (0.03 if v > 0 else -0.03), i, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", color=INK)
ax.axvline(0, color=INK2, linewidth=0.8); ax.set_xlim(-1.3, 1.9); ax.grid(axis="y", visible=False)
ax.set_title("Самые тяжёлые стандартизованные веса e17 (210 весов = 30 признаков × (1 + 6 видов контекста))", loc="left", color=INK, fontsize=11)
save(fig, "ranker_weights.svg")

# ---------- 8. presets (python tools/nn/train/model.py) ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
pre = ["go5m\nd256×6", "go19m\nd384×8", "go31m\nd512×8", "go50m\nd640×10", "go102m\nd768×12"]
params = [8.62, 18.88, 30.94, 49.82, 102.26]; gflop = [0.089, 0.189, 0.286, 0.456, 0.840]
bars(ax, pre, [params], [GO], ["параметров, M (int8 файл ≈ столько же MB)"], fmt="{:.1f}", ylabel="M параметров")
for i, g in enumerate(gflop): ax.text(i, params[i] + 7, f"{g:.3f} GFLOP/tok", ha="center", fontsize=9, color=INK2)
ax.set_ylim(0, 125)
ax.set_title("Пресеты model.py: vocab 16 384, head_dim 64, 2 KV-головы; обучающие FLOP/токен = 6N + attention", loc="left", color=INK, fontsize=11)
save(fig, "presets.svg")

# ---------- 9. latency on the real model, 8 threads EPYC (docs/NN-PARITY.md) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["prefill 1500 tok", "строка 20 tok\n(холодная)", "строка при наборе\n(KV reuse, +8 tok)"]
bars(ax, labels, [[492, 530, 47], [211, 240, 33], [147, 167, 24]], [GREY, AQUA, GO], ["scalar Kotlin", "native f32 (AVX-512)", "native q8 (VNNI)"], fmt="{:.0f}", ylabel="мс")
ax.set_title("go31m, 8 потоков, EPYC 9554: нативные q8-ядра ×3.2 на prefill, ×2 при наборе", loc="left", color=INK)
save(fig, "latency_server.svg")

# ---------- 10. Mac M1 Pro (docs/MAC-CHECK-RU.md, 4 threads, cs31m, 512-token prompt) ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, ["строка целиком\n(512 tok + 20)", "строка при наборе\n(+8 tok)"], [[252, 40], [164, 49], [68, 17]], [GREY, AQUA, GO], ["scalar Kotlin", "native f32 (NEON)", "native q8 (sdot)"], fmt="{:.0f}", ylabel="мс")
ax.set_title("MacBook Pro M1 Pro, 4 потока, cs31m: native q8 68 мс против 252 скалярно (×3.7)", loc="left", color=INK, fontsize=11)
save(fig, "latency_mac.svg")

# ---------- 11. loss curves from metrics.jsonl ----------
def load_metrics(path):
    tr, ev = [], []
    for line in open(path):
        d = json.loads(line)
        if "eval_ppl" in d: ev.append((d["tokens"], d["eval_ppl"], d.get("eval_fim_ppl")))
        elif "loss" in d: tr.append((d["tokens"], d["loss"]))
    return np.array(tr), np.array(ev, dtype=float)

def smooth(y, k=25):
    if len(y) < k: return y
    return np.convolve(y, np.ones(k) / k, mode="valid")

runs = {"go31m-e2": ("/root/work/ml-data/go/nn/go31m-e2/metrics.jsonl", GO, "-"),
        "go50m-e3-lr2e3": ("/root/work/ml-data/go/nn/go50m-e3-lr2e3/metrics.jsonl", GO, "--"),
        "cs31m-e2-lr2e3": ("/root/work/ml-data/csharp/nn/cs31m-e2-lr2e3/metrics.jsonl", CS, "-"),
        "cs50m-e3-lr2e3": ("/root/work/ml-data/csharp/nn/cs50m-e3-lr2e3/metrics.jsonl", CS, "--"),
        "cs31m-e2-spm10 (lr 1e-3, 1 M/step)": ("/root/work/ml-data/csharp/nn/cs31m-e2-spm10/metrics.jsonl", YEL, "-"),
        "cs31m-e4-2ep": ("/root/work/ml-data/csharp/nn/cs31m-e4-2ep/metrics.jsonl", VIO, ":")}
data = {}
for name, (p, c, ls) in runs.items():
    if os.path.exists(p): data[name] = (load_metrics(p), c, ls)

fig, ax = plt.subplots(figsize=(10, 4.8))
for name, ((tr, ev), c, ls) in data.items():
    if "2ep" in name: continue
    y = smooth(tr[:, 1]); x = tr[len(tr) - len(y):, 0] / 1e9
    ax.plot(x, y, color=c, linestyle=ls, linewidth=2, label=name)
ax.set_xlabel("токенов, млрд"); ax.set_ylabel("train loss (скользящее среднее 25 шагов)"); ax.set_ylim(0.9, 2.2); ax.legend(frameon=False)
ax.set_title("Кривые обучения (metrics.jsonl): Go учится легче C#; 50 M ниже 31 M", loc="left", color=INK)
save(fig, "loss_curves.svg")

fig, ax = plt.subplots(figsize=(10, 4.8))
for name, ((tr, ev), c, ls) in data.items():
    if len(ev) == 0: continue
    ax.plot(ev[:, 0] / 1e9, ev[:, 1], color=c, linestyle=ls, linewidth=2, label=name, marker="o", markersize=3)
    ax.text(ev[-1, 0] / 1e9, ev[-1, 1], f" {ev[-1,1]:.2f}", va="center", fontsize=9, color=INK)
ax.set_xlabel("токенов, млрд"); ax.set_ylabel("eval perplexity (plain, test-фолд)"); ax.set_ylim(2.5, 6); ax.legend(frameon=False)
ax.set_title("Eval ppl по ходу обучения: cosine-хвост даёт последние −10 %; 2-я эпоха C# −4 % ppl, но не строк", loc="left", color=INK, fontsize=11)
save(fig, "eval_curves.svg")

# ---------- 12. throughput (CHANGELOG e16, e18) ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
bars(ax, ["RTX PRO 6000\n(1 GPU, e16)", "H200\n(1 GPU)", "2 × H200\nDDP torchrun"], [[0.771, 1.25, 2.45]], [GO], ["M токенов/с, go31m"], fmt="{:.2f}", ylabel="M токенов / с")
ax.text(2, 2.5, "×1.96", ha="center", color=INK2)
ax.set_title("Скорость обучения 31 M: эпоха 6.8 G токенов — 2.5 ч → 77 мин → 47 мин", loc="left", color=INK)
save(fig, "throughput.svg")

# ---------- 13. C# recipe ablations (CHANGELOG e16/e18; rest of line exact, all positions, healed) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["cs31m-e1\nlr 1e-3, 1 M", "e2-spm10\nSPM 100 %", "e2-lr2e3\nlr 2e-3, 0.5 M", "e4-2ep\n2 эпохи", "e5-line\nline-FIM 0.9", "cs50m-e3\nd640×10"]
vals = [47.9, 47.7, 50.2, 50.0, 48.8, 51.6]; ppl = [3.97, 3.98, 3.80, 3.65, 3.83, 3.60]
cols = [GREY, GREY, CS, GREY, GREY, CS]
b = ax.bar(labels, vals, color=cols, width=0.6)
for i, (v, p) in enumerate(zip(vals, ppl)):
    ax.text(i, v + 0.3, f"{v:.1f} %", ha="center", color=INK); ax.text(i, 30.5, f"ppl {p:.2f}", ha="center", fontsize=9, color="white")
ax.set_ylim(30, 55); ax.set_ylabel("остаток строки верен, % (3 000 позиций)"); ax.grid(axis="x", visible=False)
ax.set_title("C#: единственное, что сдвинуло метрику — lr 2e-3 / батч 0.5 M (+2.3 п.п.) и ёмкость 50 M (+1.4)", loc="left", color=INK, fontsize=11)
save(fig, "cs_ablations.svg")

# ---------- 14. Go vs C#, 31 vs 50 (CHANGELOG e18) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["остаток строки\nверен, все позиции", "≤ 8 токенов\nосталось", "fresh-репо\n(≥ 2026-05)"]
bars(ax, labels, [[63.6, 74.8, 56.6], [65.7, 76.3, 59.2], [50.2, 62.6, 40.9], [51.6, 63.5, 43.5]], [GO, "#8fb8ea", CS, "#f3a98a"],
     ["go31m-e2", "go50m-e3", "cs31m-e2-lr2e3", "cs50m-e3"], ylabel="%", ylim=(0, 90))
ax.set_title("31 M → 50 M: +2.1 / +2.6 п.п. Go, +1.4 / +2.6 C# за 1.7× compute и 1.7× латентности", loc="left", color=INK, fontsize=11)
save(fig, "size_31_50.svg")

# ---------- 15. threshold curves (eval-go31m-e2.md / eval-cs31m-e2-lr2e3.md, spm, policy 'both') ----------
thr = [0.5, 0.6, 0.7, 0.8, 0.9]
go_shown = [36.1, 30.9, 25.8, 20.4, 13.5]; go_prec = [89.0, 91.0, 93.5, 95.3, 96.5]
cs_shown = [23.3, 18.6, 14.2, 10.0, 4.9]; cs_prec = [87.6, 90.7, 94.4, 97.3, 98.6]
go_all = [57.9, 50.5, 43.8, 36.1, 26.7]; go_all_p = [92.2, 93.7, 95.7, 97.0, 98.1]
fig, ax = plt.subplots(figsize=(10, 4.8))
for sh, pr, c, nm, ls in [(go_shown, go_prec, GO, "go31m-e2, без чистой пунктуации", "-"), (go_all, go_all_p, GO, "go31m-e2, закрывающие скобки тоже", "--"), (cs_shown, cs_prec, CS, "cs31m-e2-lr2e3, без чистой пунктуации", "-")]:
    ax.plot(sh, pr, color=c, linestyle=ls, marker="o", linewidth=2, markersize=7, label=nm)
    for t, s, p in zip(thr, sh, pr): ax.text(s + 0.5, p - 0.9, f"≥{t}", fontsize=9, color=INK2)
ax.set_xlabel("показано, % позиций"); ax.set_ylabel("целая строка верна среди показанных, %"); ax.legend(frameon=False, loc="lower left")
ax.set_title("Порог по conf_prod: гейт 0.7 = Go 25.8 % показов при 93.5 %, C# 14.2 % при 94.4 %", loc="left", color=INK)
save(fig, "threshold_curves.svg")

# ---------- 16. confidence definitions (eval-go31m-e2.md spm) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
defs = {"conf_gm3 (ср. 3 первых)": ([95.7, 92.3, 86.2, 75.7, 62.9], [66.2, 68.3, 71.3, 74.4, 77.7], GREY),
        "conf_min (минимум)": ([71.2, 59.5, 50.9, 42.0, 30.9], [82.6, 88.2, 91.8, 95.1, 97.2], YEL),
        "conf_prod (произведение)": ([57.9, 50.5, 43.8, 36.1, 26.7], [92.1, 93.6, 95.7, 97.0, 98.1], GO)}
for nm, (sh, pr, c) in defs.items():
    ax.plot(sh, pr, color=c, marker="o", linewidth=2, markersize=7, label=nm)
ax.set_xlabel("показано, % позиций"); ax.set_ylabel("целая строка верна, %"); ax.legend(frameon=False, loc="lower left")
ax.set_title("Go, SPM: при равной доле показов произведение вероятностей точнее среднего по 3 токенам", loc="left", color=INK, fontsize=11)
save(fig, "confidence_defs.svg")

# ---------- 17. token healing (NEURAL-RU §7c, go31m-e1 / cs31m-e1) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["все позиции\nGo", "все позиции\nC#", "курсор внутри\nпунктуации, Go (n=291)", "курсор внутри\nпунктуации, C# (n=327)"]
bars(ax, labels, [[55.7, 39.8, 4.8, 1.5], [61.7, 47.9, 67.0, 75.2]], [GREY, AQUA], ["без healing", "с token healing"], ylabel="остаток строки верен, %")
ax.set_title("Token healing: +6 / +8 п.п. целиком — весь выигрыш в 10 % позиций внутри пре-токена", loc="left", color=INK)
save(fig, "healing.svg")

# ---------- 18. teachers on fresh sets (CHANGELOG e18) ----------
fig, ax = plt.subplots(figsize=(10, 4.8))
labels = ["наша 31 M", "наша 50 M", "Qwen2.5-Coder\n1.5B", "Qwen2.5-Coder\n7B", "Qwen2.5-Coder\n14B", "Qwen2.5-Coder\n32B", "Qwen3.6-35B-A3B\n(не умеет FIM)"]
cs_v = [40.9, 43.5, 56.1, 64.1, 66.7, 66.3, 11.8]; go_v = [56.6, 59.2, 63.9, 70.6, np.nan, np.nan, np.nan]
bars(ax, labels, [go_v, cs_v], [GO, CS], ["Go, 2 000 fresh-позиций", "C#, 2 000 fresh-позиций"], ylabel="остаток строки верен, %", ylim=(0, 80))
ax.set_title("Потолок учителя на репозиториях, созданных после 2026-05-01: 7B — выбор, 14B/32B насыщаются", loc="left", color=INK, fontsize=11)
save(fig, "teachers.svg")

# ---------- 19. paired analysis (CHANGELOG e18; Go 'both' = 2000*0.566 − 78) ----------
fig, ax = plt.subplots(figsize=(10, 4.2))
labels = ["C#: 7B vs cs31m", "C#: 1.5B vs cs31m", "C#: 14B vs cs31m", "Go: 7B vs go31m", "Go: 1.5B vs go31m"]
both = [760, 2000 - 381 - 78 - (2000 - 1122 - 381), 2000 - 570 - 55 - (2000 - 1334 - 570), 1054, 1132 - 103]
only_t = [523, 381, 570, 357, 248]; only_o = [59, 78, 55, 78, 103]
# C# 'both' for 1.5B/14B is not published; keep only the pairs with full numbers
labels, both, only_t, only_o = [labels[0], labels[3]], [760, 1054], [523, 357], [59, 78]
y = np.arange(len(labels))
ax.barh(y, both, color=AQUA, label="обе верны"); ax.barh(y, only_t, left=both, color=VIO, label="только учитель 7B")
ax.barh(y, only_o, left=np.array(both) + np.array(only_t), color=CS, label="только наша")
for yi, (b, t, o) in enumerate(zip(both, only_t, only_o)):
    ax.text(b / 2, yi, str(b), ha="center", va="center", color="white"); ax.text(b + t / 2, yi, str(t), ha="center", va="center", color="white"); ax.text(b + t + o + 10, yi, str(o), va="center", color=INK)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlabel("позиций из 2 000 (fresh-набор)"); ax.set_xlim(0, 2000); ax.legend(frameon=False, loc="lower right", ncol=3); ax.grid(axis="y", visible=False)
ax.set_title("Парный анализ: учитель почти не ошибается там, где правы мы; его плюс — «знание», не синтаксис", loc="left", color=INK, fontsize=11)
save(fig, "paired.svg")

# ---------- 20. gap by rest length on fresh sets (CHANGELOG e18) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["1–3 токена", "4–8 токенов", "9+ токенов"]
bars(ax, labels, [[83.9, 51.7, 24.2], [87.4, 72.2, 45.4], [72.7, 34.5, 14.2], [85.7, 62.5, 42.8]], [GO, "#8fb8ea", CS, "#f3a98a"],
     ["go31m-e2", "Qwen2.5-Coder-7B на Go", "cs31m-e2-lr2e3", "Qwen2.5-Coder-7B на C#"], ylabel="остаток строки верен, %", ylim=(0, 100))
ax.set_title("Разрыв с учителем растёт с длиной остатка: 3.5 п.п. на коротких, 21–29 на длинных", loc="left", color=INK)
save(fig, "restlen_gap.svg")

# ---------- 21. contamination by repo creation date (CHANGELOG e18, 500 std C# positions) ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, ["репо созданы до 2024\n(344 позиции)", "созданы после 2025-07\n(112 позиций)"], [[50.3, 49.1], [65.7, 56.2], [71.2, 59.8]], [CS, "#b9a6e0", VIO], ["cs31m-e2-lr2e3", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-7B"], ylabel="остаток строки верен, %", ylim=(0, 85))
ax.set_title("Контаминация: у учителей −9…−11 п.п. на новых репо, у нашей модели −1 (её корпус = тот же тест-фолд)", loc="left", color=INK, fontsize=10.5)
save(fig, "contamination.svg")

# ---------- 22. PSI headroom (CHANGELOG e17; tools/roslyn/REPORT.md §3) ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
y = [0, 1]; pre = [80.7, 77.3]; psi = [4.8, 2.0]; none = [14.5, 20.8]
ax.barh(y, pre, color=GO, label="уже есть в префиксе файла"); ax.barh(y, psi, left=pre, color=AQUA, label="только из PSI / Roslyn-контекста")
ax.barh(y, none, left=np.array(pre) + np.array(psi), color=GREY, label="нигде")
for yi, (a, b, c) in enumerate(zip(pre, psi, none)):
    ax.text(a / 2, yi, f"{a} %", ha="center", va="center", color="white"); ax.text(a + b / 2, yi, f"{b}", ha="center", va="center", color="white", fontsize=9); ax.text(a + b + c / 2, yi, f"{c} %", ha="center", va="center", color="white")
ax.set_yticks(y); ax.set_yticklabels(["Go (2 436 позиций)", "C# (3 000 позиций)"]); ax.set_xlim(0, 100); ax.set_xlabel("идентификаторы в истинном остатке строки, %"); ax.legend(frameon=False, loc="lower right", fontsize=9); ax.grid(axis="y", visible=False)
ax.set_title("Структурный контекст — рычаг скорости и точности после `.`, а не recall", loc="left", color=INK)
save(fig, "psi_headroom.svg")

# ---------- 23. n-gram vs nn headline, Go (README / CHANGELOG e14, e18) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = ["показ при ≈95 %\nточности строки", "остаток строки верен,\n≤ 8 токенов", "первый токен\nверен"]
bars(ax, labels, [[10.0, 35.1, 65.2], [36.1, 74.8, 88.1]], [GREY, GO], ["n-gram e14-b (32 MB)", "go31m-e2 (31 MB int8)"], ylabel="%", ylim=(0, 100))
ax.set_title("Go: тот же размер файла, в 3.6 раза больше показов и вдвое больше верных строк", loc="left", color=INK)
save(fig, "ngram_vs_nn_go.svg")

fig, ax = plt.subplots(figsize=(10, 4.6))
bars(ax, labels, [[3.1, 31.5, 57.0], [20.0, 62.6, 82.1]], [GREY, CS], ["n-gram e15-a (32 MB)", "cs31m-e2-lr2e3 (31 MB int8)"], ylabel="%", ylim=(0, 100))
ax.set_title("C#: n-граммы почти не показывают (3 %), трансформер — 20 % при 97 %", loc="left", color=INK)
save(fig, "ngram_vs_nn_cs.svg")

# ---------- 24. PSM vs SPM (NEURAL-RU §7b) ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, ["NLL первого токена\nсередины, C#", "NLL первого токена\nсередины, Go", "строка целиком,\nC# (%)"], [[2.12, 3.16, 20.7], [0.81, 1.02, 39.8]], [GREY, AQUA], ["PSM", "SPM"], fmt="{:.2f}")
ax.set_title("FIM при 31 M: в PSM модель «не возвращается» к префиксу через суффикс — инференс в SPM", loc="left", color=INK, fontsize=11)
save(fig, "psm_spm.svg")

# ---------- 25. prefix length study (CHANGELOG e18, cs31m-e2-lr2e3) ----------
fig, ax = plt.subplots(figsize=(9, 4.2))
bars(ax, ["префикс ≤ 1450", "префикс ≤ 1024", "префикс ≤ 512"], [[50.2, 50.2, 49.4], [20.0, 21.7, 22.2]], [CS, YEL], ["остаток строки верен, %", "показано при prod ≥ 0.8, %"], ylim=(0, 60))
ax.set_title("Бюджет промпта: 1024 токена префикса бесплатно (−25 % prefill), 512 стоит 0.8 п.п.", loc="left", color=INK, fontsize=11)
save(fig, "prefix_study.svg")

# ---------- 26. Go export with module cache (tools/psi/REPORT.md) ----------
fig, ax = plt.subplots(figsize=(10, 4.4))
labels = ["gatewayd\n(43 deps)", "imposter-cli\n(23)", "thunderdome\n(36)", "drako\n(8)"]
bars(ax, labels, [[79.1, 77.6, 82.7, 91.3], [92.1, 97.4, 94.7, 95.0]], [GREY, GO], ["без go.mod / модульного кэша (e17)", "с go.mod overlay + GOMODCACHE"], ylabel="receiver после `.` разрешён, %", ylim=(0, 110))
ax.set_title("Экспорт реальных списков: корпус был без go.mod — модульный кэш даёт +20–31 % списков, recall 0.888 → 0.920", loc="left", color=INK, fontsize=10.5)
save(fig, "export_modcache.svg")

# ---------- 27. line exact by position kind, SPM (eval-go31m-e2.md, eval-cs31m-e2-lr2e3.md) ----------
fig, ax = plt.subplots(figsize=(10, 4.6))
kinds = ["после\nидентификатора", "начало\nстроки", "после\nпрочего", "после `.`", "после `(` `[` `{`", "после `,`", "после\nключевого слова", "после `=`"]
bars(ax, kinds, [[63.0, 70.1, 68.6, 59.8, 58.1, 50.6, 57.0, 57.1], [52.8, 45.0, 63.9, 42.6, 50.2, 41.4, 43.1, 35.6]], [GO, CS], ["go31m-e2", "cs31m-e2-lr2e3"], fmt="{:.0f}", ylabel="остаток строки верен, %", ylim=(0, 85))
ax.set_title("По видам позиций: труднее всего после `,`, `=` и `.` — там нужны факты из других файлов", loc="left", color=INK, fontsize=11)
save(fig, "kinds_nn.svg")

# ---------- 28. roslyn filter headroom (tools/roslyn/REPORT.md §2) ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
thr2 = ["≥0.5", "≥0.6", "≥0.7", "≥0.8", "≥0.9"]
bars(ax, thr2, [[129, 85, 46, 18, 4], [9, 4, 1, 0, 0]], [GREY, VIO], ["неверных строк показано", "из них снял Roslyn-фильтр"], fmt="{:.0f}", ylabel="позиций из 3 000")
ax.set_title("Roslyn-фильтр «строка разрешается?» снимает 1 из 46 ошибок при гейте 0.7: ошибки — валидный, но чужой код", loc="left", color=INK, fontsize=10.5)
save(fig, "roslyn_filter.svg")

# ---------- 29. C# ranker: proxy vs real preliminary (CHANGELOG e15, CLAUDE.md e18 state) ----------
fig, ax = plt.subplots(figsize=(9, 4.4))
bars(ax, ["правила плагина", "cs-rank-e18-pre\n(83 репо, 14 k списков)"], [[0.567, 0.743], [0.409, 0.625]], [CS, YEL], ["MRR", "top-1"], fmt="{:.3f}", ylim=(0, 1))
ax.set_title("C#, предварительно (экспорт ещё идёт): реальные списки → MRR 0.743 против правил 0.567", loc="left", color=INK, fontsize=11)
save(fig, "ranker_cs_pre.svg")

print("charts:", len(os.listdir(OUT)))
