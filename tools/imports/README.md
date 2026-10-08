# Import statistics (e20)

Offline corpus statistics that let the plugins pick the import path / namespace for an unresolved name and propose the imports
that usually accompany the present ones. Mined in Python, packed into a `.cml` of kind `imports`, queried by
`io.github.completionml.core.imports.ImportsModel` (docs/IMPORTS-API.md). Numbers: CHANGELOG e20.

| file | role |
|---|---|
| `imports_parse.py` | regex parsers: Go import blocks + `pkg.Ident` references; C# `using` directives, PascalCase type-like identifiers, type declarations |
| `mine_imports.py` | one task per repository over the manifest fold (status `ok`), 32 workers, file-level Counters → pickle |
| `pack_imports.py` | counts → pruned, quantised artifact (`imports_model.py` is the Python reference of the format and the two queries) |
| `eval_imports.py` | held-out evaluation (hide the import, rank it back), parity fixture for `ImportsParityTest`, example dump |
| `pack-variants.sh`, `tune.sh` | pack / evaluate several pruning variants in parallel on the rank fold |
| `go-std.txt` | `go list std` without `internal`/`vendor`: the paths without a dot in the first element that are importable |

## Commands (this server, `~/work/nn/.venv/bin/python -I`)

```
P=~/work/nn/.venv/bin/python
# 1. mine the lm fold (Go 216 s, C# 424 s on 16 workers; ≤ 500 files per repository, evenly sampled)
$P -I tools/imports/mine_imports.py --lang go     --out ~/work/ml-data/go/imports/counts-lm.pkl --jobs 16
$P -I tools/imports/mine_imports.py --lang csharp --out ~/work/ml-data/csharp/imports/counts-lm.pkl --jobs 16
# C#: the rank fold too — its declarations (dotnet/runtime, aspnetcore, efcore live there) join the name → namespace prior
# and label the evaluation; its usage statistics are NOT used
$P -I tools/imports/mine_imports.py --lang csharp --fold rank --out ~/work/ml-data/csharp/imports/counts-rank.pkl --jobs 24

# 2. pack (the published settings)
$P -I tools/imports/pack_imports.py --lang go --counts ~/work/ml-data/go/imports/counts-lm.pkl --out models/go-imports-e20.cml \
    --ctx-norm sqrt --lambda 1 --name-min-doc 5 --co-min-doc 30
$P -I tools/imports/pack_imports.py --lang csharp --counts ~/work/ml-data/csharp/imports/counts-lm.pkl \
    --decl-counts ~/work/ml-data/csharp/imports/counts-rank.pkl --out models/cs-imports-e20.cml \
    --prior both --ctx-norm sqrt --lambda 0.5 --excess-share 0.3 --max-names 40000 --decl-min-use 200

# 3. evaluate on the test fold (+ parity fixture for ml-core, + 40 true/ranked examples)
$P -I tools/imports/eval_imports.py --lang go --model models/go-imports-e20.cml --fold test --lambdas 0,1 \
    --fixture ml-core/src/test/resources/imports/go-parity.txt --dump ~/work/ml-data/go/imports/eval-test-dump.txt
$P -I tools/imports/eval_imports.py --lang csharp --model models/cs-imports-e20.cml --fold test --lambdas 0,0.5 \
    --decl-counts ~/work/ml-data/csharp/imports/counts-lm.pkl,~/work/ml-data/csharp/imports/counts-rank.pkl \
    --fixture ml-core/src/test/resources/imports/csharp-parity.txt --dump ~/work/ml-data/csharp/imports/eval-test-dump.txt

# tuning on the rank fold: tools/imports/pack-variants.sh csharp "e:--prior both ..." ; tools/imports/tune.sh csharp rank 300 "e f" --lambdas 0,0.5
```

## What is counted (file level, once per file)

- **Go**: imports from `import (...)` blocks and single `import` lines (aliases honoured, `_`/`.` imports count as imports only);
  a reference `pkg.Ident` is attributed to the import whose package name matches `pkg` (last path element, `/vN` and `.vN`
  stripped, `go-` prefix / `-go` suffix fallbacks, aliases first). Keys: `Ident` → path, and the qualifier itself (`http` → `net/http`,
  lower-case keys). The repository's own packages (`github.com/<owner>/<repo>/...`) are dropped, so are `internal`, `vendor`,
  `testdata` elements and module paths without a dot in the first element that are not in `go-std.txt`.
- **C#**: `using X.Y;` (not `static`, not aliases, `global using` included); PascalCase identifiers not followed by `(` unless after
  `new`, not declared in the file, plus `[Attr]` → `AttrAttribute`; declarations `class|struct|interface|enum|record Name` under
  the file's `namespace`. Usings that are namespaces declared anywhere in the same repository are dropped (local).
- Co-imports: every unordered pair of imports of a file.

## Model

`p(path | name)`: Go — the reference counts (exact). C# — `excess(N, U) = max(0, c(N,U) − c(N)·c(U)/D)` (co-occurrence beyond
chance; kept when ≥ 0.3·c(N) — looser thresholds add noise, see the CHANGELOG) plus declarations (`max(3, …)` file-equivalents per
declaring file in a namespace imported by ≥ 200 files). Top 16 per name, −ln p quantised to 1/16 nat in a byte.
Context: `PMI(a, b) = ln(c(a,b)·D / (c(a)·c(b)))`, top 32 partners by pair count for paths imported by ≥ 30 (Go) / 20 (C#) files,
signed byte ×16. `rankImports` score = ln p(path | name) + λ · Σ PMI(path, present) / √n (λ = 1 Go, 0.5 C#; `sum`, `mean` lost on
the rank fold); `rankCoImports` score = ln p(path) + Σ PMI over the stored lists.
