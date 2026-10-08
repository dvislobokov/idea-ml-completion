# Import statistics — plugin-facing API (`ml-core`, package `io.github.completionml.core.imports`)

What a plugin calls to rank the import path / namespace for an unresolved name (auto-import popup, "import class" quick fix
ordering) and to propose imports that usually accompany the present ones (suggested imports). Pure Kotlin, heap-resident (int arrays and two
hash maps, ~120 ms load), thread-safe, no PSI dependency: the plugin passes strings. Mined from the corpora by
`tools/imports/` (CHANGELOG e20 has the numbers: Go top-1 0.717 vs 0.685 for the frequency prior, C# 0.814 vs 0.777).

## Loading

```kotlin
val imports = ImportsModel.read(File(modelsDir, "go-imports-e20.cml"))        // or read(stream) for a bundled resource
```

Artifacts: `models/go-imports-e20.cml` (2.9 MB) and `models/cs-imports-e20.cml` (2.6 MB) — a `.cml` of kind `imports`
(`ModelFormat` gzip container; `ImportsModel.read` refuses other kinds). Load once per language, keep it for the IDE session.

## `rankImports(name, currentImports): List<Suggestion>`

```kotlin
fun rankImports(name: String, currentImports: Collection<String>, lambda: Float = this.lambda): List<Suggestion>
class Suggestion(val path: String, val score: Float)   // path = import path (Go) / namespace (C#); score in nats, best first
```

- `name` — Go: the exported identifier (`Client`, `Context`, `Logger`) **or** an unresolved package qualifier (`http`, `yaml`,
  `log` — lower-case keys); C#: the type name (`Task`, `List`, `JsonSerializer`, `ILogger`; attributes as `FooAttribute`).
- `currentImports` — the import paths / namespaces already present in the file (Go: without aliases; C#: plain `using X.Y;`
  namespaces, implicit/global usings included if known). Unknown ones are ignored.
- Result: every path the corpus has seen supplying the name (≤ 16), best first; `score = ln p(path | name) + λ·Σ PMI(path, import)/√n`.
  Scores are comparable within one call; `exp(score)` is not a probability. Empty list = never seen (fall back to the plugin's own order).
- Use: intersect with the PSI candidates that actually declare `name` (the model proposes, PSI validates); order the auto-import
  popup by the model's order; if PSI resolves exactly one candidate there is nothing to rank.
- `prior(name)` = `rankImports(name, emptyList())` — the "most frequent path for the name" baseline.

## `rankCoImports(currentImports, limit = 20): List<Suggestion>`

Imports that usually accompany the present ones ("files importing gin and net/http also import net/http/httptest"), best first,
present imports excluded; `score = ln p(path) + Σ PMI(path, import)`. Candidates come from the stored partner lists of the present
imports (top 32 per path), so a file with no known import gets an empty list. Hit@5 on the test folds: Go 0.48 (frequency order
0.44), C# 0.59 (0.48) — a hint list, not a decision.

## Cost

Measured in `ImportsParityTest` (warmed up, this server): `rankImports` 1–3 µs, `rankCoImports` 15–20 µs, load 80–120 ms.
Both calls are allocation-light and safe on the EDT, but the load belongs in a background task.

## Tests

- `ImportsModelTest` — round trip through the container and the two queries on a hand-made artifact.
- `ImportsParityTest` — the published artifacts vs the Python reference (`tools/imports/imports_model.py`) on 50 + 50 queries per
  language (`src/test/resources/imports/<lang>-parity.txt`); needs `../models` or `$CML_MODELS_DIR`, skipped otherwise.
