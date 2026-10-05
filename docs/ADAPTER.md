# Plugin adapter contract

How an IDE plugin (`idea-golang-support`, `idea-dotnet-support`) plugs its PSI into the engine. The engine never depends
on the IntelliJ platform; the plugin embeds `ml-core` and implements three things: an `MlLanguage`, an offline example
generator, and a weigher. Generator and weigher must compute the language features with the **same code**.

## 1. Embedding (git subtree)

```sh
# in the plugin repository, once
git subtree add --prefix=ml https://github.com/dvislobokov/idea-ml-completion.git main --squash
# later updates
git subtree pull --prefix=ml https://github.com/dvislobokov/idea-ml-completion.git main --squash
```

`settings.gradle.kts` of the plugin:

```kotlin
include(":ml-core")
project(":ml-core").projectDir = file("ml/ml-core")
```

`ml-core/build.gradle.kts` only declares `implementation(kotlin("stdlib"))`; the plugin's root build applies the Kotlin
plugin to all subprojects, as this repository does. `ml-train` and `tools/` come along with the subtree but are not
included in the plugin build. Rules for `ml-core`: no third-party runtime dependencies, Kotlin `apiVersion` 2.3 (stdlib
comes from the platform), no `java.awt`/`com.intellij` imports.

## 2. `MlLanguage`

One object per plugin (`ml-core` ships lexer-only `CSharpLanguage` / `GoLanguage` in `lex/Languages.kt`; the adapter
subclasses or wraps them):

```kotlin
object GoMlLanguage : MlLanguage by GoLanguage {
    override val rankFeatures = listOf("kind_local", "kind_param", "kind_field", "kind_method", "kind_func", "kind_type",
        "kind_package", "kind_keyword", "expected_type_match", "scope_level_log", "same_function", "decl_distance_log",
        "is_imported", "receiver_is_pointer")
}
```

`rankFeatures` is the **language block** of the ranker schema: `FeatureSchema.common(languageFeatures = rankFeatures)`.
Values are floats in roughly `[0, 10]`: binary flags, small counts, `ln(1 + x)` for unbounded quantities. The schema
hash is stored in the model; changing the list means retraining.

## 3. Offline generator (runs inside the platform, headless)

A Gradle task of the plugin (`mlDataset`, modelled on the existing `corpusTest` tasks that use
`intellijPlatformTesting.testIde`) walks the corpus and, for every sampled completion position, records one
`TrainingExample` into shards (`ExampleShards.Writer`, one `.cmlx` file per repository or per N files).

Per file:

```kotlin
val tokens = language.tokenizer.tokens(text)                // ml-core lexer, offsets into the text
val state = FileState(vocab, withCache = cacheLambda > 0)   // fed token by token
val extractor = FeatureExtractor(schema, vocab, lm, cacheLambda)
for ((i, t) in tokens.withIndex()) {
    if (t.kind == TokenKind.IDENT && sampled(i)) {
        val prefixLen = samplePrefixLength()                 // 0–2 characters, same distribution as ProxyExampleGenerator
        val caret = t.offset + prefixLen
        val candidates = completionAt(psiFile, caret)        // the plugin's own candidate list (see §5)
        val chosen = candidates.indexOfFirst { it.insertText == t.text }
        if (chosen >= 0 && candidates.size > 1) {
            val names = candidates.map { it.insertText }.toTypedArray()
            val lang = candidates.map { languageFeatures(it, context) }.toTypedArray()   // the shared feature code
            val base = extractor.features(state, t.text.substring(0, prefixLen), names, lang)
            writer.add(TrainingExample(contextKind(position), base, chosen, names))
        }
    }
    state.add(t)
}
```

Rules:
- **Answer = the token in the source.** No oracle is needed; positions whose answer is not in the candidate list are
  skipped (count them: that is the plugin's recall, worth reporting).
- **Candidates = exactly what the IDE would show** at that caret with that prefix, before any ML reordering. Keep the
  plugin's own rule-based order (priority) as an extra language feature, e.g. `rule_priority_log`, so the ranker can
  reuse it and the evaluation can compare "rules only" against "ML".
- **Context kind**: map the PSI position onto `ContextKind` (`AFTER_DOT`, `STATEMENT_START`, `ARGUMENT`,
  `TYPE_POSITION`, `ASSIGN_RHS`, `OTHER`); the proxy's token heuristic `ProxyExampleGenerator.contextKind` is the fallback.
- **Caps**: ≤ 60 positions per file (deterministic sampling by file hash), ≤ 100 candidates per list (keep the answer,
  sample the rest).
- **Repository folds**: the generator takes the same `sets/rank.txt` / `sets/test.txt` lists as `ml-train`; the LM used
  for `lm_logprob` is the one trained on `sets/lm.txt` (cross-fitting — see CHANGELOG e07).
- **Shard header**: `language = MlLanguage.id`, `source = "<plugin> <version> <generator git sha>"`.

Then on the training side:

```sh
ml-train l1 --lang go --shards ../ml-data/go/shards/rank --test-shards ../ml-data/go/shards/test --out rank.cml
ml-train eval-rank --lang go --rank rank.cml --shards ../ml-data/go/shards/test
```

## 4. Weigher (IDE path)

`CompletionWeigher` / `LookupElementWeigher` of the plugin:

1. Tokenise the text before the caret with `language.tokenizer` (incremental: cache per document modification stamp),
   feed a `FileState` (its cache LM is the per-file dynamic model; keep one per open document, append on edits).
2. For the shown list: `names`, language-block values from the **same** `languageFeatures(candidate, context)` function,
   `extractor.features(state, prefix, names, lang)`, then `LinearRanker.scores(TrainingExample(kind, base, 0))`.
3. Scores are the weigher's key (higher first). Budget: < 5 ms for 500 candidates — `FeatureExtractor` is O(candidates ×
   order) hash lookups; `fillListFeatures` sorts the list twice.

Parity test (mandatory per plugin): a fixture-based test that runs `completeBasic()` at a position and asserts that
the feature vectors computed by the weigher equal those produced by the generator for the same file/offset.

## 5. Per-language notes

**Go** (`idea-golang-support`): candidates from `GoCompletionContributor`, features from `GoCompletionCandidate`
(kind, scope level, expected-type match, ...). Everything needed exists; see the survey in the planning notes.

**C#** (`idea-dotnet-support`): candidates from `NativeCSharpCompletion.items(place, file, matcher)` and
`NativeCSharpMemberCompletion` (after `.`); features from `NativeCompletionKind`, `NativeCSharpExpectations`,
`CSharpNameLikeness`, the rule priority. Needs stub indices and the assembly index (not dumb mode); without built
project assemblies the member lists after `.` contain only solution types — acceptable for a first model, improve by
indexing the .NET runtime and the most common packages once on the server.
