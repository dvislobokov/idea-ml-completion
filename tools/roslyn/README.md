# tools/roslyn — Roslyn-based filter / context tool (C#)

Standalone .NET 10 console tool (`CmlRoslyn`) that gives the ML completion engine the semantic facts the IDE has and
the 31 M model does not: (A) a **filter** — does every identifier of a generated line resolve at that position
(is `x.Name` really a member of the type of `x`); (B) a **context** block — one-line signatures of the symbols a file
uses from other files of the same repository plus member lists of receiver types, to be put into the prompt / training
data. Measurement results: `REPORT.md`.

## Build

```
cd tools/roslyn/refpacks && dotnet restore          # once: pulls ~36 popular NuGet packages + the WindowsDesktop ref pack
cd ../CmlRoslyn && dotnet build -c Release
alias cmlroslyn='dotnet tools/roslyn/CmlRoslyn/bin/Release/net10.0/CmlRoslyn.dll'
```

## How a repository is loaded

The corpus snapshots contain only `*.cs` (+ LICENSE/README) — no `.csproj`/`.sln` — so `MSBuildWorkspace` /
`dotnet restore` of the repository is impossible. Every repository becomes ONE `CSharpCompilation` over all its
`.cs` files (`--max-files`, default 8000) with these metadata references (`Refs.cs`):

- `Microsoft.NETCore.App.Ref` + `Microsoft.AspNetCore.App.Ref` from the installed SDK (`/usr/lib/dotnet/packs`),
- `Microsoft.WindowsDesktop.App.Ref` (WinForms/WPF) and every compile-time assembly of `refpacks/RefPacks.csproj`
  (Newtonsoft.Json, xunit, NUnit, MSTest, Moq, NSubstitute, FluentAssertions, Microsoft.Extensions.*, EF Core, Serilog,
  NLog, log4net, Rx, Dapper, AutoMapper, MediatR, CommunityToolkit.Mvvm, Polly, RestSharp, Avalonia, Prism, SharpDX,
  OpenTK, ImageSharp, BenchmarkDotNet, Swashbuckle, …), read from `refpacks/obj/project.assets.json`;
- `--extra-refs dir;dir` adds more DLLs.

  The set was extended data-driven with the `usings` command (unresolved `using` namespaces over the 199 test repos):
  ~115 packages now, incl. UnityEngine.Modules, MonoGame, Krafs.Rimworld.Ref, Lib.Harmony, GodotSharp, Revit API, Npgsql,
  StackExchange.Redis, MongoDB, EF6, gRPC/protobuf, OpenTelemetry, Aspire.Hosting, Roslyn, MSBuild, PowerShell SDK,
  Spectre.Console, System.CommandLine, SkiaSharp, NAudio, ReactiveUI, Xamarin.Forms, Windows SDK contracts (`.winmd`),
  the Android / iOS reference packs (`PackageDownload`, globbed from `~/.nuget/packages`).
- **TypeIndex** (`TypeIndex.cs`, on by default, `--no-index` to disable): the corpus lost the `.csproj` and with it
  `<Using Include=…>` / SDK implicit usings, so a name that does not bind is looked up by simple name (+ arity) over every
  referenced assembly and the repository source; an extension method is accepted when one with that name exists anywhere;
  a receiver whose declared type is an error type takes the ambiguity candidates or the index type of that name. Names that
  resolve only this way are counted in `*_via_index`; `*_resolvable_strict` is the verdict of the semantic model alone.
  This is what the IDE with the real project would see, minus packages that are not on NuGet at all.

Consequences: packages outside the refpacks set (Terraria/tModLoader, 7 Days to Die, DOTween, UnityEditor, …) are
unresolved — such positions show up as "receiver type unknown" / "true line does not resolve"; types with the same name
in several projects of one repository bind as ambiguities (counted as "resolves"). Preprocessor symbols:
`DEBUG TRACE NET NETCOREAPP NET10_0 NET*_OR_GREATER`. Run `cmlroslyn selftest` after touching the judge (25 synthetic
receiver / index cases). `refpacks/obj/project.assets.json` is found by walking up from the binary or the cwd (or
`--assets` / `CMLROSLYN_ASSETS`); a missing file is an error (`--allow-no-assets` to run with the SDK ref packs only — the
first measurement ran like that by accident and lost every NuGet type).

## Commands

```
cmlroslyn filter --positions <eval.json> --repos <root> --out <jsonl> [--mode spm] [--context]
                 [--threads 4] [--repo-timeout 900] [--max-files 8000] [--max-repos N] [--ctx-chars 1200] [--ctx-members 40]
```
Input: an eval dump of `tools/nn/eval/eval_inline.py` (`modes.<mode>.positions`: `repo`, `path`, `line`, `true`, `gen`,
`exact`, `conf_prod`, `kind`, `typed`). For each position the caret is reconstructed (first offset of the line whose
remainder equals `true` up to trailing whitespace/comment; `prefix_ok` checks it against the harness's `typed`
remainder), the generated text is spliced in place of the true rest (`SyntaxTree.WithChangedText` +
`Compilation.ReplaceSyntaxTree`) and every `SimpleNameSyntax` in the spliced span is bound with the semantic model.
Output per position (JSONL): `gen_resolvable`, `gen_first_unresolved`, `gen_first_is_member`, `gen_recv_known`,
`gen_recv_type`, `gen_candidates` (distinct member names of the receiver), the same `true_*` fields for the real line
(the baseline of what the references can resolve), `judge_ms`, `status` (`ok | no-file | no-line | no-cursor | timeout | error`).
With `--context` it also builds the context block from the file with the middle removed and reports whether the
identifiers of the true rest are in the file prefix, only in the context block, or nowhere (`ids_*`, `ctx`).
At the end it prints the headroom table (precision / shown with and without the filter per `conf_prod` threshold; the
"ideal filter" column applies the filter only where the true line resolves, i.e. what a fully restored project would give).

```
cmlroslyn usings --positions <eval.json> | --manifest <manifest.jsonl> [--fold test] --repos <root> --out <tsv> [--threads 4]
```
Unresolved `using` namespaces per repository (TSV: namespace, repos, files, first repo names) — the input for extending
`refpacks/RefPacks.csproj`. `python3 tools/roslyn/report.py <filter.jsonl> [--bpe tokenizer/cs-16384.bpe]` prints the
coverage / headroom report of a filter run (REPORT.md is made from it).

```
cmlroslyn context --manifest <manifest.jsonl> --repos <root> --out <dir> [--fold test] [--max-repos N]
                  [--max-files-per-repo N] [--repos-list <file>] [--threads 4] [--ctx-chars 1200] [--ctx-members 40]
```
One `<dir>/<repo>.jsonl` per repository: `{repo, path, chars, lines, signatures, member_lists, dropped, ms, context}`.

## Context block format (`ContextBuilder.cs`)

Plain text, one declaration per line, entries in order of first reference in the file, deterministic, capped at
`--ctx-chars` (1200 chars ≈ 300 BPE tokens); only symbols declared in OTHER files of the same repository:

```
class Result<T> : IResult
static Result<T> Result<T>.Success(T value)
string ForecastRequestDto.PostalCode { get; set; }
members Result<T>: Conflict() CorrelationId Created() Errors IsSuccess Status Success() Value …+3
```
`members X:` lines appear for every receiver type of a `recv.Member` access that is a solution type; methods carry
`()`, nested types `{}`, names sorted, capped at `--ctx-members`.
