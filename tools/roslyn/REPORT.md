# Roslyn filter / context on the 3 000 C# eval positions — coverage and headroom (2026-10-07)

Positions: `~/work/ml-data/csharp/nn/eval-cs31m-e2-lr2e3.json`, mode `spm` (cs31m-e2-lr2e3, 199 test repos, 3 000 positions).
Outputs: `/root/work/roslyn-work/filter-r{2,3,4}.jsonl`; numbers below are from r3 (r4 was still running when this was written; refresh with `python3 tools/roslyn/report.py filter-r4.jsonl --bpe …/cs-16384.bpe`.

## 1. Coverage of the semantic model (does the TRUE line resolve?)

| run | references | true line resolves | strict (semantic model only) | receiver known (member accesses) | after-dot: all receivers known |
|---|---|---|---|---|---|
| e2 (previous agent) | SDK ref packs only — `project.assets.json` was NOT found, no NuGet types at all | 86.1 % | 86.1 % | 74.2 % (1573 / 2119) | 73.0 % |
| r2 | + the 36 refpacks packages (xunit, Newtonsoft, EF, …) | 89.8 % | 89.8 % | 79.7 % | 81.1 % |
| r3 | + ~80 packages from the `usings` scan, Android/iOS/Windows refs, **TypeIndex** fallback | 94.2 % | 92.3 % | 89.3 % | 88.9 % |
| r4 | + xunit.v3, DuckDB, OData, Aspire 13, members over all ambiguity candidates | (running — unit `roslyn-r4`, `filter-r4.out`) |  |  |  |

What was wrong and what was fixed:

1. **No NuGet references in the first run.** The binary copied to `/root/work/roslyn-work/build2` looked for
   `refpacks/obj/project.assets.json` relative to itself, did not find it and silently went on with the SDK ref packs
   (log line `refs: assets = (none)`, 349 assemblies). `Assert`, `Fact`, `Equal`, `Should` were the top unresolved names.
   Now: the file is searched by walking up from the binary and the cwd, a missing file is an error (`--allow-no-assets`).
2. **Receiver detection was not broken.** `true_recv_known = 32` is the field of the FIRST UNRESOLVED name only (it is
   `false` for the 2 584 positions where nothing is unresolved); the real statistic is `true_members_recv_known`:
   1 573 of 2 119 member accesses had a known receiver already in e2. `cmlroslyn selftest` (25 synthetic lines: locals,
   parameters, fields, properties, `this`, static classes `string.`/`Path.`/`Assert.`, call results, chains, `?.`,
   extension methods, generics, indexers, missing usings) passes 25/25.
3. **The corpus has no project files at all** (`find … -name '*.csproj' -o -name '*.sln'` → nothing), so `dotnet restore`
   of a repository is impossible and the stand-in is the synthetic `refpacks/RefPacks.csproj`. The `usings` command lists
   the `using X.Y;` directives that do not bind over the 199 repos (`/root/work/roslyn-work/usings-r2.tsv`: 2 365
   namespaces in 176 repos; UnityEngine 33 repos / 3 004 files, UnityEditor 23, StackExchange.Redis 10, Npgsql 8 / 888 files,
   Android/iOS 7, Microsoft.Xna 5 / 2 367 files, Terraria 2 / 1 573 files, …). ~80 packages were added (Unity, MonoGame,
   RimWorld, Harmony, Godot, Revit, Npgsql, Redis, MongoDB, EF6, gRPC, OpenTelemetry, Aspire, Roslyn, MSBuild, PowerShell,
   Spectre, System.CommandLine, SkiaSharp, NAudio, ReactiveUI, Xamarin.Forms, Windows SDK contracts as `.winmd`, the
   Android / iOS reference packs via `PackageDownload`) → 852 assemblies.
4. **Lost `<Using Include=…>` of the csproj / Directory.Build.props.** Meziantou.Framework, DualSenseClient (NUnit without
   `using NUnit.Framework;`), xunit-v3 projects etc. rely on csproj-level global usings. `TypeIndex` emulates what the IDE
   with the real project sees: an unbound simple name is looked up by name + arity over all referenced assemblies and the
   repository source, an unbound member is accepted when an extension method of that name exists anywhere, a receiver whose
   declared type is an error type takes the ambiguity candidates or the index type of that name. Names that resolve only
   this way are counted separately (`*_via_index`, 99 positions; `*_resolvable_strict` is the semantic model alone).
5. Member lookup on an ambiguous receiver (`Xunit.Assert` from xunit v2 and v3) now checks every candidate.

### What still does not resolve (r3: 174 positions)

true line does NOT resolve: 174: member / receiver unknown 74, simple name 53, member / receiver known 47

Top-30 first unresolved names (name, kind, count, repos):
```
  Center                   member    5  SOTS 5
  Main                     simple    4  SOTS 4
  NPC                      simple    4  SOTS 4
  Count                    member    3  Meziantou.Framework 2, Util 1
  Name                     member    3  Meziantou.Framework 2, SOTS 1
  Statements               member    3  Meziantou.Framework 3
  DuckDBParameter          simple    3  PerformanceMonitor 3
  Parameters               member    3  PerformanceMonitor 3
  velocity                 member    3  SOTS 3
  Resources                member    2  osu 1, MechJeb2 1
  EvaluateAsync            member    2  Meziantou.Framework 2
  Current                  member    2  PerformanceMonitor 2
  LongProp                 member    2  Atomics 2
  Common                   member    2  VarianDeveloper 2
  Dust                     simple    2  SOTS 2
  scale                    member    2  SOTS 2
  ItemType                 member    2  SOTS 2
  ToRotation               member    2  SOTS 2
  Item                     simple    2  SOTS 2
  ai                       member    2  SOTS 2
  direction                member    2  SOTS 2
  ErrorNoBotsDefined       member    1  ArchiSteamFarm 1
  AsUnsignedInteger        member    1  ArchiSteamFarm 1
  Bool                     member    1  Util 1
  Instruction              member    1  Discord.Net 1
  ForIncrementalGenerator  member    1  NServiceBus 1
  Keyboard                 member    1  DownmarkerWPF 1
  PetaByte                 member    1  Meziantou.Framework 1
  TortoiseIDiff            member    1  Meziantou.Framework 1
  FallBackToBaseType       member    1  Meziantou.Framework 1
```

repos with most unresolved true lines: VortexOfRainbows__SOTS 56/159, meziantou__Meziantou.Framework 23/254, erikdarlingdata__PerformanceMonitor 22/471, CommunityToolkit__Aspire 6/24, dmustanger__7dtd-ServerTools 6/37, OData__WebApi 4/88, Meowmaritus__DSAnimStudio 3/29, BoiHanny__vrcosc-magicchatbox 3/38, grevit-dev__Grevit 3/9, cnthigu__conquer-src-5517 3/32

- **Game engines that are not on NuGet** — Terraria / tModLoader (`VortexOfRainbows__SOTS`, 56 of 159 positions:
  `Projectile`, `NPC`, `Main`, `Vector2`-with-Terraria-extensions `RotatedBy`, `Center`, `velocity`), 7 Days to Die
  (`dmustanger__7dtd-ServerTools`), Conquer Online server, DSAnimStudio (SoulsFormats). Nothing to do short of shipping
  the game assemblies; this is the single largest remaining group (SOTS alone is 1.9 % of all positions).
- **Generated code that is not in the corpus**: WPF `x:Name` fields from `.xaml` (`EntraMfaPanel`, `QueryHeatmapChart`),
  source-generator output (`ForIncrementalGenerator`, Regex / JSON / Yaml generators), `.g.cs` of resource files
  (`Resources.X`).
- **Same-repo ambiguities / failed generic inference**: types with the same name in several projects of one repository
  (the whole repository is ONE compilation) make the receiver an error type; lambda parameters whose type depends on it
  (`d.Name`, `entries.Count`, `reading.Status`) stay unknown. A per-project compilation would need the project files.
- Private NuGet feeds / packages not added (`DiffEngine`, `TomlUnknownDerivedTypeHandling`, Meziantou's own packages that
  the repo consumes as NuGet, `IEdmModel` of an older OData), enum members of a newer package version than the one restored
  (`DnsQueryType.AAAA`, `StableDiffusionCppImageVariant.CudaSpark`).

## 2. Headroom of the filter (3 000 positions, r3; r4 = same + xunit.v3/DuckDB/OData/Aspire refs, running)

Filter = drop the generated line when any simple name / member name in it does not resolve (with the TypeIndex fallback).
"Complete model only" = the filter acts only where the true line resolves (what a fully restored project would give — there
the filter never removes a right line by construction).

| conf >= | shown | exact | +filter shown | exact | wrong removed | right removed | +filter (complete model only) shown | exact | wrong removed | right removed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 1233 (41.1 %) | 89.5 % | 1192 (39.7 %) | 89.9 % | 9 of 129 | 32 of 1104 | 1227 (40.9 %) | 90.0 % | 6 of 129 | 0 of 1104 |
| 0.6 | 1025 (34.2 %) | 91.7 % | 996 (33.2 %) | 91.9 % | 4 of 85 | 25 of 940 | 1021 (34.0 %) | 92.1 % | 4 of 85 | 0 of 940 |
| 0.7 | 817 (27.2 %) | 94.4 % | 798 (26.6 %) | 94.4 % | 1 of 46 | 18 of 771 | 816 (27.2 %) | 94.5 % | 1 of 46 | 0 of 771 |
| 0.8 | 599 (20.0 %) | 97.0 % | 587 (19.6 %) | 96.9 % | 0 of 18 | 12 of 581 | 599 (20.0 %) | 97.0 % | 0 of 18 | 0 of 581 |
| 0.9 | 337 (11.2 %) | 98.8 % | 333 (11.1 %) | 98.8 % | 0 of 4 | 4 of 333 | 337 (11.2 %) | 98.8 % | 0 of 4 | 0 of 333 |

Right lines removed by the filter at conf >= 0.7 (why the correct line does not resolve):
  DuckDBParameter          simple  PerformanceMonitor           2
  Count                    member  Util                         1
  IsSuccess                member  Meziantou.Framework          1
  Current                  member  PerformanceMonitor           1
  SkipWhen                 member  PerformanceMonitor           1
  CreateCommand            member  PerformanceMonitor           1
  Parameters               member  PerformanceMonitor           1
  CommandText              member  PerformanceMonitor           1
  IfNoneMatch              member  RESTier                      1
  item                     member  Grevit                       1
  Against                  member  melodee                      1
  LongProp                 member  Atomics                      1
  Common                   member  VarianDeveloper              1
  newTile                  member  SOTS                         1
  Center                   member  SOTS                         1

Reading: at conf ≥ 0.7 the model shows 27.2 % of positions at 94.4 % precision; the filter removes 1 of the 46 wrong
lines and 18 (all of them in positions where the true line does not resolve either, i.e. model incompleteness, not a judge error) right ones. The wrong lines that remain are overwhelmingly *resolvable* code that is simply not what the
author wrote (same identifiers, different literal / order / method; `return pairs;` vs `Assert.NotEmpty(pairs);`) — a
semantic filter cannot catch those. The headroom of resolution-only filtering on this model is therefore ≈ +0.1–0.5 p.p.
precision for −0.5 p.p. shown; it is worth having as a safety net (it never costs a right line when the model is
complete) but it is not a quality lever. Positions after a dot: _LINE.

## 3. Context block (r3) (file without the middle, `--context`, cap 1 200 chars / 40 members per type)

context block: mean 967 chars, BPE tokens mean 246, median 275, p90 353, empty 6.2 %
identifiers of the true rest: 6562; in the file prefix 77.3 %, only in the context block 2.0 %, nowhere 20.8 %
positions with identifiers 2164: some identifier only in the context 5.0 %, context needed and sufficient 3.0 %

The context block is small (≈ 250 BPE tokens) and cheap (≈ 150 ms per file incl. the hole compilation), but it adds
identifiers the prefix does not have in only 2 % of the identifier occurrences (5 % of positions have at least one such
identifier, 3 % have all their missing identifiers there); 21 % of the identifiers of the true rest are nowhere (new names,
literals-as-identifiers, members of unresolved types). Same conclusion as the Go PSI study: the context block is a speed /
precision lever for `.`-positions, not a recall lever.

## 4. Raw examples (r3)

10 raw examples (conf >= 0.5; wrong+dropped, right+dropped (true line unresolvable too), wrong+kept, right+kept):
- zzzprojects__Eval-Expression.NET src/Examples.Expressions.Eval/LINQ_Dynamic/Aggregate_Operators/Min.cs:22 conf 0.63 wrong, filter DROPS (unresolved `Length` on int)
    true: `();`
    gen:  `(w => w.Length);`
- VortexOfRainbows__SOTS Items/Pyramid/TheDarkEye.cs:320 conf 0.76 wrong, filter DROPS (unresolved `ModContent`, simple name)
    true: `finalPos.Y - 6 + circularRotation.Y), 0, 0, 21);`
    gen:  `finalPos.Y - 6 + circularRotation.Y), 0, 0, ModContent.DustType<Dusts.ShortlivedCurseDust>());`
- erikdarlingdata__PerformanceMonitor Darling/Darling.Tests/PgTargetMeasuredLineageTests.cs:191 conf 0.64 wrong, filter DROPS (unresolved `file`, simple name)
    true: `("PgTargetScorer.Queries.cs"));`
    gen:  `(file));`
- erikdarlingdata__PerformanceMonitor PerformanceMonitor.Ui/ChartHoverHelper.cs:303 conf 0.64 EXACT, filter DROPS (unresolved `ScottPlot`, simple name)
    true: `ScottPlot.Coordinates(nearest.X, nearest.Y));`
    gen:  `ScottPlot.Coordinates(nearest.X, nearest.Y));`
- erikdarlingdata__PerformanceMonitor Darling/Darling.Tests/StoreApplicationNamePinTests.cs:91 conf 0.75 EXACT, filter DROPS (unresolved `SkipWhen` on Assert)
    true: `Assert.SkipWhen(string.IsNullOrEmpty(baseConnectionString),`
    gen:  `Assert.SkipWhen(string.IsNullOrEmpty(baseConnectionString),`
- VortexOfRainbows__SOTS Void/VoidDamageClasses.cs:13 conf 0.70 EXACT, filter DROPS (unresolved `Full`, receiver unknown)
    true: `.Full;`
    gen:  `.Full;`
- VortexOfRainbows__SOTS SOTS.cs:295 conf 0.59 wrong, filter keeps
    true: `reader.ReadSingle();`
    gen:  `reader.ReadInt32();`
- melodee-project__melodee tests/Melodee.Tests.Common/Jobs/JobResultTests.cs:72 conf 0.61 wrong, filter keeps
    true: `JobResultStatus.Success, "Message 2");`
    gen:  `JobResultStatus.Failed, "Message 2");`
- CircuitLord__NotReaper Assets/Scripts/Tools/UndoRedoManager.cs:572 conf 0.92 EXACT, filter keeps
    true: `generatedNotes) {`
    gen:  `generatedNotes) {`
- HaloMods__Prometheus Halo 2/Tags/Classes/render_model.cs:369 conf 0.68 EXACT, filter keeps
    true: `CompressionInfo[x].Read(reader);`
    gen:  `CompressionInfo[x].Read(reader);`

## 5. Timing

judge time per position (true + generated line, incl. TypeIndex lookups on big repos): median 8 ms, p90 637 ms, p99 1342 ms, max 3738 ms; whole run 27.6 min on 4 threads (r2 without the index: 5 min — `GetSymbolsWithName` over 5 000-file repos per unresolved name is the cost; cached per name in r4). Context block build: mean 147 ms per position.

## Files

- `/root/work/roslyn-work/filter-r4.jsonl` (+ `filter-r2`, `filter-r3`, `filter-cs31m-e2` = e2), `usings-r2.tsv`, `report-r4.txt`.
- Tool: `tools/roslyn/CmlRoslyn` (`filter`, `context`, `usings`, `selftest`), `tools/roslyn/refpacks/RefPacks.csproj`, `tools/roslyn/report.py`.
