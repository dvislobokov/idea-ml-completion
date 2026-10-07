using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Text;

namespace CmlRoslyn;

public static class Program
{
    public static bool UseIndex = true;
    static readonly JsonSerializerOptions JsonOpts = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public static int Main(string[] args)
    {
        if (args.Length == 0) { Usage(); return 2; }
        var opts = new Dictionary<string, string>();
        for (var i = 1; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--")) { Console.Error.WriteLine($"unexpected argument {args[i]}"); return 2; }
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--")) opts[args[i][2..]] = args[++i]; else opts[args[i][2..]] = "true";
        }
        try
        {
            return args[0] switch
            {
                "filter" => Filter(opts),
                "context" => Context(opts),
                "usings" => Usings(opts),
                "selftest" => SelfTest.Run(opts),
                _ => Usage(),
            };
        }
        catch (Exception e) { Console.Error.WriteLine(e); return 1; }
    }

    static int Usage()
    {
        Console.Error.WriteLine("""
            CmlRoslyn — Roslyn-based filter / context tool for the ML completion engine (see README.md)
              filter  --positions <eval.json> --repos <root> --out <jsonl> [--mode spm] [--context] [--threads 4]
                      [--repo-timeout 900] [--max-files 8000] [--max-repos N] [--ctx-chars 1200] [--ctx-members 40] [--no-implicit-usings] [--no-index]
              context --manifest <manifest.jsonl> --repos <root> --out <dir> [--fold test] [--max-repos N] [--max-files-per-repo N]
                      [--threads 4] [--repo-timeout 900] [--ctx-chars 1200] [--ctx-members 40]
              usings  --positions <eval.json> | --manifest <manifest.jsonl> [--fold test] [--repos-list <file>] --repos <root> --out <tsv>
                      [--threads 4]   unresolved `using` namespaces across the repositories (what to add to refpacks)
              selftest                receiver-type detection on a synthetic file
              common: --assets <project.assets.json> (default: refpacks/obj/project.assets.json found by walking up from the binary / cwd;
                      fails when missing unless --allow-no-assets)
                      --dotnet-root <dir> --extra-refs <dir;dir>
            """);
        return 2;
    }

    static string Opt(Dictionary<string, string> o, string k, string d) => o.TryGetValue(k, out var v) ? v : d;
    static int OptInt(Dictionary<string, string> o, string k, int d) => o.TryGetValue(k, out var v) ? int.Parse(v) : d;

    internal static List<MetadataReference> LoadRefs(Dictionary<string, string> o)
    {
        var assets = Opt(o, "assets", Environment.GetEnvironmentVariable("CMLROSLYN_ASSETS") ?? "");
        if (assets == "")
        {
            // walk up from the binary (bin/Release/net10.0 → CmlRoslyn → tools/roslyn) and from the cwd looking for refpacks/obj/project.assets.json
            foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
            {
                var dir = Path.GetFullPath(start);
                for (var i = 0; i < 6 && dir != null && assets == ""; i++, dir = Path.GetDirectoryName(dir))
                    foreach (var rel in new[] { "refpacks/obj/project.assets.json", "roslyn/refpacks/obj/project.assets.json", "tools/roslyn/refpacks/obj/project.assets.json" })
                    {
                        var c = Path.Combine(dir, rel);
                        if (File.Exists(c)) { assets = c; break; }
                    }
                if (assets != "") break;
            }
        }
        if (assets == "" || !File.Exists(assets))
        {
            if (o.ContainsKey("allow-no-assets")) Console.Error.WriteLine("refs: WARNING no project.assets.json — only the SDK ref packs are referenced, NuGet types will not resolve");
            else throw new InvalidOperationException("refpacks/obj/project.assets.json not found: run `dotnet restore` in tools/roslyn/refpacks or pass --assets <file> (or --allow-no-assets)");
        }
        Console.Error.WriteLine($"refs: assets = {(assets == "" ? "(none)" : assets)}");
        return Refs.Load(o.TryGetValue("dotnet-root", out var dr) ? dr : null, assets == "" ? null : assets, o.TryGetValue("extra-refs", out var er) ? er : null, Console.Error);
    }

    // ------------------------------------------------------------------------------------------------ filter

    sealed class Pos
    {
        public string Repo = "", Path = "", Kind = "", True = "", Gen = "", Typed = "";
        public int Line; public double Conf; public bool Exact; public JsonNode Raw = null!;
        public Dictionary<string, object?> Out = new();
    }

    static int Filter(Dictionary<string, string> o)
    {
        var positionsPath = o["positions"]; var reposRoot = o["repos"]; var outPath = o["out"];
        var mode = Opt(o, "mode", "spm");
        var threads = OptInt(o, "threads", 4);
        var repoTimeout = OptInt(o, "repo-timeout", 900);
        var maxFiles = OptInt(o, "max-files", 8000);
        var maxRepos = OptInt(o, "max-repos", int.MaxValue);
        var withCtx = o.ContainsKey("context");
        var ctxChars = OptInt(o, "ctx-chars", 1200);
        var ctxMembers = OptInt(o, "ctx-members", 40);

        // the harness dumps Python NaN / Infinity literals, which are not JSON
        var rawJson = Regex.Replace(File.ReadAllText(positionsPath), @"(?<=[:,\[\s])-?(NaN|Infinity)(?=[,\]\}\s])", "null");
        var doc = JsonNode.Parse(rawJson)!;
        var arr = doc["modes"]![mode]!["positions"]!.AsArray();
        var positions = new List<Pos>();
        foreach (var n in arr)
        {
            positions.Add(new Pos
            {
                Repo = n!["repo"]!.GetValue<string>(), Path = n["path"]!.GetValue<string>(), Line = n["line"]!.GetValue<int>(),
                Kind = n["kind"]?.GetValue<string>() ?? "", True = n["true"]!.GetValue<string>(), Gen = n["gen"]!.GetValue<string>(),
                Typed = n["typed"]?.GetValue<string>() ?? "", Conf = n["conf_prod"]!.GetValue<double>(), Exact = n["exact"]!.GetValue<bool>(), Raw = n,
            });
        }
        Console.Error.WriteLine($"{positions.Count} positions, {positions.Select(p => p.Repo).Distinct().Count()} repos, mode {mode}");
        var refs = LoadRefs(o);
        Repo.UseImplicitUsings = !o.ContainsKey("no-implicit-usings");
        UseIndex = !o.ContainsKey("no-index");

        var byRepo = positions.GroupBy(p => p.Repo).OrderByDescending(g => g.Count()).Take(maxRepos).ToList();
        var t0 = Stopwatch.StartNew();
        var done = 0;
        var lockObj = new object();
        var repoLog = new List<string>();
        Parallel.ForEach(byRepo, new ParallelOptions { MaxDegreeOfParallelism = threads }, g =>
        {
            var sw = Stopwatch.StartNew();
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(repoTimeout));
            var ct = cts.Token;
            var list = g.ToList();
            string status = "ok"; Repo? repo = null;
            try
            {
                var root = Path.Combine(reposRoot, g.Key);
                if (!Directory.Exists(root)) { status = "no-repo"; }
                else
                {
                    repo = Repo.Load(g.Key, root, refs, maxFiles, ct);
                    var models = new Dictionary<SyntaxTree, SemanticModel>();
                    foreach (var p in list)
                    {
                        try { JudgePosition(repo, models, p, withCtx, ctxChars, ctxMembers, ct); }
                        catch (OperationCanceledException) { p.Out["status"] = "timeout"; status = "timeout"; }
                        catch (Exception e) { p.Out["status"] = "error"; p.Out["error"] = e.GetType().Name + ": " + e.Message; }
                    }
                }
            }
            catch (OperationCanceledException) { status = "timeout"; }
            catch (Exception e) { status = "error: " + e.Message; }
            foreach (var p in list) p.Out.TryAdd("status", status == "ok" ? "ok" : status.Split(':')[0]);
            var line = $"{g.Key}: {list.Count} positions, {repo?.FileCount ?? 0} files{(repo?.Truncated == true ? " (truncated)" : "")}, load {repo?.LoadSeconds ?? 0:F1} s, total {sw.Elapsed.TotalSeconds:F1} s, {status}";
            lock (lockObj) { done++; repoLog.Add(line); Console.Error.WriteLine($"[{done}/{byRepo.Count} {t0.Elapsed.TotalMinutes:F1} min] {line}"); }
        });

        using (var w = new StreamWriter(outPath, false, new UTF8Encoding(false)))
            foreach (var p in positions)
            {
                var rec = new Dictionary<string, object?>
                {
                    ["repo"] = p.Repo, ["path"] = p.Path, ["line"] = p.Line, ["kind"] = p.Kind, ["conf_prod"] = p.Conf, ["exact"] = p.Exact,
                    ["true"] = p.True, ["gen"] = p.Gen,
                };
                foreach (var kv in p.Out) rec[kv.Key] = kv.Value;
                w.WriteLine(JsonSerializer.Serialize(rec, JsonOpts));
            }
        Console.Error.WriteLine($"wrote {outPath} in {t0.Elapsed.TotalMinutes:F1} min");
        PrintSummary(positions, withCtx, Console.Out);
        return 0;
    }

    static readonly Regex TailOk = new(@"^\s*(//.*|/\*.*)?$", RegexOptions.Compiled);

    /// <summary>Find the caret inside a line: the first offset where the rest of the line equals <paramref name="rest"/>
    /// up to trailing whitespace / a trailing comment.</summary>
    static int FindCursor(string lineText, string rest)
    {
        if (rest.Length == 0) return -1;
        var i = lineText.IndexOf(rest, StringComparison.Ordinal);
        while (i >= 0)
        {
            if (TailOk.IsMatch(lineText[(i + rest.Length)..])) return i;
            i = lineText.IndexOf(rest, i + 1, StringComparison.Ordinal);
        }
        return -1;
    }

    static void JudgePosition(Repo repo, Dictionary<SyntaxTree, SemanticModel> models, Pos p, bool withCtx, int ctxChars, int ctxMembers, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        if (!repo.Trees.TryGetValue(p.Path, out var tree)) { p.Out["status"] = "no-file"; return; }
        var text = tree.GetText(ct);
        if (p.Line < 1 || p.Line > text.Lines.Count) { p.Out["status"] = "no-line"; return; }
        var line = text.Lines[p.Line - 1];
        var lineText = line.ToString();
        var cur = FindCursor(lineText, p.True);
        if (cur < 0) { p.Out["status"] = "no-cursor"; p.Out["line_text"] = lineText; return; }
        var abs = line.Start + cur;
        p.Out["prefix_ok"] = p.Typed.Length == 0 || lineText[..cur].EndsWith(p.Typed, StringComparison.Ordinal);

        // the true line, in the original compilation
        if (!models.TryGetValue(tree, out var model)) models[tree] = model = repo.Compilation.GetSemanticModel(tree);
        var index = Program.UseIndex ? repo.Index : null;
        var vt = Judge.JudgeSpan(model, tree.GetRoot(ct), abs, abs + p.True.Length, index, ct);
        Put(p.Out, "true", vt);

        // the generated line spliced in place of the true rest
        var genTree = tree.WithChangedText(text.WithChanges(new TextChange(new TextSpan(abs, p.True.Length), p.Gen)));
        var genComp = repo.Compilation.ReplaceSyntaxTree(tree, genTree);
        var genModel = genComp.GetSemanticModel(genTree);
        var vg = Judge.JudgeSpan(genModel, genTree.GetRoot(ct), abs, abs + p.Gen.Length, index, ct);
        Put(p.Out, "gen", vg);
        p.Out["judge_ms"] = (int)sw.Elapsed.TotalMilliseconds;

        if (withCtx)
        {
            sw.Restart();
            // context from the file WITHOUT the middle (the true rest removed): what training / inference could see
            var holeTree = tree.WithChangedText(text.WithChanges(new TextChange(new TextSpan(abs, p.True.Length), "")));
            var holeComp = repo.Compilation.ReplaceSyntaxTree(tree, holeTree);
            var holeModel = holeComp.GetSemanticModel(holeTree);
            var ctx = ContextBuilder.Build(holeModel, holeTree, ctxChars, ctxMembers, ct);
            var prefixWords = Words(text.ToString(new TextSpan(0, abs)));
            var ctxWords = Words(ctx.Text);
            var ids = Identifiers(p.True);
            int inPrefix = 0, inCtx = 0, nowhere = 0;
            var missing = new List<string>();
            foreach (var id in ids)
            {
                if (prefixWords.Contains(id)) inPrefix++;
                else if (ctxWords.Contains(id)) { inCtx++; missing.Add(id); }
                else nowhere++;
            }
            p.Out["ctx_chars"] = ctx.Text.Length; p.Out["ctx_lines"] = ctx.Lines; p.Out["ctx_sigs"] = ctx.Signatures; p.Out["ctx_members"] = ctx.MemberLists;
            p.Out["ctx_dropped"] = ctx.Dropped;
            p.Out["ids"] = ids.Count; p.Out["ids_in_prefix"] = inPrefix; p.Out["ids_in_ctx_only"] = inCtx; p.Out["ids_nowhere"] = nowhere;
            p.Out["ids_ctx_only_names"] = missing;
            p.Out["ctx_ms"] = (int)sw.Elapsed.TotalMilliseconds;
            p.Out["ctx"] = ctx.Text;
        }
        p.Out["status"] = "ok";
    }

    static void Put(Dictionary<string, object?> o, string k, Verdict v)
    {
        o[k + "_resolvable"] = v.Resolvable; o[k + "_resolvable_strict"] = v.UnresolvedStrict == 0; o[k + "_via_index"] = v.ViaIndex; o[k + "_names"] = v.Names; o[k + "_unresolved"] = v.Unresolved;
        o[k + "_members"] = v.MemberAccesses; o[k + "_members_recv_known"] = v.MemberRecvKnown;
        o[k + "_first_unresolved"] = v.FirstUnresolved; o[k + "_first_is_member"] = v.FirstIsMember;
        o[k + "_recv_known"] = v.FirstRecvKnown; o[k + "_recv_type"] = v.FirstRecvType; o[k + "_candidates"] = v.FirstCandidates;
    }

    static readonly Regex WordRe = new(@"[A-Za-z_][A-Za-z0-9_]*", RegexOptions.Compiled);
    static readonly HashSet<string> Keywords = new("abstract as base bool break byte case catch char checked class const continue decimal default delegate do double else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is lock long namespace new null object operator out override params private protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked unsafe ushort using var virtual void volatile while async await get set add remove value yield where select from let into orderby group join on equals by ascending descending nameof when init record with required file and or not global dynamic partial".Split(' '));

    static HashSet<string> Words(string s) { var h = new HashSet<string>(StringComparer.Ordinal); foreach (Match m in WordRe.Matches(s)) h.Add(m.Value); return h; }
    static List<string> Identifiers(string s)
    {
        var r = new List<string>(); var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (Match m in WordRe.Matches(s)) if (!Keywords.Contains(m.Value) && seen.Add(m.Value)) r.Add(m.Value);
        return r;
    }

    static void PrintSummary(List<Pos> positions, bool withCtx, TextWriter w)
    {
        var all = positions.Count;
        var ok = positions.Where(p => (string?)p.Out.GetValueOrDefault("status") == "ok").ToList();
        var statuses = positions.GroupBy(p => (string?)p.Out.GetValueOrDefault("status") ?? "?").OrderByDescending(g => g.Count());
        w.WriteLine($"positions: {all}; with a semantic verdict: {ok.Count} ({100.0 * ok.Count / all:F1} %); by status: {string.Join(", ", statuses.Select(g => $"{g.Key} {g.Count()}"))}");
        if (ok.Count == 0) return;
        bool B(Pos p, string k) => p.Out.TryGetValue(k, out var v) && v is bool b && b;
        var ms = ok.Select(p => Convert.ToDouble(p.Out["judge_ms"])).OrderBy(x => x).ToList();
        w.WriteLine($"judge time per position: mean {ms.Average():F0} ms, median {ms[ms.Count / 2]:F0} ms, p90 {ms[(int)(ms.Count * 0.9)]:F0} ms");
        w.WriteLine($"true line resolves: {100.0 * ok.Count(p => B(p, "true_resolvable")) / ok.Count:F1} %; generated line resolves: {100.0 * ok.Count(p => B(p, "gen_resolvable")) / ok.Count:F1} %; " +
                    $"prefix check ok: {100.0 * ok.Count(p => B(p, "prefix_ok")) / ok.Count:F1} %");
        w.WriteLine($"exact lines: gen resolves {100.0 * ok.Count(p => p.Exact && B(p, "gen_resolvable")) / Math.Max(1, ok.Count(p => p.Exact)):F1} %; " +
                    $"wrong lines: gen resolves {100.0 * ok.Count(p => !p.Exact && B(p, "gen_resolvable")) / Math.Max(1, ok.Count(p => !p.Exact)):F1} %");
        w.WriteLine();
        w.WriteLine("Headroom of the filter (positions with a verdict; `shown` = conf_prod >= threshold):");
        w.WriteLine("| conf >= | shown | precision | +filter shown | precision | wrong removed | right removed | +ideal filter shown | precision | wrong removed | right removed |");
        w.WriteLine("|---|---|---|---|---|---|---|---|---|---|---|");
        foreach (var t in new[] { 0.5, 0.6, 0.7, 0.8, 0.9 })
        {
            var shown = ok.Where(p => p.Conf >= t).ToList();
            var right = shown.Count(p => p.Exact);
            var kept = shown.Where(p => B(p, "gen_resolvable")).ToList();
            var keptRight = kept.Count(p => p.Exact);
            // ideal: the filter only acts where the true line itself resolves (what a complete project would give)
            var ideal = shown.Where(p => B(p, "gen_resolvable") || !B(p, "true_resolvable")).ToList();
            var idealRight = ideal.Count(p => p.Exact);
            string Pct(int a, int b) => b == 0 ? "—" : $"{100.0 * a / b:F1} %";
            w.WriteLine($"| {t:F1} | {shown.Count} ({Pct(shown.Count, ok.Count)}) | {Pct(right, shown.Count)} | {kept.Count} ({Pct(kept.Count, ok.Count)}) | {Pct(keptRight, kept.Count)} | " +
                        $"{shown.Count - right - (kept.Count - keptRight)} of {shown.Count - right} | {right - keptRight} of {right} | " +
                        $"{ideal.Count} ({Pct(ideal.Count, ok.Count)}) | {Pct(idealRight, ideal.Count)} | {shown.Count - right - (ideal.Count - idealRight)} of {shown.Count - right} | {right - idealRight} of {right} |");
        }
        w.WriteLine();
        w.WriteLine("Why the generated line fails (positions with a verdict, conf >= 0.7, gen does not resolve):");
        var bad = ok.Where(p => p.Conf >= 0.7 && !B(p, "gen_resolvable")).ToList();
        var byReason = bad.GroupBy(p => !B(p, "gen_first_is_member") ? "simple name unknown" : B(p, "gen_recv_known") ? "member missing on a known receiver" : "receiver type unknown")
            .OrderByDescending(g => g.Count());
        foreach (var g in byReason) w.WriteLine($"- {g.Key}: {g.Count()} (of which line exact: {g.Count(p => p.Exact)}, true line resolves: {g.Count(p => B(p, "true_resolvable"))})");
        w.WriteLine();
        w.WriteLine("By position kind (conf >= 0.7): n / precision / +filter precision / wrong removed / right removed");
        foreach (var g in ok.Where(p => p.Conf >= 0.7).GroupBy(p => p.Kind).OrderByDescending(g => g.Count()))
        {
            var n = g.Count(); var right = g.Count(p => p.Exact); var kept = g.Where(p => B(p, "gen_resolvable")).ToList(); var kr = kept.Count(p => p.Exact);
            w.WriteLine($"- {g.Key}: {n} / {100.0 * right / n:F1} % / {(kept.Count == 0 ? 0 : 100.0 * kr / kept.Count):F1} % / {n - right - (kept.Count - kr)} of {n - right} / {right - kr} of {right}");
        }
        if (withCtx)
        {
            var c = ok.Where(p => p.Out.ContainsKey("ctx_chars")).ToList();
            if (c.Count == 0) return;
            w.WriteLine();
            w.WriteLine($"Context block (file without the middle): mean {c.Average(p => Convert.ToDouble(p.Out["ctx_chars"])):F0} chars, mean {c.Average(p => Convert.ToDouble(p.Out["ctx_lines"])):F1} lines " +
                        $"({c.Average(p => Convert.ToDouble(p.Out["ctx_sigs"])):F1} signatures + {c.Average(p => Convert.ToDouble(p.Out["ctx_members"])):F1} member lists), " +
                        $"dropped entries mean {c.Average(p => Convert.ToDouble(p.Out["ctx_dropped"])):F1}, empty {100.0 * c.Count(p => Convert.ToInt32(p.Out["ctx_chars"]) == 0) / c.Count:F1} %, " +
                        $"build time mean {c.Average(p => Convert.ToDouble(p.Out["ctx_ms"])):F0} ms");
            var ids = c.Sum(p => Convert.ToInt32(p.Out["ids"])); var inP = c.Sum(p => Convert.ToInt32(p.Out["ids_in_prefix"])); var inC = c.Sum(p => Convert.ToInt32(p.Out["ids_in_ctx_only"]));
            var withIds = c.Where(p => Convert.ToInt32(p.Out["ids"]) > 0).ToList();
            w.WriteLine($"identifiers of the true rest: {ids}; in the file prefix {100.0 * inP / ids:F1} %, in the context block but not in the prefix {100.0 * inC / ids:F1} %, nowhere {100.0 * (ids - inP - inC) / ids:F1} %");
            w.WriteLine($"positions with identifiers: {withIds.Count}; all identifiers in prefix: {100.0 * withIds.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) == 0 && Convert.ToInt32(p.Out["ids_in_ctx_only"]) == 0) / withIds.Count:F1} %, " +
                        $"all in prefix ∪ context: {100.0 * withIds.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) == 0) / withIds.Count:F1} %, " +
                        $"context needed (some identifier only in context, none nowhere): {100.0 * withIds.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) == 0 && Convert.ToInt32(p.Out["ids_in_ctx_only"]) > 0) / withIds.Count:F1} %");
            foreach (var kind in new[] { "after-dot", "after-ident", "line-start", "after-open" })
            {
                var k = withIds.Where(p => p.Kind == kind).ToList();
                if (k.Count == 0) continue;
                w.WriteLine($"- {kind}: {k.Count}; all in prefix {100.0 * k.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) == 0 && Convert.ToInt32(p.Out["ids_in_ctx_only"]) == 0) / k.Count:F1} %, " +
                            $"context needed {100.0 * k.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) == 0 && Convert.ToInt32(p.Out["ids_in_ctx_only"]) > 0) / k.Count:F1} %, " +
                            $"nowhere {100.0 * k.Count(p => Convert.ToInt32(p.Out["ids_nowhere"]) > 0) / k.Count:F1} %");
            }
        }
    }

    // ------------------------------------------------------------------------------------------------ context

    static int Context(Dictionary<string, string> o)
    {
        var manifest = o["manifest"]; var reposRoot = o["repos"]; var outDir = o["out"];
        var fold = Opt(o, "fold", "test");
        var threads = OptInt(o, "threads", 4);
        var repoTimeout = OptInt(o, "repo-timeout", 900);
        var maxFiles = OptInt(o, "max-files", 8000);
        var maxRepos = OptInt(o, "max-repos", int.MaxValue);
        var maxPerRepo = OptInt(o, "max-files-per-repo", int.MaxValue);
        var ctxChars = OptInt(o, "ctx-chars", 1200);
        var ctxMembers = OptInt(o, "ctx-members", 40);
        var onlyRepos = o.TryGetValue("repos-list", out var rl) ? new HashSet<string>(File.ReadAllLines(rl).Where(x => x.Length > 0)) : null;

        var files = new Dictionary<string, List<string>>();
        foreach (var line in File.ReadLines(manifest))
        {
            var n = JsonNode.Parse(line)!;
            if (n["fold"]?.GetValue<string>() != fold) continue;
            if (n["status"]?.GetValue<string>() is { } st && st != "ok") continue;
            var repo = n["repo"]!.GetValue<string>();
            if (onlyRepos != null && !onlyRepos.Contains(repo)) continue;
            if (!files.TryGetValue(repo, out var l)) files[repo] = l = new List<string>();
            l.Add(n["path"]!.GetValue<string>());
        }
        var repos = files.OrderBy(kv => kv.Key, StringComparer.Ordinal).Take(maxRepos).ToList();
        Console.Error.WriteLine($"{repos.Count} repos, {repos.Sum(kv => Math.Min(kv.Value.Count, maxPerRepo))} files, fold {fold}");
        var refs = LoadRefs(o);
        Repo.UseImplicitUsings = !o.ContainsKey("no-implicit-usings");
        Directory.CreateDirectory(outDir);
        var t0 = Stopwatch.StartNew();
        var done = 0; var lockObj = new object();
        var stats = new List<(int chars, int lines, int sigs, int members, double ms)>();
        Parallel.ForEach(repos, new ParallelOptions { MaxDegreeOfParallelism = threads }, kv =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(repoTimeout));
            var ct = cts.Token;
            var outFile = Path.Combine(outDir, kv.Key + ".jsonl");
            var local = new List<(int, int, int, int, double)>();
            string status = "ok"; var sw = Stopwatch.StartNew(); int n = 0;
            try
            {
                var repo = Repo.Load(kv.Key, Path.Combine(reposRoot, kv.Key), refs, maxFiles, ct);
                using var w = new StreamWriter(outFile, false, new UTF8Encoding(false));
                foreach (var path in kv.Value.OrderBy(x => x, StringComparer.Ordinal).Take(maxPerRepo))
                {
                    if (!repo.Trees.TryGetValue(path, out var tree)) continue;
                    var s1 = Stopwatch.StartNew();
                    var model = repo.Compilation.GetSemanticModel(tree);
                    var ctx = ContextBuilder.Build(model, tree, ctxChars, ctxMembers, ct);
                    var rec = new Dictionary<string, object?> { ["repo"] = kv.Key, ["path"] = path, ["chars"] = ctx.Text.Length, ["lines"] = ctx.Lines, ["signatures"] = ctx.Signatures,
                                                                 ["member_lists"] = ctx.MemberLists, ["dropped"] = ctx.Dropped, ["ms"] = (int)s1.Elapsed.TotalMilliseconds, ["context"] = ctx.Text };
                    w.WriteLine(JsonSerializer.Serialize(rec, JsonOpts));
                    local.Add((ctx.Text.Length, ctx.Lines, ctx.Signatures, ctx.MemberLists, s1.Elapsed.TotalMilliseconds));
                    n++;
                }
            }
            catch (OperationCanceledException) { status = "timeout"; }
            catch (Exception e) { status = "error: " + e.Message; }
            lock (lockObj)
            {
                done++; stats.AddRange(local);
                Console.Error.WriteLine($"[{done}/{repos.Count} {t0.Elapsed.TotalMinutes:F1} min] {kv.Key}: {n} files in {sw.Elapsed.TotalSeconds:F1} s, {status}");
            }
        });
        if (stats.Count > 0)
            Console.WriteLine($"files: {stats.Count}; context mean {stats.Average(s => s.chars):F0} chars, {stats.Average(s => s.lines):F1} lines ({stats.Average(s => s.sigs):F1} signatures + " +
                              $"{stats.Average(s => s.members):F1} member lists); empty {100.0 * stats.Count(s => s.chars == 0) / stats.Count:F1} %; mean {stats.Average(s => s.ms):F0} ms per file (excl. load)");
        return 0;
    }
    // ------------------------------------------------------------------------------------------------ usings

    /// <summary>Which `using X.Y.Z;` directives do not bind, per repository: the data for extending refpacks/RefPacks.csproj.</summary>
    static int Usings(Dictionary<string, string> o)
    {
        var reposRoot = o["repos"]; var outPath = o["out"];
        var threads = OptInt(o, "threads", 4);
        var repoTimeout = OptInt(o, "repo-timeout", 900);
        var maxFiles = OptInt(o, "max-files", 8000);
        var maxRepos = OptInt(o, "max-repos", int.MaxValue);
        var repos = new SortedSet<string>(StringComparer.Ordinal);
        if (o.TryGetValue("positions", out var positionsPath))
        {
            var rawJson = Regex.Replace(File.ReadAllText(positionsPath), @"(?<=[:,\[\s])-?(NaN|Infinity)(?=[,\]\}\s])", "null");
            foreach (var n in JsonNode.Parse(rawJson)!["modes"]![Opt(o, "mode", "spm")]!["positions"]!.AsArray()) repos.Add(n!["repo"]!.GetValue<string>());
        }
        if (o.TryGetValue("manifest", out var manifest))
        {
            var fold = Opt(o, "fold", "test");
            foreach (var line in File.ReadLines(manifest))
            {
                var n = JsonNode.Parse(line)!;
                if (n["fold"]?.GetValue<string>() == fold) repos.Add(n["repo"]!.GetValue<string>());
            }
        }
        if (o.TryGetValue("repos-list", out var rl)) { var only = new HashSet<string>(File.ReadAllLines(rl).Where(x => x.Length > 0)); repos.RemoveWhere(r => !only.Contains(r)); }
        var list = repos.Take(maxRepos).ToList();
        Console.Error.WriteLine($"{list.Count} repos");
        var refs = LoadRefs(o);
        Repo.UseImplicitUsings = !o.ContainsKey("no-implicit-usings");
        // namespace → (repos, files, first unresolved part)
        var agg = new Dictionary<string, (HashSet<string> repos, int files)>(StringComparer.Ordinal);
        var lockObj = new object(); var done = 0; var t0 = Stopwatch.StartNew();
        Parallel.ForEach(list, new ParallelOptions { MaxDegreeOfParallelism = threads }, name =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(repoTimeout));
            var ct = cts.Token; var local = new Dictionary<string, int>(StringComparer.Ordinal); string status = "ok";
            try
            {
                var root = Path.Combine(reposRoot, name);
                if (!Directory.Exists(root)) { status = "no-repo"; }
                else
                {
                    var repo = Repo.Load(name, root, refs, maxFiles, ct);
                    foreach (var tree in repo.Trees.Values)
                    {
                        ct.ThrowIfCancellationRequested();
                        SemanticModel? model = null;
                        foreach (var u in tree.GetRoot(ct).DescendantNodes(n => n is Microsoft.CodeAnalysis.CSharp.Syntax.CompilationUnitSyntax or Microsoft.CodeAnalysis.CSharp.Syntax.BaseNamespaceDeclarationSyntax)
                                              .OfType<Microsoft.CodeAnalysis.CSharp.Syntax.UsingDirectiveSyntax>())
                        {
                            if (u.Name == null) continue;
                            model ??= repo.Compilation.GetSemanticModel(tree);
                            var si = model.GetSymbolInfo(u.Name, ct);
                            if (si.Symbol != null || !si.CandidateSymbols.IsDefaultOrEmpty) continue;
                            var ns = u.Name.ToString();
                            local[ns] = local.GetValueOrDefault(ns) + 1;
                        }
                    }
                }
            }
            catch (OperationCanceledException) { status = "timeout"; }
            catch (Exception e) { status = "error: " + e.Message; }
            lock (lockObj)
            {
                done++;
                foreach (var kv in local)
                {
                    if (!agg.TryGetValue(kv.Key, out var a)) agg[kv.Key] = a = (new HashSet<string>(), 0);
                    a.repos.Add(name); agg[kv.Key] = (a.repos, a.files + kv.Value);
                }
                Console.Error.WriteLine($"[{done}/{list.Count} {t0.Elapsed.TotalMinutes:F1} min] {name}: {local.Count} unresolved namespaces, {status}");
            }
        });
        using var w = new StreamWriter(outPath, false, new UTF8Encoding(false));
        w.WriteLine("namespace\trepos\tfiles\trepo_names");
        foreach (var kv in agg.OrderByDescending(kv => kv.Value.repos.Count).ThenByDescending(kv => kv.Value.files))
            w.WriteLine($"{kv.Key}\t{kv.Value.repos.Count}\t{kv.Value.files}\t{string.Join(",", kv.Value.repos.OrderBy(x => x).Take(5))}");
        Console.WriteLine($"{agg.Count} unresolved namespaces in {agg.Values.SelectMany(v => v.repos).Distinct().Count()} of {list.Count} repos; top 20:");
        foreach (var kv in agg.OrderByDescending(kv => kv.Value.repos.Count).Take(20)) Console.WriteLine($"  {kv.Key}: {kv.Value.repos.Count} repos, {kv.Value.files} files");
        return 0;
    }
}
