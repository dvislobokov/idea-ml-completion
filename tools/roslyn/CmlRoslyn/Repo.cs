using System.Diagnostics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace CmlRoslyn;

/// <summary>One repository snapshot loaded as a single <see cref="CSharpCompilation"/> over all its <c>*.cs</c> files
/// (the corpus has no project files, so every file of every project goes into one compilation; duplicate type names
/// between projects show up as ambiguities, which the filter still counts as "resolves").</summary>
public sealed class Repo
{
    public static readonly CSharpParseOptions ParseOptions = new CSharpParseOptions(LanguageVersion.Preview, DocumentationMode.None,
        preprocessorSymbols: new[] { "DEBUG", "TRACE", "NET", "NETCOREAPP", "NET10_0", "NET10_0_OR_GREATER", "NET9_0_OR_GREATER", "NET8_0_OR_GREATER",
                                     "NET7_0_OR_GREATER", "NET6_0_OR_GREATER", "NET5_0_OR_GREATER", "NETCOREAPP3_1_OR_GREATER", "NETCOREAPP3_0_OR_GREATER" });

    public string Name { get; }
    public string Root { get; }
    public CSharpCompilation Compilation { get; }
    public Dictionary<string, SyntaxTree> Trees { get; }   // relative path (forward slashes) → tree
    public int FileCount => Trees.Count;
    public double LoadSeconds { get; }
    public bool Truncated { get; }

    private Repo(string name, string root, CSharpCompilation c, Dictionary<string, SyntaxTree> trees, double secs, bool truncated)
    { Name = name; Root = root; Compilation = c; Trees = trees; LoadSeconds = secs; Truncated = truncated; }

    /// <summary>What `ImplicitUsings=enable` of Microsoft.NET.Sdk generates (the corpus has no project files). The Web SDK set
    /// (Microsoft.AspNetCore.*, Microsoft.Extensions.*) is deliberately NOT added: it creates ambiguities (`IResult`, `Timer`, `ILogger`)
    /// that cost more resolutions than they gain.</summary>
    public const string ImplicitUsings = """
        global using System;
        global using System.Collections.Generic;
        global using System.IO;
        global using System.Linq;
        global using System.Net.Http;
        global using System.Threading;
        global using System.Threading.Tasks;
        """;

    public static bool UseImplicitUsings = true;

    public static Repo Load(string name, string root, IReadOnlyList<MetadataReference> refs, int maxFiles, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        var files = Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories)
            .Select(f => Path.GetRelativePath(root, f).Replace('\\', '/'))
            .Where(f => !f.Contains("/obj/") && !f.Contains("/bin/") && !f.StartsWith("obj/") && !f.StartsWith("bin/"))
            .OrderBy(f => f, StringComparer.Ordinal).ToList();
        var truncated = files.Count > maxFiles;
        if (truncated) files = files.Take(maxFiles).ToList();
        var trees = new Dictionary<string, SyntaxTree>(files.Count, StringComparer.Ordinal);
        foreach (var rel in files)
        {
            ct.ThrowIfCancellationRequested();
            var text = SourceText.From(File.ReadAllText(Path.Combine(root, rel)), System.Text.Encoding.UTF8);
            trees[rel] = CSharpSyntaxTree.ParseText(text, ParseOptions, path: rel, cancellationToken: ct);
        }
        var options = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true, nullableContextOptions: NullableContextOptions.Disable,
            concurrentBuild: false, reportSuppressedDiagnostics: false, warningLevel: 0);
        var allTrees = trees.Values.ToList();
        if (UseImplicitUsings) allTrees.Add(CSharpSyntaxTree.ParseText(ImplicitUsings, ParseOptions, path: "<implicit-usings>.cs"));
        var comp = CSharpCompilation.Create(Sanitize(name), allTrees, refs, options);
        return new Repo(name, root, comp, trees, sw.Elapsed.TotalSeconds, truncated);
    }

    static string Sanitize(string s) => new string(s.Select(ch => char.IsLetterOrDigit(ch) || ch == '.' || ch == '_' ? ch : '_').ToArray());

    /// <summary>Is the symbol declared in source of this compilation (as opposed to metadata)?</summary>
    public static bool IsSource(ISymbol s) => s.Locations.Any(l => l.IsInSource);
}
