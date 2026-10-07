using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace CmlRoslyn;

/// <summary>Unit-style check of receiver-type detection (`recv.Member`) on a synthetic file: locals, parameters, fields,
/// properties, `this`, static classes, method-call results, chained accesses, `?.`, extension methods, generics.
/// Every line with a `// expect: A, B` comment must yield exactly those receiver types for its member accesses, in order.</summary>
public static class SelfTest
{
    const string Source = """
        using System;
        using System.IO;
        using System.Collections.Generic;
        using System.Linq;
        using Xunit;
        namespace T;
        class Other
        {
            public string Name { get; set; } = "";
            public List<int> Items = new();
            public Other Next() => this;
            public static Other Make() => new();
            public Dictionary<string, Other> Map = new();
        }
        static class Ext { public static int Twice(this int x) => x * 2; }
        class C<TItem> where TItem : Other
        {
            Other fld = new();
            Other Prop { get; set; } = new();
            static Other Stat = new();
            void M(Other param, int n, TItem item, IEnumerable<Other> seq)
            {
                var local = new Other();
                _ = local.Name.Length;                 // expect: Other, string
                _ = param.Items.Count;                 // expect: Other, List<int>
                _ = fld.Next().Name;                   // expect: Other, Other
                _ = Prop.Next().Next().Items;          // expect: Other, Other, Other
                _ = this.fld.Name;                     // expect: C<TItem>, Other
                _ = Stat.Name;                         // expect: Other
                _ = string.Join(",", local.Items);     // expect: string, Other
                _ = Path.Combine("a", "b");            // expect: Path
                _ = Console.Out;                       // expect: Console
                Assert.Equal(1, n);                    // expect: Assert
                _ = Other.Make().Name;                 // expect: Other, Other
                _ = local?.Name;                       // expect: Other
                _ = local.Items.Select(x => x).First(); // expect: Other, List<int>, IEnumerable<int>
                _ = n.Twice().ToString();              // expect: int, int
                _ = item.Name;                         // expect: TItem
                _ = seq.First().Map["k"].Next();       // expect: IEnumerable<Other>, Other, Other
                _ = Environment.NewLine.Trim();        // expect: Environment, string
                _ = local.Nope;                        // expect: Other
                _ = nope.Name;                         // expect: ?
                _ = DateTime.Now.Year;                 // expect: DateTime, DateTime
            }
        }
        """;

    public static int Run(Dictionary<string, string> o)
    {
        var refs = Program.LoadRefs(o);
        var tree = CSharpSyntaxTree.ParseText(Source, Repo.ParseOptions, path: "selftest.cs");
        var comp = CSharpCompilation.Create("selftest", new[] { tree }, refs, new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var model = comp.GetSemanticModel(tree);
        var text = tree.GetText();
        int fail = 0, total = 0;
        foreach (var line in text.Lines)
        {
            var s = line.ToString();
            var i = s.IndexOf("// expect:", StringComparison.Ordinal);
            if (i < 0) continue;
            total++;
            var expected = s[(i + 10)..].Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            var v = Judge.JudgeSpan(model, tree.GetRoot(), line.Start, line.Start + i, CancellationToken.None);
            var got = v.Trace.Select(t => t.recv ?? "?").ToArray();
            var ok = got.SequenceEqual(expected);
            if (!ok) fail++;
            Console.WriteLine($"{(ok ? "ok  " : "FAIL")} {s[..i].Trim(),-45} expected [{string.Join(", ", expected)}] got [{string.Join(", ", got)}]" +
                              (v.Unresolved > 0 ? $" unresolved: {v.FirstUnresolved}" : ""));
        }
        Console.WriteLine($"selftest: {total - fail}/{total} lines ok");
        return fail == 0 ? 0 : 1;
    }
}
