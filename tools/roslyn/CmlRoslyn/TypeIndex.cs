using System.Collections.Concurrent;
using Microsoft.CodeAnalysis;

namespace CmlRoslyn;

/// <summary>What an IDE with the real project would still know when a name does not bind in our project-less compilation:
/// types reachable through a `using` that lives in the lost .csproj (`&lt;Using Include=…&gt;`, ImplicitUsings of other SDKs),
/// looked up by simple name over every referenced assembly plus the source of the repository; and whether an extension
/// method with a given name exists anywhere (missing `using` for the extension namespace). Metadata type names are
/// compilation-independent and cached once per process; source symbols come from <see cref="Compilation.GetSymbolsWithName"/>.</summary>
public sealed class TypeIndex
{
    static readonly object InitLock = new();
    static Dictionary<string, List<string>>? MetaTypesByName;       // simple name → metadata names (top-level public types, with `n arity suffix)
    static HashSet<string>? MetaNamespaces;                         // every namespace segment seen in metadata
    static readonly ConcurrentDictionary<string, bool> ExtMethodCache = new(StringComparer.Ordinal);
    static IReadOnlyList<MetadataReference>? IndexedRefs;

    readonly Compilation comp;
    readonly ConcurrentDictionary<(string, int), INamedTypeSymbol[]> cache = new();

    public TypeIndex(Compilation comp)
    {
        this.comp = comp;
        lock (InitLock)
        {
            if (MetaTypesByName != null) return;
            var types = new Dictionary<string, List<string>>(StringComparer.Ordinal);
            var namespaces = new HashSet<string>(StringComparer.Ordinal);
            foreach (var r in comp.References)
            {
                if (comp.GetAssemblyOrModuleSymbol(r) is not IAssemblySymbol asm) continue;
                Walk(asm.GlobalNamespace, types, namespaces);
            }
            MetaTypesByName = types; MetaNamespaces = namespaces; IndexedRefs = comp.References.ToList();
        }
    }

    static void Walk(INamespaceSymbol ns, Dictionary<string, List<string>> types, HashSet<string> namespaces)
    {
        foreach (var m in ns.GetMembers())
        {
            if (m is INamespaceSymbol child) { namespaces.Add(child.Name); Walk(child, types, namespaces); }
            else if (m is INamedTypeSymbol t && t.DeclaredAccessibility == Accessibility.Public && t.CanBeReferencedByName)
            {
                if (!types.TryGetValue(t.Name, out var l)) types[t.Name] = l = new List<string>();
                var md = t.ContainingNamespace.IsGlobalNamespace ? t.MetadataName : t.ContainingNamespace.ToDisplayString() + "." + t.MetadataName;
                if (!l.Contains(md)) l.Add(md);
            }
        }
    }

    public bool IsNamespaceSegment(string name) => MetaNamespaces!.Contains(name) || comp.GetSymbolsWithName(name, SymbolFilter.Namespace).Any();

    /// <summary>Types with this simple name and arity, metadata (public) and source, in no particular order; empty when none.</summary>
    public INamedTypeSymbol[] Lookup(string name, int arity)
    {
        return cache.GetOrAdd((name, arity), k =>
        {
            var res = new List<INamedTypeSymbol>();
            if (MetaTypesByName!.TryGetValue(k.Item1, out var mds))
                foreach (var md in mds)
                {
                    if ((k.Item2 == 0 && md.Contains('`')) || (k.Item2 > 0 && !md.EndsWith("`" + k.Item2, StringComparison.Ordinal))) continue;
                    res.AddRange(comp.GetTypesByMetadataName(md));
                }
            foreach (var s in comp.GetSymbolsWithName(k.Item1, SymbolFilter.Type))
                if (s is INamedTypeSymbol t && t.Arity == k.Item2 && !res.Contains(t, SymbolEqualityComparer.Default)) res.Add(t);
            return res.ToArray();
        });
    }

    /// <summary>Does an extension method with this name exist in any referenced assembly or in source?</summary>
    readonly ConcurrentDictionary<string, bool> srcExtCache = new(StringComparer.Ordinal);

    public bool HasExtensionMethod(string name)
    {
        if (srcExtCache.GetOrAdd(name, n => comp.GetSymbolsWithName(n, SymbolFilter.Member).Any(s => s is IMethodSymbol { IsExtensionMethod: true }))) return true;
        return ExtMethodCache.GetOrAdd(name, n =>
        {
            foreach (var r in IndexedRefs!)
            {
                if (comp.GetAssemblyOrModuleSymbol(r) is not IAssemblySymbol asm || !asm.MightContainExtensionMethods) continue;
                if (HasExt(asm.GlobalNamespace, n)) return true;
            }
            return false;
        });
    }

    static bool HasExt(INamespaceSymbol ns, string name)
    {
        foreach (var m in ns.GetMembers())
        {
            if (m is INamespaceSymbol child) { if (HasExt(child, name)) return true; }
            else if (m is INamedTypeSymbol { IsStatic: true, MightContainExtensionMethods: true } t)
                foreach (var mm in t.GetMembers(name)) if (mm is IMethodSymbol { IsExtensionMethod: true, DeclaredAccessibility: Accessibility.Public }) return true;
        }
        return false;
    }
}
