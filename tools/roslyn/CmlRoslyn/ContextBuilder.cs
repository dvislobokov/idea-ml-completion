using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace CmlRoslyn;

/// <summary>Compact project context for one file: one-line signatures of the source symbols from OTHER files that the
/// file references, and member lists of the solution types used as receivers in `recv.Member`. Plain text, one
/// declaration per line, entries in order of first reference in the file, capped at <c>maxChars</c>.</summary>
public static class ContextBuilder
{
    static readonly SymbolDisplayFormat MemberFormat = new(
        globalNamespaceStyle: SymbolDisplayGlobalNamespaceStyle.Omitted,
        typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameOnly,
        genericsOptions: SymbolDisplayGenericsOptions.IncludeTypeParameters,
        memberOptions: SymbolDisplayMemberOptions.IncludeParameters | SymbolDisplayMemberOptions.IncludeContainingType | SymbolDisplayMemberOptions.IncludeType
                       | SymbolDisplayMemberOptions.IncludeModifiers | SymbolDisplayMemberOptions.IncludeConstantValue,
        parameterOptions: SymbolDisplayParameterOptions.IncludeType | SymbolDisplayParameterOptions.IncludeName | SymbolDisplayParameterOptions.IncludeParamsRefOut
                          | SymbolDisplayParameterOptions.IncludeDefaultValue,
        propertyStyle: SymbolDisplayPropertyStyle.ShowReadWriteDescriptor,
        miscellaneousOptions: SymbolDisplayMiscellaneousOptions.UseSpecialTypes | SymbolDisplayMiscellaneousOptions.EscapeKeywordIdentifiers);

    static readonly SymbolDisplayFormat TypeFormat = new(
        globalNamespaceStyle: SymbolDisplayGlobalNamespaceStyle.Omitted,
        typeQualificationStyle: SymbolDisplayTypeQualificationStyle.NameOnly,
        genericsOptions: SymbolDisplayGenericsOptions.IncludeTypeParameters,
        miscellaneousOptions: SymbolDisplayMiscellaneousOptions.UseSpecialTypes);

    public sealed class Result
    {
        public string Text = "";
        public int Lines;
        public int Signatures;
        public int MemberLists;
        public int Dropped;          // entries that did not fit into the budget
    }

    public static Result Build(SemanticModel model, SyntaxTree tree, int maxChars, int maxMembersPerType, CancellationToken ct)
    {
        var root = tree.GetRoot(ct);
        var seen = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
        var seenTypes = new HashSet<ISymbol>(SymbolEqualityComparer.Default);
        var entries = new List<(int pos, string line, bool members)>();

        foreach (var name in root.DescendantNodes(descendIntoTrivia: false).OfType<SimpleNameSyntax>())
        {
            ct.ThrowIfCancellationRequested();
            if (name.IsVar) continue;
            var info = model.GetSymbolInfo(name, ct);
            var sym = info.Symbol ?? (info.CandidateSymbols.IsDefaultOrEmpty ? null : info.CandidateSymbols[0]);
            if (sym is IMethodSymbol { ReducedFrom: { } rf }) sym = rf;
            sym = sym?.OriginalDefinition;
            if (sym is INamedTypeSymbol or IMethodSymbol or IPropertySymbol or IFieldSymbol or IEventSymbol)
            {
                if (sym is IMethodSymbol { MethodKind: MethodKind.LocalFunction }) sym = null;
                else if (!Repo.IsSource(sym) || DeclaredOnlyIn(sym, tree)) sym = null;
            }
            else sym = null;
            if (sym != null && seen.Add(sym)) entries.Add((name.SpanStart, Signature(sym), false));

            // member list of the receiver type of `recv.Name`
            if (name.Parent is MemberAccessExpressionSyntax ma && ma.Name == name || name.Parent is MemberBindingExpressionSyntax)
            {
                var recv = Judge.ReceiverSymbolOf(model, name, ct);
                if (recv is INamedTypeSymbol t && t.TypeKind != TypeKind.Error)
                {
                    t = t.OriginalDefinition;
                    if (Repo.IsSource(t) && !DeclaredOnlyIn(t, tree) && seenTypes.Add(t))
                    {
                        var line = MemberLine(t, tree, maxMembersPerType);
                        if (line != null) entries.Add((name.SpanStart, line, true));
                    }
                }
            }
        }

        var r = new Result();
        var sb = new StringBuilder();
        foreach (var (_, line, members) in entries)
        {
            if (sb.Length + line.Length + 1 > maxChars) { r.Dropped++; continue; }
            sb.Append(line).Append('\n');
            r.Lines++;
            if (members) r.MemberLists++; else r.Signatures++;
        }
        r.Text = sb.ToString();
        return r;
    }

    static bool DeclaredOnlyIn(ISymbol s, SyntaxTree tree) =>
        s.DeclaringSyntaxReferences.Length > 0 && s.DeclaringSyntaxReferences.All(d => d.SyntaxTree == tree);

    public static string Signature(ISymbol sym)
    {
        switch (sym)
        {
            case INamedTypeSymbol t:
            {
                var kw = t.TypeKind switch
                {
                    TypeKind.Interface => "interface", TypeKind.Enum => "enum", TypeKind.Struct => t.IsRecord ? "record struct" : "struct",
                    TypeKind.Delegate => "delegate", _ => t.IsRecord ? "record" : (t.IsStatic ? "static class" : t.IsAbstract ? "abstract class" : "class"),
                };
                var sb = new StringBuilder();
                sb.Append(kw).Append(' ');
                if (t.ContainingType != null) sb.Append(t.ContainingType.ToDisplayString(TypeFormat)).Append('.');
                sb.Append(t.ToDisplayString(TypeFormat));
                if (t.TypeKind == TypeKind.Delegate && t.DelegateInvokeMethod != null)
                    sb.Append('(').Append(string.Join(", ", t.DelegateInvokeMethod.Parameters.Select(p => p.ToDisplayString(MemberFormat)))).Append(") : ")
                      .Append(t.DelegateInvokeMethod.ReturnType.ToDisplayString(TypeFormat));
                else
                {
                    var bases = new List<string>();
                    if (t.BaseType != null && t.BaseType.SpecialType != SpecialType.System_Object && t.BaseType.SpecialType != SpecialType.System_ValueType
                        && t.BaseType.SpecialType != SpecialType.System_Enum) bases.Add(t.BaseType.ToDisplayString(TypeFormat));
                    bases.AddRange(t.Interfaces.Select(i => i.ToDisplayString(TypeFormat)));
                    if (bases.Count > 0) sb.Append(" : ").Append(string.Join(", ", bases));
                }
                return sb.ToString();
            }
            default:
                return sym.ToDisplayString(MemberFormat);
        }
    }

    /// <summary>`members Foo: Bar() Baz Qux …` — own and inherited source members declared outside <paramref name="tree"/>;
    /// methods carry `()`, sorted by name, capped.</summary>
    static string? MemberLine(INamedTypeSymbol t, SyntaxTree tree, int cap)
    {
        var names = new SortedDictionary<string, string>(StringComparer.Ordinal);
        for (var cur = (INamedTypeSymbol?)t; cur != null; cur = cur.BaseType)
        {
            if (!Repo.IsSource(cur)) break;
            foreach (var m in cur.GetMembers())
            {
                if (!Judge.IsListable(m) || m.DeclaredAccessibility == Accessibility.Private) continue;
                if (m.DeclaringSyntaxReferences.Length > 0 && m.DeclaringSyntaxReferences.All(d => d.SyntaxTree == tree)) continue;
                if (m.IsImplicitlyDeclared) continue;
                names.TryAdd(m.Name, m is IMethodSymbol ? m.Name + "()" : m is INamedTypeSymbol ? m.Name + "{}" : m.Name);
            }
        }
        if (names.Count == 0) return null;
        var list = names.Values.Take(cap).ToList();
        var more = names.Count > cap ? $" …+{names.Count - cap}" : "";
        return $"members {t.ToDisplayString(TypeFormat)}: {string.Join(" ", list)}{more}";
    }
}
