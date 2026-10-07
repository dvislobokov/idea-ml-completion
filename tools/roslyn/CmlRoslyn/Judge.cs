using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace CmlRoslyn;

/// <summary>Verdict for one line of code spliced into a file: does every simple name / member name in the span resolve?</summary>
public sealed class Verdict
{
    public int Names;                 // simple names examined in the span
    public int Unresolved;            // of which unresolved
    public int MemberAccesses;        // names that are the member part of `recv.Name`
    public int MemberRecvKnown;       // member accesses whose receiver type/namespace is known
    public string? FirstUnresolved;   // first unresolved name
    public bool FirstIsMember;        // it is the member part of a member access
    public bool FirstRecvKnown;       // its receiver type/namespace is known (then the member really does not exist on it)
    public string? FirstRecvType;     // display of the receiver type
    public int FirstCandidates;       // number of distinct member names on the receiver (0 when unknown)
    public bool Resolvable => Unresolved == 0;
    public List<(string name, string? recv, bool resolved)> Trace = new();   // every member access in the span: name, receiver type display (null = unknown), resolved?
}

public static class Judge
{
    static readonly HashSet<string> Special = new(StringComparer.Ordinal) { "var", "dynamic", "nameof", "global", "_", "value", "await", "async", "field", "args", "scoped", "notnull", "unmanaged" };

    /// <summary>Judge every <see cref="SimpleNameSyntax"/> whose span intersects [start, end).</summary>
    public static Verdict JudgeSpan(SemanticModel model, SyntaxNode root, int start, int end, CancellationToken ct)
    {
        var v = new Verdict();
        if (end <= start) return v;
        var span = new Microsoft.CodeAnalysis.Text.TextSpan(start, end - start);
        var names = root.DescendantNodes(span, descendIntoTrivia: false).OfType<SimpleNameSyntax>()
            .Where(n => n.Span.End > start && n.Span.Start < end).OrderBy(n => n.SpanStart).ToList();
        foreach (var name in names)
        {
            ct.ThrowIfCancellationRequested();
            if (name.IsVar) { v.Names++; continue; }
            var text = name.Identifier.ValueText;
            ExpressionSyntax? receiver = ReceiverOf(name);
            v.Names++;
            if (Special.Contains(text) && receiver == null) continue;
            var isMember = receiver != null || name.Parent is MemberBindingExpressionSyntax;
            if (isMember) v.MemberAccesses++;
            bool recvKnown = false; string? recvType = null; int candidates = 0;
            if (isMember)
            {
                var recvSym = ReceiverSymbol(model, name, receiver, ct);
                if (recvSym != null)
                {
                    recvKnown = true; v.MemberRecvKnown++;
                    recvType = recvSym.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);
                    candidates = CountMembers(recvSym);
                }
            }
            var info = model.GetSymbolInfo(name, ct);
            var resolved = info.Symbol != null || !info.CandidateSymbols.IsDefaultOrEmpty;
            if (!resolved)
            {
                // attributes `[Foo]` bind as FooAttribute; labels, preprocessor and alias names: fall back on type info
                var ti = model.GetTypeInfo(name, ct);
                if (ti.Type != null && ti.Type.TypeKind != TypeKind.Error) resolved = true;
                else if (name.Parent is AttributeSyntax) { var s = model.GetSymbolInfo(name.Parent, ct); resolved = s.Symbol != null || !s.CandidateSymbols.IsDefaultOrEmpty; }
                else if (name.Parent is GotoStatementSyntax or LabeledStatementSyntax) resolved = true;
            }
            if (isMember) v.Trace.Add((text, recvType, resolved));
            if (!resolved)
            {
                v.Unresolved++;
                if (v.FirstUnresolved == null)
                {
                    v.FirstUnresolved = text; v.FirstIsMember = isMember; v.FirstRecvKnown = recvKnown; v.FirstRecvType = recvType; v.FirstCandidates = candidates;
                }
            }
        }
        return v;
    }

    static ExpressionSyntax? ReceiverOf(SimpleNameSyntax name) => name.Parent switch
    {
        MemberAccessExpressionSyntax ma when ma.Name == name => ma.Expression,
        QualifiedNameSyntax qn when qn.Right == name => qn.Left,
        AliasQualifiedNameSyntax aq when aq.Name == name => aq.Alias,
        _ => null,
    };

    public static INamespaceOrTypeSymbol? ReceiverSymbolOf(SemanticModel model, SimpleNameSyntax name, CancellationToken ct) => ReceiverSymbol(model, name, ReceiverOf(name), ct);

    static INamespaceOrTypeSymbol? ReceiverSymbol(SemanticModel model, SimpleNameSyntax name, ExpressionSyntax? receiver, CancellationToken ct)
    {
        if (receiver == null)
        {
            // `?.Name`: the conditional access expression gives the receiver
            var cond = name.Parent?.Parent;
            while (cond != null && cond is not ConditionalAccessExpressionSyntax) cond = cond.Parent;
            if (cond is ConditionalAccessExpressionSyntax cae) receiver = cae.Expression; else return null;
        }
        var ti = model.GetTypeInfo(receiver, ct);
        if (ti.Type != null && ti.Type.TypeKind != TypeKind.Error) return ti.Type;
        var si = model.GetSymbolInfo(receiver, ct);
        var sym = si.Symbol ?? (si.CandidateSymbols.IsDefaultOrEmpty ? null : si.CandidateSymbols[0]);
        return sym switch
        {
            INamespaceOrTypeSymbol nt when nt is not ITypeSymbol { TypeKind: TypeKind.Error } => nt,
            ILocalSymbol l when l.Type.TypeKind != TypeKind.Error => l.Type,
            IParameterSymbol p when p.Type.TypeKind != TypeKind.Error => p.Type,
            IFieldSymbol f when f.Type.TypeKind != TypeKind.Error => f.Type,
            IPropertySymbol pr when pr.Type.TypeKind != TypeKind.Error => pr.Type,
            _ => null,
        };
    }

    /// <summary>Distinct member names visible on a type (including base types and interfaces for interfaces) or namespace.</summary>
    public static int CountMembers(INamespaceOrTypeSymbol s) => MemberNames(s).Count;

    public static HashSet<string> MemberNames(INamespaceOrTypeSymbol s)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        if (s is INamespaceSymbol ns)
        {
            foreach (var m in ns.GetMembers()) names.Add(m.Name);
            return names;
        }
        var t = (ITypeSymbol)s;
        if (t is ITypeParameterSymbol tp)
        {
            foreach (var c in tp.ConstraintTypes) foreach (var n in MemberNames(c)) names.Add(n);
            names.UnionWith(ObjectMembers);
            return names;
        }
        for (var cur = t; cur != null; cur = cur.BaseType)
            foreach (var m in cur.GetMembers()) if (IsListable(m)) names.Add(m.Name);
        if (t.TypeKind == TypeKind.Interface)
        {
            foreach (var i in t.AllInterfaces) foreach (var m in i.GetMembers()) if (IsListable(m)) names.Add(m.Name);
            names.UnionWith(ObjectMembers);
        }
        return names;
    }

    static readonly string[] ObjectMembers = { "ToString", "Equals", "GetHashCode", "GetType" };

    public static bool IsListable(ISymbol m)
    {
        if (m.Name.Length == 0 || m.Name[0] == '.' || m.Name[0] == '<') return false;
        if (m is IMethodSymbol ms && ms.MethodKind != MethodKind.Ordinary && ms.MethodKind != MethodKind.UserDefinedOperator) return false;
        return m.DeclaredAccessibility != Accessibility.Private || Repo.IsSource(m);
    }
}
