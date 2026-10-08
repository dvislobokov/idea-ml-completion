"""Lightweight regex parsers for the import statistics (e20): Go import blocks + `pkg.Ident` references,
C# `using` directives + PascalCase type-like identifiers + type declarations. Shared by the miner and the evaluation.
No real parser on purpose: we need file-level sets, not exact ASTs, and the corpora are tens of GB."""
import re

# ------------------------------------------------------------------------------------------------------ Go

_GO_IMPORT_BLOCK = re.compile(r'^import\s*\(([^)]*)\)', re.M)
_GO_IMPORT_ONE = re.compile(r'^import\s+(?:([\w.]+)\s+)?"([^"\n]+)"', re.M)
_GO_IMPORT_LINE = re.compile(r'(?:^|\n)\s*(?:([\w.]+)\s+)?"([^"\n]+)"')
_GO_STRIP = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\\n])*"|`[^`]*`|\'(?:\\.|[^\'\\\n])*\'')
_GO_REF = re.compile(r'(?<![\w.])([a-z][A-Za-z0-9_]*)\s*\.\s*([A-Z][A-Za-z0-9_]*)')
_GO_VERSION = re.compile(r'^v\d+$')
_GO_DOTVERSION = re.compile(r'\.v\d+$')


def go_package_names(path):
    """Candidate package names for an import path, most likely first (`net/http` -> http, `gopkg.in/yaml.v3` -> yaml,
    `github.com/mattn/go-sqlite3` -> sqlite3, `github.com/x/y/v2` -> y)."""
    parts = path.split('/')
    last = parts[-1]
    if _GO_VERSION.match(last) and len(parts) > 1:
        last = parts[-2]
    cands = [last]
    s = _GO_DOTVERSION.sub('', last)
    if s != last:
        cands.append(s)
    for c in list(cands):
        if c.startswith('go-'):
            cands.append(c[3:])
        if c.endswith('-go'):
            cands.append(c[:-3])
        if c.endswith('.go'):
            cands.append(c[:-3])
    out = []
    for c in cands:
        for v in (c, c.replace('-', '_'), c.replace('-', ''), c.replace('.', '_'), c.lower(), c.replace('-', '').lower()):
            if v and v not in out:
                out.append(v)
    return out or [path]


def parse_go(text, local_prefix=None):
    """Returns (imports: set of import paths, refs: set of (ident, path), qualifiers: set of (pkg-as-written, path)).
    `local_prefix` (lower-case `github.com/owner/repo`) drops the repository's own packages."""
    imports = {}  # path -> alias or None
    head = text[:200_000]
    for m in _GO_IMPORT_BLOCK.finditer(head):
        for lm in _GO_IMPORT_LINE.finditer(m.group(1)):
            imports[lm.group(2)] = lm.group(1)
    for m in _GO_IMPORT_ONE.finditer(head):
        imports[m.group(2)] = m.group(1)
    imports = {p: a for p, a in imports.items() if p.strip('/') and p.isascii() and len(p) < 200}
    if local_prefix:
        imports = {p: a for p, a in imports.items() if not p.lower().startswith(local_prefix)}
    if not imports:
        return set(), set(), set()
    # name -> path: aliases first, then the exact last element, then the fallbacks (only if the name is still free)
    names = {}
    for p, a in imports.items():
        if a and a not in ('_', '.'):
            names[a] = p
    for p, a in imports.items():
        if a:
            continue
        c = go_package_names(p)[0]
        names.setdefault(c, p)
    for p, a in imports.items():
        if a:
            continue
        for c in go_package_names(p)[1:]:
            names.setdefault(c, p)
    code = _GO_STRIP.sub(' ', text)
    refs = set()
    quals = set()
    for m in _GO_REF.finditer(code):
        p = names.get(m.group(1))
        if p is None:
            continue
        refs.add((m.group(2), p))
        quals.add((m.group(1), p))
    return set(imports), refs, quals


# ------------------------------------------------------------------------------------------------------ C#

_CS_USING = re.compile(r'^\s*(?:global\s+)?using\s+(static\s+)?(?:(\w+)\s*=\s*)?([A-Za-z_][\w.]*)\s*;', re.M)
_CS_NAMESPACE = re.compile(r'^\s*namespace\s+([A-Za-z_][\w.]*)', re.M)
_CS_DECL = re.compile(r'\b(?:class|struct|interface|enum|record)\s+([A-Z]\w*)')
_CS_STRIP = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|\$?@"(?:[^"]|"")*"|\$"(?:\\.|[^"\\\n])*"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_CS_IDENT = re.compile(r'(?<![\w.@])([A-Z][A-Za-z0-9_]*)')
_CS_NEW = re.compile(r'new\s+$')


def parse_cs(text, local_namespaces=None):
    """Returns (usings: set, idents: set of type-like identifiers used but not declared here, decls: set of (name, namespace)).
    `local_namespaces` (the repository's own declared namespaces) are dropped from `usings`."""
    head = text[:200_000]
    usings = set()
    for m in _CS_USING.finditer(head):
        if m.group(1) or m.group(2):
            continue  # `using static`, aliases
        u = m.group(3)
        if u.isascii() and len(u) < 200:
            usings.add(u)
    if local_namespaces:
        usings = {u for u in usings if u not in local_namespaces}
    code = _CS_STRIP.sub(' ', text)
    nm = _CS_NAMESPACE.search(code)
    ns = nm.group(1) if nm else ''
    decls = set()
    local = set()
    for m in _CS_DECL.finditer(code):
        local.add(m.group(1))
        if ns:
            decls.add((m.group(1), ns))
    idents = set()
    for m in _CS_IDENT.finditer(code):
        name = m.group(1)
        if name in local:
            continue
        end = m.end()
        nxt = code[end:end + 1]
        if nxt == '(' and not _CS_NEW.search(code[max(0, m.start() - 8):m.start()]):
            continue  # a method call, not a type
        idents.add(name)
        # [Fact] -> FactAttribute
        before = code[max(0, m.start() - 2):m.start()].rstrip()
        if before.endswith('[') or before.endswith(','):
            idents.add(name + 'Attribute')
    return usings, idents, decls
