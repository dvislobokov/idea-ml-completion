"""Python reference implementation of the `imports` .cml artifact (kind "imports" inside the CML1 gzip container, see
io.github.completionml.core.imports.ImportsModel): writer, reader and the two queries. The Kotlin reader must agree with this
file bit for bit on the quantised scores (parity fixture produced by eval_imports.py)."""
import gzip, math, struct, time

MAGIC = b'CML1'
VERSION = 2
KIND = 'imports'
LOGP_SCALE = 16.0   # q = round(-ln p * 16), 0..255  (p down to e^-16)
PMI_SCALE = 16.0    # q = round(pmi * 16), -127..127


class _Out:
    def __init__(self):
        self.parts = []

    def utf(self, s):
        b = s.encode('utf-8')
        assert len(b) < 65536 and '\0' not in s
        self.parts.append(struct.pack('>H', len(b)) + b)

    def i32(self, v): self.parts.append(struct.pack('>i', v))
    def i64(self, v): self.parts.append(struct.pack('>q', v))
    def u8(self, v): self.parts.append(struct.pack('>B', v))
    def i8(self, v): self.parts.append(struct.pack('>b', v))

    def bytes(self): return b''.join(self.parts)


class _In:
    def __init__(self, data):
        self.d = data; self.p = 0

    def utf(self):
        n = struct.unpack_from('>H', self.d, self.p)[0]; self.p += 2
        s = self.d[self.p:self.p + n].decode('utf-8'); self.p += n
        return s

    def i32(self): v = struct.unpack_from('>i', self.d, self.p)[0]; self.p += 4; return v
    def i64(self): v = struct.unpack_from('>q', self.d, self.p)[0]; self.p += 8; return v
    def u8(self): v = self.d[self.p]; self.p += 1; return v
    def i8(self): v = struct.unpack_from('>b', self.d, self.p)[0]; self.p += 1; return v


def q_logp(p):
    return max(0, min(255, int(round(-math.log(p) * LOGP_SCALE))))


def q_pmi(pmi):
    return max(-127, min(127, int(round(pmi * PMI_SCALE))))


class ImportsModel:
    """paths: list of str; path_doc: list of int; names: dict name -> (docCount, [(pathId, q)]); co: dict pathId -> [(otherId, qpmi)]"""

    def __init__(self, language, params, total_docs, paths, path_doc, names, co, corpus_id=''):
        self.language = language
        self.params = params
        self.total_docs = total_docs
        self.paths = paths
        self.path_doc = path_doc
        self.names = names
        self.co = co
        self.corpus_id = corpus_id
        self.path_id = {p: i for i, p in enumerate(paths)}
        self.lam = float(params.get('lambda', '1.0'))
        self.ctx_norm = params.get('ctx_norm', 'sum')
        self.co_index = {a: {b: q for b, q in lst} for a, lst in co.items()}

    # ------------------------------------------------------------------ format
    def write(self, file):
        o = _Out()
        o.parts.append(MAGIC)
        o.i32(VERSION); o.utf(KIND); o.utf(self.language); o.i64(0); o.i64(int(time.time() * 1000)); o.utf(self.corpus_id)
        o.i32(len(self.params))
        for k, v in self.params.items():
            o.utf(k); o.utf(str(v))
        o.i32(self.total_docs)
        o.i32(len(self.paths))
        for p, c in zip(self.paths, self.path_doc):
            o.utf(p); o.i32(c)
        o.i32(len(self.names))
        for name in sorted(self.names):
            doc, lst = self.names[name]
            o.utf(name); o.i32(doc); o.u8(len(lst))
            for pid, q in lst:
                o.i32(pid); o.u8(q)
        o.i32(len(self.co))
        for pid in sorted(self.co):
            lst = self.co[pid]
            o.i32(pid); o.u8(len(lst))
            for oid, q in lst:
                o.i32(oid); o.i8(q)
        with gzip.open(file, 'wb', compresslevel=9) as f:
            f.write(o.bytes())

    @staticmethod
    def read(file):
        with gzip.open(file, 'rb') as f:
            inp = _In(f.read())
        assert inp.d[:4] == MAGIC; inp.p = 4
        assert inp.i32() == VERSION
        kind = inp.utf(); assert kind == KIND, kind
        language = inp.utf(); inp.i64(); inp.i64(); corpus_id = inp.utf()
        params = {}
        for _ in range(inp.i32()):
            k = inp.utf(); params[k] = inp.utf()
        total = inp.i32()
        n = inp.i32()
        paths = []; path_doc = []
        for _ in range(n):
            paths.append(inp.utf()); path_doc.append(inp.i32())
        names = {}
        for _ in range(inp.i32()):
            name = inp.utf(); doc = inp.i32(); k = inp.u8()
            names[name] = (doc, [(inp.i32(), inp.u8()) for _ in range(k)])
        co = {}
        for _ in range(inp.i32()):
            pid = inp.i32(); k = inp.u8()
            co[pid] = [(inp.i32(), inp.i8()) for _ in range(k)]
        return ImportsModel(language, params, total, paths, path_doc, names, co, corpus_id)

    # ------------------------------------------------------------------ queries (mirror ImportsModel.kt exactly)
    def pmi(self, a, b):
        """Quantised PMI between two path ids (0 when the pair is not stored)."""
        q = self.co_index.get(a, {}).get(b)
        if q is None:
            q = self.co_index.get(b, {}).get(a, 0)
        return q

    def rank_imports(self, name, current, lam=None):
        """[(path, score)] for an unresolved identifier given the imports present in the file; score = ln p(path | name)
        + lambda * sum of PMI(path, import) over the present imports (quantised arithmetic on int16-ish units: /16)."""
        lam = self.lam if lam is None else lam
        ent = self.names.get(name)
        if ent is None:
            return []
        ctx = [self.path_id[c] for c in current if c in self.path_id]
        n = len(ctx)
        scale = 1.0 if self.ctx_norm == 'sum' or n == 0 else (1.0 / math.sqrt(n) if self.ctx_norm == 'sqrt' else 1.0 / n)
        out = []
        for pid, q in ent[1]:
            s = float(-q)
            if lam != 0.0 and n > 0:
                acc = 0
                for c in ctx:
                    if c != pid:
                        acc += self.pmi(pid, c)
                s += lam * acc * scale
            out.append((self.paths[pid], s / LOGP_SCALE))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out

    def rank_co_imports(self, current, limit=20):
        """[(path, score)] of imports that usually accompany the present ones: ln p(path) + sum PMI(path, import)."""
        ctx = [self.path_id[c] for c in current if c in self.path_id]
        ctx_set = set(ctx)
        acc = {}
        for c in ctx:
            for oid, q in self.co.get(c, ()):
                if oid not in ctx_set:
                    acc[oid] = acc.get(oid, 0) + q
        out = []
        for oid, s in acc.items():
            prior = q_logp(self.path_doc[oid] / self.total_docs)
            out.append((self.paths[oid], (s - prior) / LOGP_SCALE))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out[:limit]
