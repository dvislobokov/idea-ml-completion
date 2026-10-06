"""encode_corpus.py with the secret filter applied at read time - without modifying encode_corpus.py.

    cd ~/work/nn/clean && ~/work/nn/.venv/bin/python -I encode_corpus_clean.py --vocab <vocab.cml> --lang go --fold lm [...]

Every option is passed through to encode_corpus.main().  `encode_corpus.read_file` is replaced by a version that runs
secret_filter.scrub_bytes() on the file: scrubbed values are replaced in place; a dropped file (private key block,
>= 3 credential hits, data file) becomes an empty file, i.e. a 0-token entry in <fold>.offsets.u64 so the file table
stays aligned with the manifest (the data loader already skips empty files via offsets[i] == offsets[i+1]).
Set SECRET_FILTER_POLICY=json to override secret_filter.DEFAULT_POLICY (e.g. '{"drop_on_private_key": false}').
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.expanduser("~/work/nn/tokenizer")
sys.path.insert(0, HERE)
sys.path.insert(0, TOKENIZER_DIR)

import secret_filter as sf  # noqa: E402
import manifest  # noqa: E402
import encode_corpus  # noqa: E402

POLICY = dict(sf.DEFAULT_POLICY, **json.loads(os.environ.get("SECRET_FILTER_POLICY", "{}")))


def read_file_clean(lang, d):
    data = manifest.read_file(lang, d)
    clean, _stats = sf.scrub_bytes(data, d.get("path", ""), POLICY)
    return clean if clean is not None else b""


encode_corpus.read_file = read_file_clean  # _work() resolves read_file from encode_corpus' globals, also in forked workers

if __name__ == "__main__":
    print(f"secret filter active: policy={POLICY}", flush=True)
    encode_corpus.main()
