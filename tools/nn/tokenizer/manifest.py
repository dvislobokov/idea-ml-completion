import json, os

DATA = os.path.expanduser("~/work/ml-data")


def read_manifest(lang="go", fold="lm", status="ok"):
    """Yield manifest records; tolerates a half-written last line (the manifest may still be growing)."""
    path = f"{DATA}/{lang}/prepared/manifest.jsonl"
    with open(path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d["fold"] == fold and d["status"] == status:
                yield d


def file_path(lang, d):
    return f"{DATA}/{lang}/repos/{d['repo']}/{d['path']}"


def read_file(lang, d):
    with open(file_path(lang, d), "rb") as f:
        return f.read()
