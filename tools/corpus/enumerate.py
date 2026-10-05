"""Enumerates GitHub repositories of a language through the search API (via an authenticated `gh`) into a catalogue.

usage: uv run --python 3.12 python tools/corpus/enumerate.py <go|csharp> <out.jsonl> [--min-stars 20] [--max-stars N] [--min-size-kb 300]

The search API returns at most 1000 results per query, so the space is sliced by creation date: a window that still holds more
than 1000 repositories is halved until it fits. Each line of the output: full_name, stars, size_kb (whole repository, history
included), license, created_at, pushed_at, archived, default_branch. Resumable: windows already written are skipped.
Rate limit: 30 search requests per minute; ~40 k repositories take about an hour.
"""
import json
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

LANG = {"go": "go", "csharp": "csharp"}


def gh_search(q: str, page: int = 1, per_page: int = 100, retries: int = 6) -> dict:
    for attempt in range(retries):
        r = subprocess.run(["gh", "api", "-X", "GET", "search/repositories", "-f", f"q={q}", "-f", f"per_page={per_page}",
                            "-f", f"page={page}", "-f", "sort=stars"], capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            return json.loads(r.stdout)
        msg = (r.stderr or r.stdout)[:200]
        wait = 65 if "rate limit" in msg.lower() or "403" in msg else 10 * (attempt + 1)
        print(f"  retry in {wait}s: {msg.strip()}", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"search failed: {q}")


def main() -> None:
    args = sys.argv[1:]
    lang, out = LANG[args[0]], Path(args[1])
    opts = dict(zip(args[2::2], args[3::2]))
    min_stars = int(opts.get("--min-stars", 20))
    max_stars = opts.get("--max-stars")
    min_size = int(opts.get("--min-size-kb", 300))
    stars = f"stars:{min_stars}..{max_stars}" if max_stars else f"stars:>={min_stars}"
    base = f"language:{lang} fork:false size:>{min_size} {stars}"

    done_path = out.with_suffix(".windows")
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    seen = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["full_name"])
    print(f"{lang}: {len(seen)} repositories already catalogued, {len(done)} windows done", file=sys.stderr)

    # initial windows: one per year up to 2015, then quarters; windows over 1000 results are split on the fly
    windows: list[tuple[date, date]] = []
    y = 2008
    while y <= 2015:
        windows.append((date(y, 1, 1), date(y, 12, 31))); y += 1
    d = date(2016, 1, 1)
    today = date.today()
    while d <= today:
        e = min(d + timedelta(days=91), today)
        windows.append((d, e)); d = e + timedelta(days=1)

    with out.open("a", encoding="utf-8") as f, done_path.open("a") as dp:
        while windows:
            a, b = windows.pop(0)
            key = f"{a}..{b}"
            if key in done:
                continue
            q = f"{base} created:{a}..{b}"
            first = gh_search(q, 1)
            total = first["total_count"]
            if total > 1000 and (b - a).days >= 1:
                mid = a + (b - a) / 2
                windows[:0] = [(a, mid), (mid + timedelta(days=1), b)]
                print(f"  split {key}: {total}", file=sys.stderr)
                continue
            pages = [first] + [gh_search(q, p) for p in range(2, min(10, (total + 99) // 100) + 1)]
            n = 0
            for page in pages:
                for item in page["items"]:
                    name = item["full_name"]
                    if name in seen:
                        continue
                    seen.add(name); n += 1
                    f.write(json.dumps({
                        "full_name": name, "stars": item["stargazers_count"], "size_kb": item["size"],
                        "license": (item.get("license") or {}).get("spdx_id"), "created_at": item["created_at"],
                        "pushed_at": item["pushed_at"], "archived": item["archived"], "default_branch": item["default_branch"],
                    }, ensure_ascii=False) + "\n")
            f.flush(); dp.write(key + "\n"); dp.flush()
            print(f"{key}: {total} found, {n} new, {len(seen)} total", file=sys.stderr)
            time.sleep(2.1)   # 30 searches per minute


if __name__ == "__main__":
    main()
