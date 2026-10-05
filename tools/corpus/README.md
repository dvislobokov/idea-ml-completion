# tools/corpus — selecting pretraining repositories

Shell + `gh` + `jq` + `duckdb` pipeline that produces a reviewable list of GitHub repositories per language.
Nothing is cloned here; the output is a list for human review (`data/<lang>-repos.csv|.md`).

Pipeline (all steps are resumable, re-running skips what is done):

1. `./search.sh csharp "C#"` / `./search.sh go Go` — GitHub Search API, sliced by star ranges to bypass the
   1000-results cap. Query: `language:<L> fork:false archived:false stars:>=1000 pushed:>2025-06-01`.
   Output `data/<lang>-search.jsonl`.
2. `./enrich.sh <lang> <Key> [jobs]` — prefilter (license MIT/Apache-2.0/BSD-2/BSD-3, size 1 MB..3 GB), then
   `/repos/{o}/{r}/languages` for the language share. Output `data/<lang>-enriched.jsonl`.
3. `./verify.sh <lang> [jobs]` — for share >= 60%: `/repos/{o}/{r}` for created_at, forks, watchers, open issues.
   Output `data/<lang>-verified.jsonl`.
4. `./select.sh <lang> [limit]` — quality filters and per-owner quotas, sorted by stars:
   created before 2024-07-01; forks/stars >= 0.03 and forks <= stars (drops star-inflated repos and fork farms);
   watchers >= 20; name/description/topics do not match lists/tutorials/templates/samples;
   quota 5 repos per owner (dotnet/microsoft 15; golang/kubernetes/hashicorp 12); Unity <= 10 (C#).
   Output `data/<lang>-repos.csv`.
5. `./report.sh <lang> <Title>` — Markdown table `data/<lang>-repos.md`.

API budget: ~1 search call per 100 repos, 1 call per repo for enrich and verify (core limit 5000/h).
