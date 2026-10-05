# Server runbook

Scripts for a full training run on a Linux box (target: 24 cores, 64 GB RAM, Ubuntu/Debian or RHEL-family). All paths
are relative to the repository root; data goes to `../ml-data` next to the clone.

| step | command | what to send back |
|---|---|---|
| 0 | `git clone https://github.com/dvislobokov/idea-ml-completion.git && cd idea-ml-completion` | — |
| 1 | `tools/server/setup.sh` | the `== environment` block |
| 2 | `nohup tools/server/fetch.sh > ../ml-data/fetch.log 2>&1 &` — later `tools/server/stats.sh` | the stats lines |
| 3 | `tools/server/sets.sh go && tools/server/sets.sh csharp` | the two printed lines |
| 4 | `nohup tools/server/train.sh go full-mkn4 > /dev/null 2>&1 &` (then the same for `csharp`) | `../ml-data/<lang>/exp/<tag>/summary.txt` |
| 5 | `tools/server/report.sh` | `../ml-data/report.txt` |

Progress of a running step: `tail -f ../ml-data/fetch.log` or `tail -f ../ml-data/<lang>/exp/<tag>/l2.log`.

`train.sh` passes extra arguments to `l2` (`--min-count 1,1,2,2`, `--min-repos 1,1,1,3`, `--order 5`, `--smoothing jm`), so an
experiment series is a list of one-liners with different tags; each run is self-contained in its `exp/<tag>` directory.
Memory knobs: `XMX_LM` (default 40g), `XMX_RANK` (20g); `PER_FILE` (20 ranker lists per file) and `MAX_TEST` (4000
evaluation files) bound ranker RAM and evaluation time. Expected on the full corpus: cloning 30–60 min, LM counting
10–30 min per language, ranker 5–15 min.
