# Real C# completion lists (e18) — exported from the .NET plugin headlessly on the server, 2026-10-07

One `.cmlx` example shard per repository (gzip, `ml-core` `ExampleShards`: 224-feature vectors per candidate, chosen index,
candidate names, context kind), produced by `idea-dotnet-support` `mlDataset` (`-Pml.maxFiles=60 -Pml.maxCopy=400 -Pml.names=true`,
10 positions per file, n-gram features from `cs-ngram-e15-a.cml`). Repositories: the `rank` fold (`rank-repos.txt`, 300 sampled);
the `test` fold (`test-repos.txt`, first 100) follows. Train/evaluate: `ml-train l1 --lang csharp --shards <dir> --test-shards <dir>`.
Kept in git because the server is temporary; ~0.3 MB per repository.
