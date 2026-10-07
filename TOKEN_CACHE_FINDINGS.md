# Token-cache findings

The training window path now stops after the requested `skip + max_tokens` prefix when no cache is configured. The first persistent-cache run performs one full, ordered Rayon tokenization and writes `<dataset>.pssatok`; later chained windows read the compact binary cache in one buffered read.

## Timing

| workload | old full serial pass | new first cached pass | new cached window |
| --- | ---: | ---: | ---: |
| 4.38 GB molab corpus / 500k-token link | 35+ min (reported production run) | one parallel full pass (not measured in this checkout) | token-window selection only (not measured in this checkout) |

A reproducible large-file measurement still needs to run in the codespace. The required `csrun` check could not start here because the local GitHub CLI has no `codespace` authentication (`gh auth login -s codespace` is required); no local cargo build or test was run because this worktree explicitly forbids it. The production log line is:

```text
token_cache=built|reused token_window_seconds=<seconds>
```

The cache key includes dataset size, mtime, first/last 1 MiB FNV-1a hashes, and tokenizer identity. Corrupt or stale files are ignored and rebuilt through a temporary file followed by rename, so an interrupted build cannot replace a valid cache.
