# Oxide AI continuation checkpoint — 2026-09-17

## Current user-facing state

The user asked to keep going and finish everything, supplied the updated handoff PDF, instructed ordinary Git commands for pushes, and pointed to the more recent draft PR. The authoritative source was recovered from draft PR #2's branch, not the older ZIP.

Correctness/preflight repairs are complete. **Danil explicitly approved the estimated 2–3-hour full study at 13:26 Pacific and requested GitHub publication. The sequential study is RUNNING**, launched via `scripts/run-study.sh` (initial driver PID 7686; first configuration started 20:26:51 UTC). Do not launch a duplicate. A temporary 20-minute monitoring automation is active: `cti_rkx68pgfmbyh7maar4qq`, first check 13:47 Pacific, 12 scheduled checks. Follow `TRAINING-MONITOR.md` for current progress/recovery/completion instructions.

## Accepted milestone

- Source branch: `research/stacked-depth-wip-20260917`.
- Recovered remote base: `5cdd9fb7e7cfcbf646da570944e6cd238f545aa7`.
- New local accepted commit: `9410737958a358d25a2a315b3c303d1576ed19e7`.
- Author and committer: `sparticle62ops <sparticle62ops@users.noreply.github.com>`.
- Local working tree was clean after commit.
- `cargo test --locked --release -j1 --all-targets -- --nocapture`: **56 passed, 0 failed**, 18 executable targets including zero-test targets. Separate release runner build exit 0. Rust 1.90.0, locked deps, `CARGO_BUILD_JOBS=1`, `CARGO_PROFILE_RELEASE_LTO=false`, `RUSTFLAGS='-Dwarnings -C target-cpu=native'`.
- Normal `git push --dry-run` failed with missing HTTPS authentication. After Danil explicitly requested publication and was told the connected account would be used, the accepted 11-file tree was pushed to existing draft PR #2 via the connection. Remote commit: `319bb99fac9c3c897d0a1564074af0cb8b4e7d37`, with explicit sparticle62ops co-author trailer. Normal Git fetch/diff proved that its tree is identical to the locally accepted pure-identity commit. A PR comment records tests and the training launch. **Publication succeeded; no merge occurred.** Receipts: `publication/github-push-receipt.json` and `publication/github-comment-receipt.json`.

## Proven defects repaired

1. Training grouping mixed group and chunk indices, causing its original unit test to fail with an empty update group. Fixed range `[g*A, min((g+1)*A,N))`; added full coverage tests.
2. Default serde_json float parsing changed the beta2 numeric value by one f64 ULP on round trip, breaking exact provenance equality. Reproduced with a dedicated test; enabled `float_roundtrip`, without relaxing comparisons.
3. Several old tests and perf example retained pre-extraction field paths. Restored exact field-only migrations from the ZIP's Stage-A reference; mechanically proved removal of `.block.` recovers original source exactly for all four old test files. Discarded an initially overcompressed checkpoint-test rewrite.
4. The first gradient-fixture proposal merely changed a shared output-head offset, which softmax cancels; it failed empirically. Corrected by scaling head contrast by four. All original FD tolerances and non-vacuity thresholds retained. Six focused tests pass, including exhaustive small-fixture parameter coverage at depths 2/4 with populated memory. See `gradient-audit-validation.md`; it supersedes the unverified earlier proposal in `gradient-audit.md`.
5. Retained periodic checkpoints; added initial validation, cursor/update/exposure invariants, path checks, preflight binding, mismatch evidence isolation and completed-checkpoint validation. Uninterrupted/segmented resume matches byte-for-byte at depths 1/2/4.

## Frozen study readiness

Private persistent root: `/tasklet/threads/a_a155y2axk4bq63ydhfnc/work/study`.

- `inputs/`: exact frozen train/validation/prompts and initial width64 V7 tokenizer control. SHA-256 checked against prior experiment metadata. Test corpus was NOT read or evaluated and remains only in the original private ZIP.
- `manifests/`: preflight, full-run and bounded variants for w64-d1, w256-d1, w64-d2, w64-d4.
- `preflight-acceptance.json`: all four accepted; **840951 transitions/pass, 3363804 total, 13184 updates**, identical frozen controls and tokenizer metadata SHA-256.
- `provenance.json`: input hashes, tokenizer-source identity via preflights, native binary hash, accepted commit and CPU flags.
- `runs/w64-d1/`: 64 updates / 16251 transitions completed as a resumable fresh-baseline slice. Uses the full four-pass schedule, not a shortened schedule. State is authoritative; no epochs completed yet.
- Initial validation reproduces NLL 7.9479711999981015, perplexity 2829.827967720145, accuracy 0.00026441236241113854 (fraction).
- Bounded compute: 4.2458 seconds, about 3827.56 transitions/compute-second. Estimated baseline full training compute ~14.65 minutes; wider/deeper configurations plus overhead motivate the 2–3-hour overall estimate. Only baseline timing is measured; other configuration durations are extrapolations.

## Launch after approval

No Rust reinstall/rebuild is needed on a compatible x86_64 CPU. The accepted native runner binary is durable at `work/bin/width_depth_study-linux-x86_64` (hash in provenance). Its source is recoverable from `publication/oxide-tested-source.tar.gz` or Git bundle.

```sh
nohup sh /tasklet/threads/a_a155y2axk4bq63ydhfnc/work/scripts/run-study.sh \
  > /tasklet/threads/a_a155y2axk4bq63ydhfnc/work/study/logs/study-driver.log 2>&1 &
```

The launcher checks the strict frozen-study verifier and binary SHA, acquires a persistent lock, then executes the four full manifests sequentially. It copies the binary to `/tmp/oxide-study-runtime` for execution. All meaningful checkpoints/logs/metrics are written directly to persistent storage. Native binaries may not run on a different CPU; compare recorded flags or rebuild identically if necessary. Do not launch concurrent trainers. If a stale `study/worker.lock` exists after a reset, inspect its owner and running processes before removing it.

Runner checkpoints every 512 updates and diagnostics every 256. `state.json` advances during training; `experiment.json` may still show the previous paused status until completion, so use authoritative state plus process status for progress. Preserve every checkpoint and failure. The launch script writes per-run timing/peak-RSS JSON and a final driver exit code. Avoid rerunning `prepare-study.ts` after restoration: it references the original `/tmp` import; the durable manifests are already prepared.

Temporary monitoring is already scheduled every 20 minutes beginning 13:47 Pacific, with bounded budget-model inspections and final review instructions in `TRAINING-MONITOR.md`. Automation ID: `cti_rkx68pgfmbyh7maar4qq`. Delete it after completed review or a terminal unresolved blocker; list current-thread automations first. Do not create a duplicate.

## After the study

Compare final-epoch validation and all 16 unselected generations per configuration, diagnostics and throughput. Do not tune on the test set or pick flattering samples. A one-seed result is only a pilot; replicate a promising improvement before broad claims. Coherence is still unproven. Follow the user's mandatory order: coherent/logical model first → CLI redesign → WGPU. Do not merge draft PR #2 without explicit permission.

## Preserved artifacts

- `publication/oxide-study-repair.patch` — clean attributed milestone patch.
- `publication/oxide-study-repair.bundle` — complete Git history through the accepted local commit; verified.
- `publication/oxide-tested-source.tar.gz` — committed source tree.
- `publication/artifact-checksums.json` — SHA-256/size records.
- `evidence/` — passing and failed build/test logs, exit codes, and `acceptance-summary.json`.
- `scripts/` — preparation, validation, measurement, strict preflight verification and sequential launcher.

Never publish the private recovery ZIP, transcripts, corpus or this private continuation file to GitHub. Public commit scope was explicitly allowlisted to source, tests, build metadata, the verifier and status notes.
