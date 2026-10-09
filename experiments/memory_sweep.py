#!/usr/bin/env python3
"""Short CPU capacity/retrieval sweep; run after building pssa through csrun.
Usage: python3 experiments/memory_sweep.py "$CARGO_TARGET_DIR/fast/pssa"
No dependencies. All corpora/checkpoints are temporary and deleted on exit.
Reports per-run JSON (not a benchmark claim). Linux wait4 records child peak RSS.
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def corpus(seed, n):
    rng = random.Random(seed)
    # >exp(3.5) word vocabulary keeps the existing surprise-gated writer active
    # for enough of this short run to probe growing occupancy. No gate changes.
    subjects = [f"object{i}" for i in range(32)]
    places = [f"box{i}" for i in range(32)]
    return " ".join(f"{rng.choice(subjects)} moves to {rng.choice(places)} ." for _ in range(n))


def run(binary):
    with tempfile.TemporaryDirectory(prefix="pssa-memory-sweep-") as temp:
        root = Path(temp)
        train, held = root / "train.txt", root / "held.txt"
        train.write_text(corpus(17, 7000))
        held.write_text(corpus(29, 1000))
        env = dict(os.environ, RAYON_NUM_THREADS="1")
        for seed in [7, 19]:
            for capacity in [64, 256, 1024, 4096]:
                for k in [None, 4]:
                    model, logfile = root / "model.pssa", root / "train.log"
                    flags = [] if k is None else ["--memory-top-k", str(k)]
                    args = [binary, "train", str(train), "--backend", "cpu", "--threads", "1",
                            "--tokenizer", "word", "--latent", "16", "--state", "4", "--key", "4",
                            "--memory", str(capacity), "--chunk", "8", "--accumulate", "1",
                            "--epochs", "1", "--max-tokens", "32769", "--seed", str(seed),
                            "--no-tui", "--no-feed-telemetry", "-o", str(model), *flags]
                    start = time.perf_counter()
                    sampled_rss = 0
                    with logfile.open("w") as log:
                        process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
                        while True:
                            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                            if pid:
                                process.returncode = os.waitstatus_to_exitcode(status)
                                break
                            try:
                                status_text = Path(f"/proc/{process.pid}/status").read_text()
                                # wait4 can include the Python fork's pre-exec RSS floor.
                                # Separately sample the binary's post-exec high-water mark.
                                if re.search(r"^Name:\s+pssa$", status_text, re.M):
                                    sampled_rss = max(sampled_rss, int(re.search(r"VmHWM:\s+(\d+)", status_text)[1]))
                            except (FileNotFoundError, ProcessLookupError, TypeError):
                                pass
                            if time.perf_counter() - start > 90:
                                process.kill()
                                _, status, usage = os.wait4(process.pid, 0)
                                process.returncode = os.waitstatus_to_exitcode(status)
                                raise TimeoutError("bounded training run exceeded 90 seconds")
                            time.sleep(0.01)
                    wall = time.perf_counter() - start
                    log = logfile.read_text()
                    if process.returncode:
                        raise RuntimeError(log[-3000:])
                    seconds = float(re.search(r"training_seconds=([\d.]+)", log)[1])
                    tokens = int(re.search(r"epoch 1/1 loss=\S+ tokens=(\d+)", log)[1])
                    score = subprocess.run([binary, "score", str(held), "--model", str(model), *flags],
                                           capture_output=True, text=True, env=env, check=True, timeout=90)
                    metrics = json.loads(next(line for line in score.stdout.splitlines() if line.startswith('{"cross_entropy"')))
                    occupancy = re.findall(r"memory_occupancy=(\d+)/(\d+)", log)
                    print(json.dumps(dict(seed=seed, capacity=capacity, top_k=k, train_tokens=tokens,
                        training_seconds=seconds, tokens_per_second=round(tokens / seconds, 1),
                        wall_seconds=round(wall, 3), peak_rss_kib=usage.ru_maxrss, sampled_post_exec_hwm_kib=sampled_rss,
                        heldout_ce=metrics["cross_entropy"], heldout_tokens=metrics["token_count"],
                        final_reported_occupancy=occupancy[-1] if occupancy else None)), flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
