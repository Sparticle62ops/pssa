#!/usr/bin/env python3
"""Opt-in sequential paired runner. Standard library only; never invokes a shell.

See docs/PAIRED-EVAL.md for the manifest, card contract and statistical caveats.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

DEFAULT_SEEDS = [11, 29, 47]
DEFAULT_METRICS = {
    "cross_entropy": {"path": "/metrics/cross_entropy", "direction": "lower"},
    "perplexity": {"path": "/metrics/perplexity", "direction": "lower"},
    "targets_per_second": {"path": "/metrics/targets_per_second", "direction": "higher"},
}
DEFAULT_MATCH = ["/identity", "/protocol", "/model/parameters", "/model/optimizer_updates"]
MASK = (1 << 64) - 1


def read_json(text):
    def invalid(value):
        raise ValueError(f"nonstandard JSON constant: {value}")
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed
    return json.loads(text, parse_constant=invalid, parse_float=number)


def pointer(card, path):
    if not path.startswith("/"):
        raise ValueError("card paths must be JSON pointers starting with /")
    value = card
    for key in path[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    if value is None:
        raise ValueError(f"null card field: {path}")
    return value


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("metric must be a finite number")
    return float(value)


def mean(values):
    values = list(values)
    scale = max(abs(value) for value in values)
    return (math.fsum(value / scale for value in values) / len(values)) * scale if scale else 0.0


def quantile(sorted_values, p):
    position = (len(sorted_values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    fraction = position - lo
    return (1 - fraction) * sorted_values[lo] + fraction * sorted_values[hi]


class SplitMix64:
    """Specified integer PRNG, independent of Python's random implementation."""
    def __init__(self, seed):
        self.state = seed

    def below(self, bound):
        limit = (1 << 64) - (1 << 64) % bound
        while True:
            self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
            z = self.state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
            z ^= z >> 31
            if z < limit:
                return z % bound


def bootstrap_interval(deltas, resamples, seed):
    if len(deltas) < 2:
        return None
    rng = SplitMix64(seed)
    means = sorted(mean(deltas[rng.below(len(deltas))] for _ in deltas)
                   for _ in range(resamples))
    return [quantile(means, 0.025), quantile(means, 0.975)]


def aggregate(records, metrics, match, resamples, bootstrap_seed):
    """Pair by explicit seed, never unpaired means or successful-side counts."""
    summaries = {}
    anchor = None
    for record in records:
        record["pair_error"] = None
        if not all(record[side]["ok"] for side in ("control", "variant")):
            record["pair_error"] = "one or both commands failed"
            continue
        try:
            left = [pointer(record["control"]["card"], path) for path in match]
            right = [pointer(record["variant"]["card"], path) for path in match]
            if left != right:
                raise ValueError("paired card identity/protocol/budget mismatch")
            if anchor is not None and left != anchor:
                raise ValueError("card identity/protocol/budget differs across seeds")
            anchor = left
        except (KeyError, IndexError, TypeError, ValueError) as error:
            record["pair_error"] = str(error)
    for name, spec in metrics.items():
        pairs, errors = [], []
        for record in records:
            try:
                if record["pair_error"]:
                    raise ValueError(record["pair_error"])
                a = finite_number(pointer(record["control"]["card"], spec["path"]))
                b = finite_number(pointer(record["variant"]["card"], spec["path"]))
                delta = finite_number(b - a)
                pairs.append({"seed": record["seed"], "control": a, "variant": b, "delta": delta})
            except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
                errors.append({"seed": record["seed"], "error": str(error)})
        deltas = [p["delta"] for p in pairs]
        interval = bootstrap_interval(deltas, resamples, bootstrap_seed)
        signal_direction = "unavailable" if interval is None else (
            "lower" if interval[1] < 0 else "higher" if interval[0] > 0 else "overlaps_zero")
        summaries[name] = {
            "direction": spec["direction"], "n": len(pairs), "failure_count": len(errors),
            "pairs": pairs, "failures": errors,
            "control_mean": mean(p["control"] for p in pairs) if pairs else None,
            "variant_mean": mean(p["variant"] for p in pairs) if pairs else None,
            "mean_delta": mean(deltas) if pairs else None,
            "median_delta": quantile(sorted(deltas), 0.5) if pairs else None,
            "bootstrap_95": interval, "interval_signal": signal_direction,
        }
    return summaries


def validate(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise ValueError("manifest schema must be 1")
    seeds = manifest.get("seeds", DEFAULT_SEEDS)
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or not 0 <= s <= MASK for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct u64 integers in an explicit ordered list")
    for side in ("control", "variant"):
        steps = manifest.get(side)
        if (not isinstance(steps, list) or not steps or
                any(not isinstance(cmd, list) or not cmd or
                    any(not isinstance(arg, str) or not arg for arg in cmd) for cmd in steps)):
            raise ValueError(f"{side} must contain nonempty argv arrays")
        if not any("{seed}" in arg for cmd in steps for arg in cmd):
            raise ValueError(f"{side} must use {{seed}} (do not relabel one run as multiple seeds)")
    metrics = manifest.get("metrics", DEFAULT_METRICS)
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a nonempty mapping")
    for name, spec in metrics.items():
        if (not isinstance(spec, dict) or spec.get("direction") not in ("lower", "higher") or
                not isinstance(spec.get("path"), str) or not spec["path"].startswith("/")):
            raise ValueError(f"invalid metric {name}")
    match = manifest.get("match", DEFAULT_MATCH)
    if not isinstance(match, list) or not match or any(not isinstance(p, str) or not p.startswith("/") for p in match):
        raise ValueError("match must list at least one card identity/budget JSON pointer")
    timeout = manifest.get("timeout_seconds", 240)
    if type(timeout) not in (float, int) or not 0 < timeout <= 300:
        raise ValueError("timeout_seconds must be in (0, 300]")
    for field, default, lo, hi in [("bootstrap_resamples", 10000, 100, 1000000),
                                    ("bootstrap_seed", 20260501, 0, MASK)]:
        value = manifest.get(field, default)
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"invalid {field}")
    env = manifest.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("env must map names to strings; do not put secrets in manifests")
    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, list) or any(not isinstance(p, str) for p in inputs):
        raise ValueError("inputs must list immutable local data/source files")
    return seeds, metrics, match


def run_side(steps, seed, side, out, env, timeout):
    run_dir = out / f"{seed}-{side}"
    run_dir.mkdir()
    history = []
    try:
        for index, template in enumerate(steps):
            command = [arg.replace("{seed}", str(seed)).replace("{side}", side)
                       .replace("{run_dir}", str(run_dir)) for arg in template]
            stdout = run_dir / f"{index}.stdout"
            stderr = run_dir / f"{index}.stderr"
            with stdout.open("xb") as output, stderr.open("xb") as errors:
                start = time.monotonic()
                entry = {"argv": command, "exit_code": None, "timeout": False}
                history.append(entry)
                with subprocess.Popen(command, stdout=output, stderr=errors, env=env,
                                      start_new_session=True) as process:
                    timed_out = False
                    try:
                        code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
                entry.update({"exit_code": code, "timeout": timed_out,
                              "seconds": time.monotonic() - start})
                if code != 0 or timed_out:
                    raise ValueError(f"step {index}: {'timeout' if timed_out else f'exit {code}'}")
        if stdout.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("final card exceeds 8 MiB")
        card = read_json(stdout.read_text())
        if not isinstance(card, dict):
            raise ValueError("final stdout must be exactly one JSON card object")
        return {"ok": True, "card": card, "commands": history}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error), "commands": history}


def file_hashes(paths):
    result = {}
    for name in paths:
        digest = hashlib.sha256()
        with open(name, "rb") as source:
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result


def run(manifest, out):
    seeds, metrics, match = validate(manifest)
    # All paths/commands are relative to the caller's cwd, not the manifest.
    before = file_hashes(manifest.get("inputs", []))
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    env = {**os.environ, "RAYON_NUM_THREADS": "1", **manifest.get("env", {})}
    records = []
    for index, seed in enumerate(seeds):
        order = ["control", "variant"] if index % 2 == 0 else ["variant", "control"]
        record = {"seed": seed, "order": order}
        for side in order:
            record[side] = run_side(manifest[side], seed, side, out, env,
                                    manifest.get("timeout_seconds", 240))
        records.append(record)
    try:
        unchanged = before == file_hashes(manifest.get("inputs", []))
    except OSError:
        unchanged = False
    summary = aggregate(records, metrics, match, manifest.get("bootstrap_resamples", 10000),
                        manifest.get("bootstrap_seed", 20260501))
    report = {
        "schema": 1, "kind": "paired_multiseed", "seeds": seeds, "records": records,
        "metrics": summary,
        "failures": {side: sum(not r[side]["ok"] for r in records) for side in ("control", "variant")},
        "failed_pairs": sum(r["pair_error"] is not None for r in records),
        "inputs_sha256": before, "inputs_unchanged": unchanged,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "protocol": {
            "delta": "variant minus control", "bootstrap_unit": "complete seed pair",
            "bootstrap": "SplitMix64 percentile 95%, linear interpolated quantiles",
            "bootstrap_seed": manifest.get("bootstrap_seed", 20260501),
            "bootstrap_resamples": manifest.get("bootstrap_resamples", 10000),
            "rayon_threads": env["RAYON_NUM_THREADS"], "cwd": os.getcwd(),
            "warning": "Three seeds are smoke evidence, not a powered study. Percentile intervals are fragile at small n; no multiple-comparison correction. Missing pairs can bias survivor estimates. Timing is not byte-stable.",
        },
    }
    report["valid"] = unchanged and not any(item["failure_count"] for item in summary.values())
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new directory; never overwrite")
    args = parser.parse_args()
    try:
        report = run(read_json(args.manifest.read_text()), args.out)
    except (OSError, ValueError) as error:
        parser.exit(2, f"paired-eval: {error}\n")
    print(json.dumps({"metrics": report["metrics"], "failures": report["failures"],
                      "valid": report["valid"]}, indent=2, allow_nan=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
