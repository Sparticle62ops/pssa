"""Run: python3 -m unittest discover -s scripts -p 'test_paired_eval.py' -v"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import paired_eval as harness


class PairedTests(unittest.TestCase):
    def manifest(self):
        # This is an exact synthetic statistical fixture, NOT a model benchmark.
        worker = (
            "import json,sys; seed=int(sys.argv[1]); side=sys.argv[2]; "
            "a={11:4.,29:6.,47:8.}[seed]; "
            "print(json.dumps({'identity':'same', 'metrics':{'ce':a-(1 if side=='variant' else 0), "
            "'boxes':0.5+(0.25 if side=='variant' else 0)}}))"
        )
        command = [sys.executable, "-c", worker, "{seed}", "{side}"]
        return {
            "schema": 1, "seeds": [11, 29, 47],
            "control": [command], "variant": [command], "match": ["/identity"],
            "metrics": {"ce": {"path": "/metrics/ce", "direction": "lower"},
                        "boxes": {"path": "/metrics/boxes", "direction": "higher"}},
            "bootstrap_resamples": 1000,
        }

    def run_fixture(self, manifest):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            report = harness.run(manifest, out)
            persisted = json.loads((out / "report.json").read_text())
            self.assertEqual(report, persisted)
            self.assertEqual(json.loads((out / "manifest.json").read_text()), manifest)
            return report

    def test_real_subprocess_pairing_reproducibility_order_and_boxes(self):
        first = self.run_fixture(self.manifest())
        second = self.run_fixture(self.manifest())
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["seeds"], [11, 29, 47])
        self.assertEqual([r["order"] for r in first["records"]],
                         [["control", "variant"], ["variant", "control"], ["control", "variant"]])
        ce = first["metrics"]["ce"]
        self.assertEqual((ce["control_mean"], ce["variant_mean"], ce["mean_delta"]), (6, 5, -1))
        self.assertEqual(ce["bootstrap_95"], [-1, -1])
        self.assertEqual(ce["failure_count"], 0)
        self.assertEqual(first["metrics"]["boxes"]["bootstrap_95"], [0.25, 0.25])
        self.assertTrue(first["valid"])

    def test_failed_seed_is_never_used_unpaired_and_other_side_still_runs(self):
        manifest = self.manifest()
        manifest["variant"].insert(0, [sys.executable, "-c", "import sys; sys.exit(7 if sys.argv[1]=='11' else 0)", "{seed}"])
        report = self.run_fixture(manifest)
        ce = report["metrics"]["ce"]
        self.assertEqual([p["seed"] for p in ce["pairs"]], [29, 47])
        self.assertEqual((ce["control_mean"], ce["variant_mean"], ce["mean_delta"]), (7, 6, -1))
        self.assertEqual(report["failures"], {"control": 0, "variant": 1})
        self.assertEqual(report["failed_pairs"], 1)
        self.assertTrue(report["records"][0]["control"]["ok"])
        self.assertEqual(ce["failure_count"], 1)
        self.assertFalse(report["valid"])

    def test_invalid_json_and_timeout_are_failures_not_zero_scores(self):
        for command in [[sys.executable, "-c", "print('not json')", "{seed}"],
                        [sys.executable, "-c", "import time; time.sleep(5)", "{seed}"]]:
            manifest = self.manifest()
            manifest["seeds"] = [11]
            manifest["timeout_seconds"] = 0.1
            manifest["variant"] = [command]
            report = self.run_fixture(manifest)
            self.assertEqual(report["metrics"]["ce"]["n"], 0)
            self.assertIsNone(report["metrics"]["ce"]["mean_delta"])
            self.assertIsNone(report["metrics"]["ce"]["bootstrap_95"])
            self.assertEqual(report["failures"]["variant"], 1)

    def test_mismatched_cards_and_cross_seed_drift_are_excluded(self):
        report = self.run_fixture(self.manifest())
        records = report["records"]
        records[1]["variant"]["card"]["identity"] = "different"
        records[2]["control"]["card"]["identity"] = "both different"
        records[2]["variant"]["card"]["identity"] = "both different"
        metrics = harness.aggregate(records, self.manifest()["metrics"], ["/identity"], 1000, 12)
        self.assertEqual(metrics["ce"]["n"], 1)
        self.assertEqual(metrics["ce"]["failure_count"], 2)
        self.assertIsNone(metrics["ce"]["bootstrap_95"])

    def test_nonfinite_and_null_fail_per_metric_with_other_metrics_retained(self):
        records = self.run_fixture(self.manifest())["records"]
        for record, invalid in zip(records, [None, float("inf"), True]):
            record["variant"]["card"]["metrics"]["ce"] = invalid
        summary = harness.aggregate(records, self.manifest()["metrics"], ["/identity"], 1000, 12)
        self.assertEqual(summary["ce"]["failure_count"], 3)
        self.assertEqual(summary["boxes"]["n"], 3)
        for token in ["NaN", "Infinity", "-Infinity", "1e999"]:
            with self.assertRaises(ValueError):
                harness.read_json('{"ce":' + token + '}')

    def test_duplicate_seeds_and_ambiguous_settings_rejected(self):
        for change in [{"seeds": [1, 1]}, {"seeds": []}, {"seeds": [True]},
                       {"seeds": [-1]}, {"timeout_seconds": 301}, {"match": []},
                       {"bootstrap_resamples": 1}, {"control": [["echo", "same seed"]]},
                       {"metrics": {"ce": {"path": "bad", "direction": "lower"}}}]:
            manifest = {**self.manifest(), **change}
            with self.assertRaises(ValueError, msg=str(change)):
                harness.validate(manifest)
        manifest = self.manifest()
        del manifest["seeds"]
        self.assertEqual(harness.validate(manifest)[0], [11, 29, 47])

    def test_percentile_matches_exact_three_seed_distribution_and_even_median(self):
        self.assertEqual(harness.bootstrap_interval([-3., -2., -1.], 10000, 12), [-3., -1.])
        self.assertIsNone(harness.bootstrap_interval([1.], 1000, 12))
        self.assertEqual(harness.quantile([0, 10], 0.5), 5)
        a = harness.bootstrap_interval([1, -2, 3, -4], 1000, 99)
        self.assertEqual(a, harness.bootstrap_interval([1, -2, 3, -4], 1000, 99))

    def test_large_finite_metrics_do_not_overflow_the_mean_or_interval(self):
        self.assertEqual(harness.mean([1e308, 1e308]), 1e308)
        self.assertEqual(harness.quantile([-1e308, 1e308], 0.5), 0)
        self.assertEqual(harness.bootstrap_interval([1e308, 1e308], 1000, 12), [1e308, 1e308])

    def test_paths_with_spaces_are_argv_no_shell_and_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="paired fixture ") as directory:
            manifest = self.manifest()
            manifest["control"].insert(0, [sys.executable, "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')",
                "{run_dir}/literal ; file"])
            out = Path(directory) / "out"
            harness.run(manifest, out)
            self.assertEqual((out / "11-control" / "literal ; file").read_text(), "ok")
            before = (out / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                harness.run(manifest, out)
            self.assertEqual((out / "report.json").read_bytes(), before)

    def test_input_mutation_invalidates_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.txt"
            source.write_text("original")
            manifest = self.manifest()
            manifest["inputs"] = [str(source)]
            manifest["control"].insert(0, [sys.executable, "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('changed')", str(source)])
            report = harness.run(manifest, Path(directory) / "out")
            self.assertFalse(report["inputs_unchanged"])
            self.assertFalse(report["valid"])

    def test_cli_exit_code_and_saved_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest()
            manifest["seeds"] = [11]
            manifest["variant"] = [[sys.executable, "-c", "import sys; sys.exit(2)", "{seed}"]]
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            out = Path(directory) / "out"
            result = subprocess.run([sys.executable, str(Path(harness.__file__)), str(path), "--out", str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads((out / "report.json").read_text())["valid"])


if __name__ == "__main__":
    unittest.main()
