//! Small, deterministic feature measurements built on the verification benchmark
//! command.  The experiments intentionally use the public scalar/reference APIs:
//! they are not a second training harness and do not alter default PSSA behavior.

use crate::adapter::PlasticAdapterV2;
use crate::memory::HyperbolicEpisodicBankV2;
use crate::pssa::{PSSAConfigV2, PSSALayerV2};
use serde_json::{Value, json};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_OUTPUT: &str = "__agent__/feature_results";
const REPRO_ROOT: &str = "/workspace/oxide-ai";

fn benchmark_command(feature: &str) -> String {
    format!(
        "cd {REPRO_ROOT} && cargo run --release -- benchmark --feature {feature} --out {DEFAULT_OUTPUT}"
    )
}

fn write_json(dir: &Path, name: &str, record: &Value) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{name}.json"));
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| format!("encode {name}: {e}"))?;
    fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}

fn tiny_config(memory: usize, ema_alpha: f32) -> PSSAConfigV2 {
    PSSAConfigV2 {
        depth: 1,
        d_vocab: 64,
        d_latent: 12,
        d_state: 3,
        d_mem_key: 6,
        mem_capacity: memory,
        chunk_len: 8,
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.99,
        weight_decay: 0.01,
        eps: 1e-8,
        tau_mem: 0.12,
        ema_alpha,
    }
}

fn config_json(c: &PSSAConfigV2) -> Value {
    json!({
        "depth": c.depth,
        "d_vocab": c.d_vocab,
        "d_latent": c.d_latent,
        "d_state": c.d_state,
        "d_mem_key": c.d_mem_key,
        "mem_capacity": c.mem_capacity,
        "chunk_len": c.chunk_len,
        "lr": c.lr,
        "beta1": c.beta1,
        "beta2": c.beta2,
        "weight_decay": c.weight_decay,
        "eps": c.eps,
        "tau_mem": c.tau_mem,
        "ema_alpha": c.ema_alpha,
    })
}

fn task_a() -> (Vec<usize>, Vec<usize>) {
    let inputs = vec![0, 1, 2, 0, 1, 2, 0, 1];
    let targets = vec![1, 2, 0, 1, 2, 0, 1, 2];
    (inputs, targets)
}

fn task_b() -> (Vec<usize>, Vec<usize>) {
    let inputs = vec![3, 4, 5, 3, 4, 5, 3, 4];
    let targets = vec![4, 5, 3, 4, 5, 3, 4, 5];
    (inputs, targets)
}

fn eval_loss(model: &mut PSSALayerV2, inputs: &[usize], targets: &[usize]) -> f32 {
    // Evaluation must not become an extra training prefix.  Save the carry,
    // score from a clean boundary, then restore the carry that training had
    // established before this measurement.
    let mut carries = Vec::with_capacity(1 + model.extra_blocks.len());
    for block in std::iter::once(&model.block).chain(&model.extra_blocks) {
        carries.push(block.h_persistent.clone());
    }
    model.reset_recurrent_state();
    let loss = model.forward_train_chunk(inputs, targets);
    for (block, carry) in std::iter::once(&mut model.block)
        .chain(&mut model.extra_blocks)
        .zip(carries)
    {
        block.h_persistent = carry;
    }
    loss
}

fn l2_norm(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
}

fn effective_up(adapter: &PlasticAdapterV2) -> Vec<f32> {
    adapter
        .up_proj
        .data
        .iter()
        .zip(&adapter.consolidated_up)
        .map(|(fast, slow)| fast + slow)
        .collect()
}

fn train_epoch(
    model: &mut PSSALayerV2,
    inputs: &[usize],
    targets: &[usize],
    memory_enabled: bool,
    consolidate: bool,
) -> Value {
    // Each task is one document containing one chunk. Repeating an epoch is
    // a document restart, not a continuation (A ends in 1 but restarts at 0).
    // Match the production document selector, including on repeated epochs.
    model.reset_recurrent_state();
    let memory_slots_before = model.block.memory.count;
    let loss = model.forward_train_chunk(inputs, targets);
    let memory_injection_l2 = l2_norm(&model.block.tape.m_inj[..inputs.len() * model.cfg.d_latent]);
    model.zero_gradients();
    model.backward_chunk(inputs.len(), 1.0);
    // Retrieval and its adjoint must see the same bank. The production helper
    // owns the loss > 3.5 and refractory gates; never force a benchmark write.
    if memory_enabled {
        model.insert_training_memory(loss, inputs.len());
    }
    model.apply_adamw(model.cfg.lr);
    // One update per epoch here, so production's end-of-epoch consolidation
    // occurs after each update. It is a coefficient transfer, not a ridge solve.
    let mut consolidation_max_effective_change = 0.0f32;
    if consolidate {
        let before = effective_up(&model.block.adapters[0]);
        model.ema_consolidate_plasticity();
        consolidation_max_effective_change = before
            .iter()
            .zip(effective_up(&model.block.adapters[0]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
    }
    json!({
        "optimizer_step": model.step_counter,
        "training_loss": loss,
        "memory_write_api_called": memory_enabled,
        "memory_slots_before": memory_slots_before,
        "memory_slots_after": model.block.memory.count,
        "memory_injection_l2": memory_injection_l2,
        "consolidation_called": consolidate,
        "consolidation_max_effective_change": consolidation_max_effective_change,
    })
}

fn adapter_state(model: &PSSALayerV2) -> Value {
    let adapter = &model.block.adapters[0];
    json!({
        "fast_up_l2": l2_norm(&adapter.up_proj.data),
        "consolidated_up_l2": l2_norm(&adapter.consolidated_up),
        "effective_up_l2": l2_norm(&effective_up(adapter)),
    })
}

fn loss_metrics(before: f32, after: f32) -> Value {
    // No denominator floor or clamp in the primary measurements. A zero
    // denominator has no defined ratio; serialize null instead of inventing one.
    let before = before as f64;
    let after = after as f64;
    let ratio = (before > 0.0).then(|| after / before);
    json!({
        "a_loss_increase": after - before,
        "loss_ratio_after_over_before": ratio,
        "retention_ratio": (after > 0.0).then(|| before / after),
        // Retain the old score explicitly for comparison with historical files.
        "legacy_clamped_retention_score": ratio.map(|r| (2.0 - r).clamp(0.0, 1.0)),
    })
}

fn continual_variant(
    name: &str,
    memory_enabled: bool,
    consolidate: bool,
    seed: u64,
    epochs_per_task: usize,
    cfg: &PSSAConfigV2,
) -> Value {
    let (a_inputs, a_targets) = task_a();
    let (b_inputs, b_targets) = task_b();
    let mut model = PSSALayerV2::new(cfg.clone(), seed);
    let initial_a_loss = eval_loss(&mut model, &a_inputs, &a_targets);
    let a_epochs: Vec<_> = (0..epochs_per_task)
        .map(|_| {
            train_epoch(
                &mut model,
                &a_inputs,
                &a_targets,
                memory_enabled,
                consolidate,
            )
        })
        .collect();
    let memory_slots_after_a = model.block.memory.count;
    let adapter_after_a = adapter_state(&model);
    let a_loss_before_b = eval_loss(&mut model, &a_inputs, &a_targets);
    let b_loss_before_b = eval_loss(&mut model, &b_inputs, &b_targets);
    // Bank, parameters, slow copy and Adam moments survive the task boundary;
    // only the document's recurrent carry resets in train_epoch.
    let b_epochs: Vec<_> = (0..epochs_per_task)
        .map(|_| {
            train_epoch(
                &mut model,
                &b_inputs,
                &b_targets,
                memory_enabled,
                consolidate,
            )
        })
        .collect();
    let a_loss_after_b = eval_loss(&mut model, &a_inputs, &a_targets);
    let b_loss_after_b = eval_loss(&mut model, &b_inputs, &b_targets);
    let consolidation_calls = a_epochs
        .iter()
        .chain(&b_epochs)
        .filter(|e| e["consolidation_called"] == true)
        .count();
    let mut record = json!({
        "variant": name,
        "memory_bank_enabled": memory_enabled,
        "consolidation_enabled": consolidate,
        "seed": seed,
        "epochs_per_task": epochs_per_task,
        "a_loss_initial": initial_a_loss,
        "a_loss_before_b": a_loss_before_b,
        "a_loss_after_b": a_loss_after_b,
        "b_loss_before_b": b_loss_before_b,
        "b_loss_after_b": b_loss_after_b,
        "memory_slots_after_a": memory_slots_after_a,
        "memory_slots_used": model.block.memory.count,
        "consolidation_calls": consolidation_calls,
        "adapter_after_a": adapter_after_a,
        "adapter_after_b": adapter_state(&model),
        "task_a_epochs": a_epochs,
        "task_b_epochs": b_epochs,
        "optimizer_steps": model.step_counter,
    });
    record.as_object_mut().unwrap().extend(
        loss_metrics(a_loss_before_b, a_loss_after_b)
            .as_object()
            .unwrap()
            .clone(),
    );
    record
}

fn continual(dir: &Path) -> Result<(), String> {
    // Preserve the configuration already present when this harness was audited.
    // Unlike the stale vocab=8 / weight_decay=0 record, vocab=64 can naturally
    // cross the production write threshold and AdamW can distinguish fast/slow.
    let cfg = tiny_config(64, 0.20);
    let seed = 4101;
    let epochs_per_task = 18;
    let record = json!({
        "schema_version": 2,
        "experiment": "continual_learning_catastrophic_forgetting",
        "reproduce_command": benchmark_command("continual"),
        "protocol": {
            "task_a_inputs": task_a().0,
            "task_a_targets": task_a().1,
            "task_b_inputs": task_b().0,
            "task_b_targets": task_b().1,
            "task_order": ["A", "B"],
            "epochs_per_task": epochs_per_task,
            "updates_per_epoch": 1,
            "target_tokens_per_task": epochs_per_task * task_a().1.len(),
            "seed": seed,
            "recurrent_state_policy": "reset at every one-document epoch and evaluation; evaluation restores training carry",
            "memory_policy": "empty initial bank; enabled calls insert_training_memory after backward, before AdamW; production loss > 3.5 and refractory gates unchanged; disabled skips writes so retrieval stays empty",
            "consolidation_policy": "enabled calls ema_consolidate_plasticity after every one-update epoch; disabled skips it; fast+slow is preserved, only fast is AdamW-decayed; no ridge regression is implemented by this API",
            "primary_metrics": ["a_loss_before_b", "a_loss_after_b", "a_loss_increase", "loss_ratio_after_over_before"],
            "loss_retention_definition": "v2: retention_ratio=A_before_B/A_after_B, unclamped reciprocal loss ratio, not fraction of knowledge retained; >1 means improvement; zero denominator yields null",
            "legacy_score_definition": "legacy_clamped_retention_score=clamp(2-A_after_B/A_before_B,0,1); zero means loss at least doubled, not total forgetting",
            "limitation": "single-seed synthetic training-document probe, not held-out evidence of general continual-learning ability",
            "model": config_json(&cfg),
        },
        "variants": [
            continual_variant("memory_and_consolidation", true, true, seed, epochs_per_task, &cfg),
            continual_variant("memory_disabled", false, true, seed, epochs_per_task, &cfg),
            continual_variant("consolidation_disabled", true, false, seed, epochs_per_task, &cfg),
            continual_variant("memory_and_consolidation_disabled", false, false, seed, epochs_per_task, &cfg),
        ],
    });
    write_json(dir, "continual_learning", &record)
}

fn projected(raw: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; raw.len()];
    HyperbolicEpisodicBankV2::diffeomorphic_project(raw, &mut out);
    out
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn episodic_retention(dir: &Path) -> Result<(), String> {
    let capacities = [64usize, 256, 1024];
    let gaps = [0usize, 8, 32, 128, 512, 2048];
    let dim_key = 4;
    let dim_val = 2;
    let tau = 0.05f32;
    let fact_key = projected(&[1.0, 0.0, 0.0, 0.0]);
    let mut measurements = Vec::new();
    for &capacity in &capacities {
        for &gap in &gaps {
            let mut bank = HyperbolicEpisodicBankV2::new(capacity, dim_key, dim_val);
            bank.insert(&fact_key, &[1.0, 0.0]);
            for i in 0..gap {
                let raw = [0.0, 1.0 + (i % 17) as f32 * 0.001, 0.01, 0.0];
                bank.insert(&projected(&raw), &[0.0, 1.0]);
            }
            let mut value = vec![0.0; dim_val];
            let mut weights = vec![0.0; capacity];
            let min_distance = bank.retrieve_soft_into(&fact_key, tau, &mut value, &mut weights);
            let fact_probability = value[0].clamp(1e-12, 1.0);
            measurements.push(json!({
                "bank_capacity": capacity,
                "distractor_gap": gap,
                "tau": tau,
                "fact_present": bank.values[..bank.count * dim_val]
                    .chunks_exact(dim_val)
                    .any(|v| v == [1.0, 0.0]),
                "slots_after_insertion": bank.count,
                "write_head": bank.write_head,
                "min_poincare_distance": min_distance,
                "retrieved_value": value,
                "recall_accuracy": if argmax(&value) == 0 { 1.0 } else { 0.0 },
                "cross_entropy_loss": -fact_probability.ln(),
            }));
        }
    }
    let record = json!({
        "experiment": "episodic_memory_retention_vs_distractor_gap",
        "reproduce_command": benchmark_command("retention"),
        "protocol": {
            "bank_capacities": capacities,
            "distractor_gaps": gaps,
            "dim_key": dim_key,
            "dim_value": dim_val,
            "tau": tau,
            "fact_value": [1.0, 0.0],
            "distractor_value": [0.0, 1.0],
            "key_projection": "q/(1+||q||), open-ball bounded",
            "write_order": "fact first, then distractors in ascending gap order",
        },
        "measurements": measurements,
    });
    write_json(dir, "episodic_retention", &record)
}

fn cross_entropy(value: &[f32], target: usize) -> f32 {
    -value[target].clamp(1e-12, 1.0).ln()
}

fn hyperbolic_vs_euclidean(dir: &Path) -> Result<(), String> {
    // Two facts deliberately share a Euclidean direction but have different
    // radii. Cosine cannot separate that pair; Poincare distance can.
    let raw_keys = [
        [0.20, 0.0, 0.0, 0.0],
        [0.70, 0.0, 0.0, 0.0],
        [0.0, 0.40, 0.0, 0.0],
        [0.0, 0.0, 0.80, 0.0],
    ];
    let dim_key = 4;
    let classes = raw_keys.len();
    let tau = 0.10f32;
    let mut bank = HyperbolicEpisodicBankV2::new(classes, dim_key, classes);
    let keys: Vec<Vec<f32>> = raw_keys.iter().map(|k| projected(k)).collect();
    for (i, key) in keys.iter().enumerate() {
        let mut value = vec![0.0; classes];
        value[i] = 1.0;
        bank.insert(key, &value);
    }
    let mut hyper_correct = 0usize;
    let mut euclidean_correct = 0usize;
    let mut hyper_loss = 0.0;
    let mut euclidean_loss = 0.0;
    let mut rows = Vec::new();
    for (target, query) in keys.iter().enumerate() {
        let mut h_value = vec![0.0; classes];
        let mut h_weights = vec![0.0; classes];
        let h_distance = bank.retrieve_soft_into(query, tau, &mut h_value, &mut h_weights);
        let mut e_value = vec![0.0; classes];
        let mut e_weights = vec![0.0; classes];
        let e_distance =
            bank.retrieve_soft_euclidean_into(query, tau, &mut e_value, &mut e_weights);
        let h_pred = argmax(&h_value);
        let e_pred = argmax(&e_value);
        hyper_correct += usize::from(h_pred == target);
        euclidean_correct += usize::from(e_pred == target);
        let h_ce = cross_entropy(&h_value, target);
        let e_ce = cross_entropy(&e_value, target);
        hyper_loss += h_ce;
        euclidean_loss += e_ce;
        rows.push(json!({
            "target": target,
            "hyperbolic_prediction": h_pred,
            "euclidean_cosine_prediction": e_pred,
            "hyperbolic_value": h_value,
            "euclidean_cosine_value": e_value,
            "hyperbolic_min_distance": h_distance,
            "euclidean_cosine_min_distance": e_distance,
            "hyperbolic_cross_entropy": h_ce,
            "euclidean_cosine_cross_entropy": e_ce,
        }));
    }
    let record = json!({
        "experiment": "hyperbolic_vs_euclidean_read",
        "reproduce_command": benchmark_command("geometry"),
        "protocol": {
            "stored_key_coordinates": "Poincare coordinates produced by diffeomorphic_project",
            "euclidean_control": "cosine distance on those stored coordinates; opt-in helper only",
            "tau": tau,
            "dim_key": dim_key,
            "number_of_facts": classes,
            "raw_keys": raw_keys,
            "values": "one-hot class vectors",
        },
        "aggregate": {
            "hyperbolic_accuracy": hyper_correct as f32 / classes as f32,
            "euclidean_cosine_accuracy": euclidean_correct as f32 / classes as f32,
            "hyperbolic_loss": hyper_loss / classes as f32,
            "euclidean_cosine_loss": euclidean_loss / classes as f32,
        },
        "measurements": rows,
    });
    write_json(dir, "hyperbolic_vs_euclidean", &record)
}

fn refractory_variant(enabled: bool, updates: usize, surprise: f32) -> Value {
    let key = projected(&[0.4]);
    let contradictory = projected(&[-0.4]);
    let mut bank = HyperbolicEpisodicBankV2::new(1, 1, 1);
    bank.insert(&key, &[1.0]);
    let initial_value = bank.values[0];
    let mut defended = 0usize;
    let mut overwritten = 0usize;
    for step in 1..=updates {
        if enabled {
            match bank.insert_protected(&contradictory, &[-1.0], surprise, step) {
                Some(_) => overwritten += 1,
                None => defended += 1,
            }
        } else {
            bank.insert(&contradictory, &[-1.0]);
            overwritten += 1;
        }
    }
    let final_value = bank.values[0];
    json!({
        "refractory_enabled": enabled,
        "initial_value": initial_value,
        "final_value": final_value,
        "value_degradation": (initial_value - final_value).abs(),
        "normalized_value_retention": (final_value / initial_value).max(0.0),
        "final_confidence": bank.confidence[0],
        "defended_updates": defended,
        "overwritten_updates": overwritten,
        "updates": updates,
        "surprise": surprise,
        "capacity": 1,
        "current_step_spacing": 1,
    })
}

fn refractory(dir: &Path) -> Result<(), String> {
    let updates = 256;
    let surprise = 2.5;
    let record = json!({
        "experiment": "refractory_protection_contradiction_spam",
        "reproduce_command": benchmark_command("refractory"),
        "protocol": {
            "updates": updates,
            "surprise": surprise,
            "capacity": 1,
            "initial_value": 1.0,
            "contradictory_value": -1.0,
            "step_spacing": 1,
            "gate": "RateLimiterGate::apply_refractory_overwrite",
        },
        "variants": [refractory_variant(true, updates, surprise), refractory_variant(false, updates, surprise)],
    });
    write_json(dir, "refractory_protection", &record)
}

fn l2_loss(actual: &[f32], target: &[f32]) -> f32 {
    actual
        .iter()
        .zip(target)
        .map(|(a, t)| (a - t) * (a - t))
        .sum::<f32>()
        / actual.len() as f32
}

fn adapter_output(adapter: &PlasticAdapterV2, input: &[f32]) -> Vec<f32> {
    let mut activation = vec![0.0; adapter.rank];
    let mut output = vec![0.0; adapter.d_latent];
    adapter.forward_into(input, &mut activation, &mut output);
    output
}

fn vector_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn ridge_consolidation(dir: &Path) -> Result<(), String> {
    let mut cfg = tiny_config(8, 1.0);
    cfg.d_vocab = 4;
    cfg.d_latent = 8;
    cfg.d_state = 2;
    cfg.d_mem_key = 3;
    cfg.ema_alpha = 1.0;
    let mut model = PSSALayerV2::new(cfg.clone(), 9917);
    let input = vec![0.4, -0.2, 0.7, 0.1, -0.5, 0.3, 0.9, -0.8];
    let target = vec![0.20, -0.10, 0.05, 0.30, -0.25, 0.15, 0.10, -0.05];
    for (i, weight) in model.block.adapters[0].up_proj.data.iter_mut().enumerate() {
        *weight = ((i * 17 % 23) as f32 - 11.0) * 0.006;
    }
    let before_output = adapter_output(&model.block.adapters[0], &input);
    let loss_before = l2_loss(&before_output, &target);
    let fast_norm_before = model.block.adapters[0]
        .up_proj
        .data
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    model.ema_consolidate_plasticity();
    let after_consolidation_output = adapter_output(&model.block.adapters[0], &input);
    let loss_after_consolidation = l2_loss(&after_consolidation_output, &target);
    let effective_change = vector_l2(&before_output, &after_consolidation_output);
    let consolidated_norm = model.block.adapters[0]
        .consolidated_up
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    model.block.adapters[0].up_proj.data.fill(0.0);
    let after_clear_output = adapter_output(&model.block.adapters[0], &input);
    let loss_after_fast_clear = l2_loss(&after_clear_output, &target);
    let clear_output_change = vector_l2(&before_output, &after_clear_output);
    let record = json!({
        "experiment": "ridge_consolidation_fast_weight_fold",
        "reproduce_command": benchmark_command("ridge"),
        "protocol": {
            "model": config_json(&cfg),
            "seed": 9917,
            "input": input,
            "target": target,
            "consolidation_api": "PSSALayerV2::ema_consolidate_plasticity -> PlasticAdapterV2::consolidate",
            "consolidation_alpha": cfg.ema_alpha,
            "clear_operation": "up_proj.data.fill(0.0)",
            "effective_weight_definition": "up_proj.fast + consolidated_up",
        },
        "measurements": {
            "loss_before_consolidation": loss_before,
            "loss_after_consolidation": loss_after_consolidation,
            "loss_after_fast_state_clear": loss_after_fast_clear,
            "fast_weight_l2_before": fast_norm_before,
            "consolidated_weight_l2_after": consolidated_norm,
            "effective_output_change_after_consolidation": effective_change,
            "output_change_after_fast_clear": clear_output_change,
            "fast_state_cleared": model.block.adapters[0].up_proj.data.iter().all(|x| *x == 0.0),
            "clearing_preserved_behavior": clear_output_change < 1e-6,
        },
    });
    write_json(dir, "ridge_consolidation", &record)
}

fn read_json(dir: &Path, name: &str) -> Option<Value> {
    let bytes = fs::read(dir.join(format!("{name}.json"))).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn summary(dir: &Path) -> Result<(), String> {
    let mut out = String::from("# PSSA feature benchmark results\n\n");
    out.push_str("Generated by the existing `benchmark` command extension. Values below are read from the JSON records.\n\n");
    out.push_str(
        "| Experiment | Headline measured numbers | Reproduction command |\n|---|---|---|\n",
    );
    if let Some(v) = read_json(dir, "continual_learning") {
        let mut cells = Vec::new();
        if let Some(rows) = v["variants"].as_array() {
            for row in rows {
                cells.push(format!(
                    "{}: A {:.9} -> {:.9}, increase {:+.9}, loss ratio {:.9}, inverse-loss retention {:.9}; slots {} -> {}, consolidation calls {}",
                    row["variant"].as_str().unwrap_or("?"),
                    row["a_loss_before_b"].as_f64().unwrap_or(f64::NAN),
                    row["a_loss_after_b"].as_f64().unwrap_or(f64::NAN),
                    row["a_loss_increase"].as_f64().unwrap_or(f64::NAN),
                    row["loss_ratio_after_over_before"].as_f64().unwrap_or(f64::NAN),
                    row["retention_ratio"].as_f64().unwrap_or(f64::NAN),
                    row["memory_slots_after_a"],
                    row["memory_slots_used"],
                    row["consolidation_calls"]
                ));
            }
        }
        writeln!(
            out,
            "| continual learning | {} | `{}` |",
            cells.join("; "),
            v["reproduce_command"]
        )
        .unwrap();
    }
    if let Some(v) = read_json(dir, "episodic_retention") {
        let mut cells = Vec::new();
        if let Some(rows) = v["measurements"].as_array() {
            for capacity in [64, 256, 1024] {
                let vals: Vec<String> = rows
                    .iter()
                    .filter(|r| r["bank_capacity"] == capacity)
                    .map(|r| {
                        format!(
                            "{}:{:.0}",
                            r["distractor_gap"],
                            r["recall_accuracy"].as_f64().unwrap_or(0.0)
                        )
                    })
                    .collect();
                cells.push(format!("cap {} [{}]", capacity, vals.join(", ")));
            }
        }
        writeln!(
            out,
            "| episodic retention | {} (gap:accuracy) | `{}` |",
            cells.join("; "),
            v["reproduce_command"]
        )
        .unwrap();
    }
    if let Some(v) = read_json(dir, "hyperbolic_vs_euclidean") {
        let a = &v["aggregate"];
        writeln!(out, "| hyperbolic vs Euclidean | hyperbolic acc {:.3}, loss {:.4}; cosine acc {:.3}, loss {:.4} | `{}` |", a["hyperbolic_accuracy"], a["hyperbolic_loss"], a["euclidean_cosine_accuracy"], a["euclidean_cosine_loss"], v["reproduce_command"]).unwrap();
    }
    if let Some(v) = read_json(dir, "refractory_protection") {
        let rows = v["variants"].as_array().cloned().unwrap_or_default();
        let text: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "{} {:.4} -> {:.4} (degradation {:.4})",
                    if r["refractory_enabled"].as_bool().unwrap_or(false) {
                        "on"
                    } else {
                        "off"
                    },
                    r["initial_value"],
                    r["final_value"],
                    r["value_degradation"]
                )
            })
            .collect();
        writeln!(
            out,
            "| refractory protection | {} | `{}` |",
            text.join("; "),
            v["reproduce_command"]
        )
        .unwrap();
    }
    if let Some(v) = read_json(dir, "ridge_consolidation") {
        let m = &v["measurements"];
        writeln!(out, "| ridge consolidation | loss {:.6} -> {:.6} -> {:.6} (fast clear preserved: {}) | `{}` |", m["loss_before_consolidation"], m["loss_after_consolidation"], m["loss_after_fast_state_clear"], m["clearing_preserved_behavior"], v["reproduce_command"]).unwrap();
    }
    if let Some(v) = read_json(dir, "continual_learning") {
        writeln!(
            out,
            "\n## Continual-learning interpretation\n\n{}\n\n{}\n\n{}\n\n{}",
            v["protocol"]["loss_retention_definition"]
                .as_str()
                .unwrap_or("Legacy result: see its JSON metric definition."),
            v["protocol"]["legacy_score_definition"]
                .as_str()
                .unwrap_or(""),
            v["protocol"]["consolidation_policy"].as_str().unwrap_or(""),
            v["protocol"]["limitation"].as_str().unwrap_or("")
        )
        .unwrap();
    }
    fs::write(dir.join("summary.md"), out).map_err(|e| format!("write summary: {e}"))
}

/// Run one feature or all five and write records to the requested directory.
pub fn run(feature: &str, output: Option<&str>) -> Result<(), String> {
    let dir = PathBuf::from(output.unwrap_or(DEFAULT_OUTPUT));
    match feature {
        "continual" | "continual_learning" => continual(&dir)?,
        "retention" | "episodic_retention" => episodic_retention(&dir)?,
        "geometry" | "hyperbolic_vs_euclidean" => hyperbolic_vs_euclidean(&dir)?,
        "refractory" | "refractory_protection" => refractory(&dir)?,
        "ridge" | "ridge_consolidation" => ridge_consolidation(&dir)?,
        "all" => {
            continual(&dir)?;
            episodic_retention(&dir)?;
            hyperbolic_vs_euclidean(&dir)?;
            refractory(&dir)?;
            ridge_consolidation(&dir)?;
        }
        other => {
            return Err(format!(
                "unknown benchmark feature '{other}' (try continual, retention, geometry, refractory, ridge, or all)"
            ));
        }
    }
    summary(&dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defense::RateLimiterGate;
    use crate::linalg::SimpleRng;

    #[test]
    fn continual_metrics_do_not_saturate_or_invent_zero_denominators() {
        let measured = loss_metrics(0.98475116, 2.3483992);
        assert_eq!(measured["legacy_clamped_retention_score"], 0.0);
        assert!(measured["loss_ratio_after_over_before"].as_f64().unwrap() > 2.38);
        assert!(measured["retention_ratio"].as_f64().unwrap() > 0.41);
        assert!(
            loss_metrics(1.0, 3.0)["retention_ratio"] != loss_metrics(1.0, 4.0)["retention_ratio"]
        );
        assert_eq!(loss_metrics(1.0, 0.5)["retention_ratio"], 2.0);
        assert!(loss_metrics(0.0, 1.0)["loss_ratio_after_over_before"].is_null());
        assert!(loss_metrics(1.0, 0.0)["retention_ratio"].is_null());
    }

    #[test]
    fn continual_epoch_matches_production_document_reset_and_update_order() {
        let cfg = tiny_config(64, 0.2);
        let mut actual = PSSALayerV2::new(cfg.clone(), 4101);
        let mut reference = PSSALayerV2::new(cfg, 4101);
        let (inputs, targets) = task_a();
        // Repeated epochs must ignore the previous document's carry.
        actual.block.h_persistent.fill(99.0);
        for _ in 0..3 {
            reference.reset_recurrent_state();
            let loss = reference.forward_train_chunk(&inputs, &targets);
            reference.zero_gradients();
            reference.backward_chunk(inputs.len(), 1.0);
            reference.insert_training_memory(loss, inputs.len());
            reference.apply_adamw(reference.cfg.lr);
            reference.ema_consolidate_plasticity();
            let audit = train_epoch(&mut actual, &inputs, &targets, true, true);
            assert_eq!(audit["training_loss"], json!(loss));
            assert_eq!(actual.block.memory, reference.block.memory);
            assert_eq!(actual.block.h_persistent, reference.block.h_persistent);
            assert_eq!(actual.embed_w, reference.embed_w);
            assert_eq!(actual.unembed_w, reference.unembed_w);
            assert_eq!(actual.block.w_qx, reference.block.w_qx);
            assert_eq!(actual.block.w_qh, reference.block.w_qh);
            assert_eq!(actual.block.adapters, reference.block.adapters);
        }
        assert!(actual.block.memory.count > 0);
        assert_eq!(actual.step_counter, 3);
    }

    #[test]
    fn continual_evaluation_restores_carry_without_learning_or_writing_memory() {
        let mut cfg = tiny_config(64, 0.2);
        cfg.depth = 2;
        let mut model = PSSALayerV2::new(cfg, 4101);
        let (inputs, targets) = task_a();
        train_epoch(&mut model, &inputs, &targets, true, true);
        let before: Vec<_> = std::iter::once(&model.block)
            .chain(&model.extra_blocks)
            .map(|b| (b.h_persistent.clone(), b.memory.clone(), b.adapters.clone()))
            .collect();
        let embedding = model.embed_w.clone();
        let step = model.step_counter;
        let first = eval_loss(&mut model, &inputs, &targets);
        assert_eq!(first, eval_loss(&mut model, &inputs, &targets));
        for (b, (carry, bank, adapters)) in std::iter::once(&model.block)
            .chain(&model.extra_blocks)
            .zip(before)
        {
            assert_eq!(b.h_persistent, carry);
            assert_eq!(b.memory, bank);
            assert_eq!(b.adapters, adapters);
        }
        assert_eq!(model.embed_w, embedding);
        assert_eq!(model.step_counter, step);
    }

    #[test]
    fn continual_memory_does_not_bypass_the_production_loss_gate() {
        let mut cfg = tiny_config(64, 0.2);
        cfg.d_vocab = 8;
        let mut model = PSSALayerV2::new(cfg, 4101);
        let (inputs, targets) = task_a();
        let audit = train_epoch(&mut model, &inputs, &targets, true, true);
        assert!(audit["training_loss"].as_f64().unwrap() < 3.5);
        assert_eq!(audit["memory_write_api_called"], true);
        assert_eq!(audit["memory_slots_after"], 0);
        assert_eq!(audit["memory_injection_l2"], 0.0);
    }

    #[test]
    fn continual_four_controls_exercise_the_features_without_assuming_a_winner() {
        let cfg = tiny_config(64, 0.2);
        let mut losses = Vec::new();
        for (memory, consolidate) in [(true, true), (false, true), (true, false), (false, false)] {
            let v = continual_variant("test", memory, consolidate, 4101, 18, &cfg);
            assert_eq!(v["optimizer_steps"], 36);
            assert_eq!(v["consolidation_calls"], if consolidate { 36 } else { 0 });
            for phase in ["adapter_after_a", "adapter_after_b"] {
                assert_eq!(
                    v[phase]["consolidated_up_l2"].as_f64().unwrap() > 0.0,
                    consolidate
                );
            }
            assert_eq!(v["memory_slots_after_a"].as_u64().unwrap() > 0, memory);
            assert_eq!(v["memory_slots_used"].as_u64().unwrap() > 0, memory);
            let mut retrieved = false;
            for (i, epoch) in v["task_a_epochs"]
                .as_array()
                .unwrap()
                .iter()
                .chain(v["task_b_epochs"].as_array().unwrap())
                .enumerate()
            {
                assert_eq!(epoch["optimizer_step"], i + 1);
                assert_eq!(epoch["memory_write_api_called"], memory);
                assert_eq!(epoch["consolidation_called"], consolidate);
                assert!(
                    epoch["consolidation_max_effective_change"]
                        .as_f64()
                        .unwrap()
                        < 1e-6
                );
                let before = epoch["memory_slots_before"].as_u64().unwrap();
                let after = epoch["memory_slots_after"].as_u64().unwrap();
                // Capacity exceeds all 36 updates, so every accepted write is
                // visible as one additional slot and no overwrite is possible.
                assert_eq!(
                    after - before,
                    u64::from(memory && epoch["training_loss"].as_f64().unwrap() > 3.5)
                );
                retrieved |= epoch["memory_injection_l2"].as_f64().unwrap() > 0.0;
            }
            assert_eq!(retrieved, memory);
            losses.push(v["a_loss_after_b"].as_f64().unwrap());
        }
        for i in 0..losses.len() {
            for j in 0..i {
                assert_ne!(losses[i].to_bits(), losses[j].to_bits(), "{losses:?}");
            }
        }
    }

    #[test]
    fn zero_decay_consolidation_changes_storage_not_the_learning_rule() {
        let mut cfg = tiny_config(64, 0.2);
        cfg.weight_decay = 0.0;
        for memory in [false, true] {
            let on = continual_variant("on", memory, true, 4101, 18, &cfg);
            let off = continual_variant("off", memory, false, 4101, 18, &cfg);
            assert!(
                on["adapter_after_b"]["consolidated_up_l2"]
                    .as_f64()
                    .unwrap()
                    > 0.0
            );
            assert_eq!(off["adapter_after_b"]["consolidated_up_l2"], 0.0);
            // Fast+slow and both forward/backward rules are invariant when
            // weight decay is zero. Only floating-point regrouping differs.
            for field in ["a_loss_before_b", "a_loss_after_b", "b_loss_after_b"] {
                let delta = (on[field].as_f64().unwrap() - off[field].as_f64().unwrap()).abs();
                assert!(delta < 1e-5, "{field}: {delta}");
            }
        }
    }

    #[test]
    fn euclidean_control_sees_cosine_tie_but_poincare_sees_radius() {
        let mut bank = HyperbolicEpisodicBankV2::new(2, 1, 2);
        bank.insert(&projected(&[0.2]), &[1.0, 0.0]);
        bank.insert(&projected(&[0.7]), &[0.0, 1.0]);
        let query = projected(&[0.7]);
        let mut h = [0.0; 2];
        let mut hw = [0.0; 2];
        bank.retrieve_soft_into(&query, 0.1, &mut h, &mut hw);
        let mut e = [0.0; 2];
        let mut ew = [0.0; 2];
        bank.retrieve_soft_euclidean_into(&query, 0.1, &mut e, &mut ew);
        assert_eq!(argmax(&h), 1);
        assert!(e[0] > 0.0 && e[1] > 0.0);
    }

    #[test]
    fn full_consolidation_preserves_adapter_output_when_fast_state_is_cleared() {
        let mut rng = SimpleRng::new(17);
        let mut adapter = PlasticAdapterV2::new(4, 2, &mut rng);
        for (i, x) in adapter.up_proj.data.iter_mut().enumerate() {
            *x = (i as f32 - 3.0) * 0.02;
        }
        let input = [0.3, -0.5, 0.2, 0.7];
        let before = adapter_output(&adapter, &input);
        adapter.consolidate(1.0);
        adapter.up_proj.data.fill(0.0);
        let after = adapter_output(&adapter, &input);
        assert!(vector_l2(&before, &after) < 1e-6);
    }

    #[test]
    fn refractory_benchmark_distinguishes_spam_controls() {
        let on = refractory_variant(true, 64, 2.5);
        let off = refractory_variant(false, 64, 2.5);
        assert!(
            on["value_degradation"].as_f64().unwrap() < off["value_degradation"].as_f64().unwrap()
        );
        assert_eq!(off["overwritten_updates"], 64);
        let _ = RateLimiterGate::compute_refractory_gain(1, 60.0);
    }
}
