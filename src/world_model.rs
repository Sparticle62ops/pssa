//! Opt-in, non-biological boxes-world experiment. Not an actor/critic agent.
//!
//! A continuous PSSA block supplies learned deterministic recurrence. A learned
//! categorical posterior observes the current frame; its straight-through sample
//! conditions the next transition. A learned categorical prior never sees the
//! next frame. Training combines reconstruction, KL(q || p), binary reward and
//! continuation losses. Imagination samples only the prior. No diffusion,
//! episodic writes, replay, adapter consolidation or checkpoint changes are used.
//!
//! The small full-observation deterministic task does not require stochasticity.
//! This is an RSSM-style prototype, not a reproduction of a full Dreamer system.
//! Straight-through gradients are biased. Recurrent gradients are NOT frozen or
//! detached between frames: prefix VJPs implement the categorical feedback graph
//! exactly for that estimator, at O(sequence_length^2) cost. Episode boundaries
//! are detached. The plain autoregressive control receives the same transitions,
//! updates and hidden/state widths, not the same parameter count or wall time.

use crate::linalg::{SimpleRng, sigmoid};
use crate::pssa::{PSSAContinuousBlockV2, PSSAContinuousConfigV2, ParamMatrix, ParamVector};
use std::collections::VecDeque;
use std::time::Instant;

const ACTIONS: usize = 5;
const FIELDS: usize = 3;
const MAX_HORIZON: usize = 24;

/// Small CPU-only defaults. `validate` must run before allocation or generation.
#[derive(Clone, Debug)]
pub struct WorldModelConfig {
    pub seed: u64,
    pub side: usize,
    pub hidden: usize,
    pub state: usize,
    /// One categorical variable, not a continuous autoregressive latent.
    pub categories: usize,
    pub train_episodes: usize,
    pub heldout_episodes: usize,
    pub horizon: usize,
    pub rollout_horizon: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub kl_weight: f32,
    pub auxiliary_weight: f32,
    pub gradient_clip: f32,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            seed: 73,
            side: 4,
            hidden: 16,
            state: 2,
            categories: 16,
            train_episodes: 32,
            heldout_episodes: 12,
            horizon: 12,
            rollout_horizon: 8,
            epochs: 3,
            learning_rate: 0.003,
            kl_weight: 0.1,
            auxiliary_weight: 0.25,
            gradient_clip: 5.0,
        }
    }
}

impl WorldModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value, low, high) in [
            ("side", self.side, 3, 6),
            ("hidden", self.hidden, 4, 48),
            ("state", self.state, 1, 8),
            ("categories", self.categories, 2, 32),
            ("train_episodes", self.train_episodes, 1, 128),
            ("heldout_episodes", self.heldout_episodes, 1, 64),
            ("horizon", self.horizon, 1, MAX_HORIZON),
            ("rollout_horizon", self.rollout_horizon, 1, MAX_HORIZON),
            ("epochs", self.epochs, 1, 12),
        ] {
            if !(low..=high).contains(&value) {
                return Err(format!("{name} must be in {low}..={high}"));
            }
        }
        if self.rollout_horizon > self.horizon {
            return Err("rollout_horizon must not exceed horizon".into());
        }
        for (name, value, low, high) in [
            ("learning_rate", self.learning_rate, 0.00001, 0.03),
            ("kl_weight", self.kl_weight, 0.001, 2.0),
            ("auxiliary_weight", self.auxiliary_weight, 0.0, 2.0),
            ("gradient_clip", self.gradient_clip, 0.1, 20.0),
        ] {
            if !value.is_finite() || !(low..=high).contains(&value) {
                return Err(format!("{name} must be finite and in {low}..={high}"));
            }
        }
        // All operands have already been bounded, so this cannot overflow.
        // Includes quadratic prefix backpropagation, both models, and state cost.
        let work = self.train_episodes
            * self.epochs
            * (self.horizon + 1).pow(2)
            * self.hidden
            * (self.hidden + self.state + self.categories);
        if work > 32_000_000 {
            return Err("requested recurrent training work exceeds the short CPU budget; reduce episodes, epochs, horizon or width".into());
        }
        Ok(())
    }

    fn cells(&self) -> usize {
        self.side * self.side
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxAction {
    Up,
    Down,
    Left,
    Right,
    Wait,
}

impl BoxAction {
    pub const ALL: [Self; ACTIONS] = [Self::Up, Self::Down, Self::Left, Self::Right, Self::Wait];

    fn index(self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
            Self::Wait => 4,
        }
    }
}

/// Row-major cell IDs for agent, one pushable box and a fixed goal. Predicted
/// frames may be physically invalid; evaluation does not repair those outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxesObservation {
    pub agent: usize,
    pub box_cell: usize,
    pub goal: usize,
}

impl BoxesObservation {
    fn fields(self) -> [usize; FIELDS] {
        [self.agent, self.box_cell, self.goal]
    }

    fn validate(self, side: usize) -> Result<(), String> {
        if self.fields().iter().any(|&p| p >= side * side) || self.agent == self.box_cell {
            Err("observation needs in-bounds cells and distinct agent/box positions".into())
        } else {
            Ok(())
        }
    }

    fn features(self, cells: usize) -> Vec<f32> {
        let mut features = vec![0.0; cells * FIELDS];
        for (field, position) in self.fields().into_iter().enumerate() {
            features[field * cells + position] = 1.0;
        }
        features
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxesTransition {
    pub action: BoxAction,
    pub observation: BoxesObservation,
    /// One on a newly completed push, zero otherwise.
    pub reward: f32,
    /// False for task completion, not for an external horizon truncation.
    pub continued: bool,
}

/// Deterministic single-box pushing, with impassable outer walls. Terminal
/// states are absorbing and cannot yield a second reward.
#[derive(Clone, Debug)]
pub struct BoxesWorld {
    side: usize,
    observation: BoxesObservation,
}

impl BoxesWorld {
    pub fn new(side: usize, seed: u64) -> Result<Self, String> {
        if !(3..=6).contains(&side) {
            return Err("boxes side must be in 3..=6".into());
        }
        let mut rng = SimpleRng::new(seed);
        let box_cell = (1 + rng.next_u32() as usize % (side - 2)) * side
            + 1
            + rng.next_u32() as usize % (side - 2);
        let mut agent = rng.next_u32() as usize % (side * side);
        while agent == box_cell {
            agent = rng.next_u32() as usize % (side * side);
        }
        let mut goal = rng.next_u32() as usize % (side * side);
        while goal == box_cell {
            goal = rng.next_u32() as usize % (side * side);
        }
        Self::from_observation(
            side,
            BoxesObservation {
                agent,
                box_cell,
                goal,
            },
        )
    }

    pub fn from_observation(side: usize, observation: BoxesObservation) -> Result<Self, String> {
        if !(3..=6).contains(&side) {
            return Err("boxes side must be in 3..=6".into());
        }
        observation.validate(side)?;
        Ok(Self { side, observation })
    }

    pub fn observation(&self) -> BoxesObservation {
        self.observation
    }

    fn neighbor(&self, cell: usize, action: BoxAction) -> Option<usize> {
        let row = cell / self.side;
        let col = cell % self.side;
        match action {
            BoxAction::Up if row > 0 => Some(cell - self.side),
            BoxAction::Down if row + 1 < self.side => Some(cell + self.side),
            BoxAction::Left if col > 0 => Some(cell - 1),
            BoxAction::Right if col + 1 < self.side => Some(cell + 1),
            BoxAction::Wait => Some(cell),
            _ => None,
        }
    }

    pub fn step(&mut self, action: BoxAction) -> BoxesTransition {
        let was_done = self.observation.box_cell == self.observation.goal;
        if !was_done {
            if let Some(next) = self.neighbor(self.observation.agent, action) {
                if next != self.observation.box_cell {
                    self.observation.agent = next;
                } else if let Some(pushed) = self.neighbor(next, action) {
                    self.observation.agent = next;
                    self.observation.box_cell = pushed;
                }
            }
        }
        let done = self.observation.box_cell == self.observation.goal;
        BoxesTransition {
            action,
            observation: self.observation,
            reward: if done && !was_done { 1.0 } else { 0.0 },
            continued: !done,
        }
    }

    // A bounded shortest-path behavior policy supplies useful pushes; half the
    // actions remain random. This is dataset collection, NOT a learned actor.
    // Both models see exactly the same policy, trajectories and held-out seeds.
    fn guided_action(&self) -> Option<BoxAction> {
        let cells = self.side * self.side;
        let mut seen = vec![false; cells * cells];
        let mut queue = VecDeque::new();
        queue.push_back((self.observation, None));
        seen[self.observation.agent * cells + self.observation.box_cell] = true;
        while let Some((observation, first)) = queue.pop_front() {
            if observation.box_cell == observation.goal {
                return first;
            }
            for action in BoxAction::ALL[..4].iter().copied() {
                let mut world = Self {
                    side: self.side,
                    observation,
                };
                let next = world.step(action).observation;
                let index = next.agent * cells + next.box_cell;
                if !seen[index] {
                    seen[index] = true;
                    queue.push_back((next, first.or(Some(action))));
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoxesEpisode {
    pub seed: u64,
    pub initial: BoxesObservation,
    pub transitions: Vec<BoxesTransition>,
}

pub fn boxes_episode(side: usize, horizon: usize, seed: u64) -> Result<BoxesEpisode, String> {
    if !(1..=MAX_HORIZON).contains(&horizon) {
        return Err(format!("episode horizon must be in 1..={MAX_HORIZON}"));
    }
    let mut world = BoxesWorld::new(side, seed)?;
    let initial = world.observation();
    let mut rng = SimpleRng::new(seed ^ 0xa517_9e31_28bd_460f);
    let mut transitions = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let random = BoxAction::ALL[rng.next_u32() as usize % ACTIONS];
        let action = if rng.next_u32() & 1 == 0 {
            world.guided_action().unwrap_or(random)
        } else {
            random
        };
        let transition = world.step(action);
        let continued = transition.continued;
        transitions.push(transition);
        if !continued {
            break;
        }
    }
    Ok(BoxesEpisode {
        seed,
        initial,
        transitions,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldPrediction {
    pub observation: BoxesObservation,
    /// Three normalized marginal distributions, in agent/box/goal order.
    pub observation_probabilities: Vec<Vec<f32>>,
    pub reward_probability: f32,
    pub continue_probability: f32,
    /// Sample used in stochastic recurrent feedback, not the observation mode.
    pub latent_category: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct WorldModelMetrics {
    pub transitions: usize,
    /// Prior-only teacher-conditioned next-frame mean NLL per field.
    pub one_step_nll: f64,
    pub one_step_field_accuracy: f64,
    pub one_step_exact_accuracy: f64,
    pub reward_brier: f64,
    pub continue_brier: f64,
    pub rollout_transitions: usize,
    /// Open-loop NLL per field, with only the initial frame observed.
    pub rollout_nll: f64,
    pub rollout_field_accuracy: f64,
    pub rollout_exact_accuracy: f64,
    /// The field accuracy at each available open-loop distance (index 0 = 1).
    pub rollout_accuracy_by_step: Vec<f64>,
    pub rollout_counts_by_step: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ModelComparisonResult {
    pub parameters: usize,
    pub optimizer_updates: usize,
    pub training_transitions: usize,
    pub first_epoch_objective: f64,
    pub last_epoch_objective: f64,
    /// All continuous-block parameter changes, including learned SSM weights.
    pub backbone_update_l2: f64,
    pub training_ms: f64,
    pub before: WorldModelMetrics,
    pub heldout: WorldModelMetrics,
}

#[derive(Clone, Debug)]
pub struct WorldModelResult {
    pub config: WorldModelConfig,
    pub train_seeds: Vec<u64>,
    pub heldout_seeds: Vec<u64>,
    pub stochastic: ModelComparisonResult,
    pub autoregressive: ModelComparisonResult,
}

impl WorldModelResult {
    /// CLI-friendly report without claiming that the stochastic model wins.
    pub fn report(&self) -> String {
        let mut report = format!(
            "boxes-world: RSSM-style categorical prototype, CPU, no actor/critic or diffusion\nseed={} side={} train_episodes={} heldout_episodes={} epochs={} horizon={} rollout_horizon={}\ncomparison: same episodes/updates/hidden widths; NOT parameter- or time-matched\nobjectives differ: stochastic adds KL and initial-frame reconstruction; heldout metrics use identical next frames\n",
            self.config.seed,
            self.config.side,
            self.config.train_episodes,
            self.config.heldout_episodes,
            self.config.epochs,
            self.config.horizon,
            self.config.rollout_horizon,
        );
        for (name, result) in [
            ("stochastic-pssa", &self.stochastic),
            ("autoregressive-pssa", &self.autoregressive),
        ] {
            report.push_str(&format!(
                "{name}: params={} updates={} transitions={} train_ms={:.1} objective={:.4}->{:.4} backbone_delta_l2={:.6}\n  heldout prior/next nll={:.4}->{:.4} field_acc={:.3} exact_acc={:.3} reward_brier={:.4} continue_brier={:.4}\n  open_loop nll={:.4}->{:.4} field_acc={:.3} exact_acc={:.3} steps={}\n",
                result.parameters, result.optimizer_updates, result.training_transitions,
                result.training_ms, result.first_epoch_objective, result.last_epoch_objective,
                result.backbone_update_l2, result.before.one_step_nll, result.heldout.one_step_nll,
                result.heldout.one_step_field_accuracy, result.heldout.one_step_exact_accuracy,
                result.heldout.reward_brier, result.heldout.continue_brier,
                result.before.rollout_nll, result.heldout.rollout_nll,
                result.heldout.rollout_field_accuracy, result.heldout.rollout_exact_accuracy,
                result.heldout.rollout_transitions,
            ));
        }
        report.push_str("scope: full-observation deterministic toy task; one categorical variable; biased straight-through estimator; quadratic episode BPTT; no learned policy, planning benchmark or generalization claim\n");
        report
    }
}

struct Linear {
    weight: ParamMatrix,
    bias: ParamVector,
}

impl Linear {
    fn new(output: usize, input: usize, rng: &mut SimpleRng) -> Self {
        let mut weight = ParamMatrix::random_xavier(output, input, rng);
        for value in &mut weight.data {
            *value *= 0.25;
        }
        Self {
            weight,
            bias: ParamVector::new(output, 0.0),
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = self.bias.data.clone();
        let mut projected = vec![0.0; out.len()];
        self.weight.matvec(input, &mut projected);
        for (out, projected) in out.iter_mut().zip(projected) {
            *out += projected;
        }
        out
    }

    fn backward(&mut self, input: &[f32], grad: &[f32]) -> Vec<f32> {
        let mut input_grad = vec![0.0; input.len()];
        for (row, &g) in grad.iter().enumerate() {
            self.bias.grad[row] += g;
            for (col, &x) in input.iter().enumerate() {
                let index = row * input.len() + col;
                self.weight.grad[index] += g * x;
                input_grad[col] += g * self.weight.data[index];
            }
        }
        input_grad
    }

    fn zero(&mut self) {
        self.weight.zero_grad();
        self.bias.zero_grad();
    }

    fn step(&mut self, lr: f32, step: usize) {
        self.weight.step_adamw(lr, 0.9, 0.999, 0.0, 1e-8, step);
        self.bias.step_adamw(lr, 0.9, 0.999, 0.0, 1e-8, step);
    }

    fn parameters(&self) -> usize {
        self.weight.data.len() + self.bias.data.len()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let total: f32 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= total;
    }
    probabilities
}

fn log_probability(value: f32) -> f32 {
    value.max(1e-30).ln()
}

fn softmax_vjp(probabilities: &[f32], grad: &[f32]) -> Vec<f32> {
    let dot: f32 = probabilities.iter().zip(grad).map(|(&p, &g)| p * g).sum();
    probabilities
        .iter()
        .zip(grad)
        .map(|(&p, &g)| p * (g - dot))
        .collect()
}

fn sample(probabilities: &[f32], rng: &mut SimpleRng) -> (usize, Vec<f32>) {
    let value = rng.gen_range_f32(0.0, 1.0);
    let mut cumulative = 0.0;
    let mut category = probabilities.len() - 1;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if value < cumulative {
            category = index;
            break;
        }
    }
    let mut one_hot = vec![0.0; probabilities.len()];
    one_hot[category] = 1.0;
    (category, one_hot)
}

fn add_into(target: &mut [f32], values: &[f32]) {
    for (target, &value) in target.iter_mut().zip(values) {
        *target += value;
    }
}

fn backbone(cfg: &WorldModelConfig, seed: u64) -> PSSAContinuousBlockV2 {
    let mut rng = SimpleRng::new(seed);
    PSSAContinuousBlockV2::new_with_rng(
        PSSAContinuousConfigV2 {
            d_latent: cfg.hidden,
            d_state: cfg.state,
            d_mem_key: 2,
            mem_capacity: 1,
            chunk_len: cfg.horizon + 1,
            tau_mem: 1.0,
            ema_alpha: 0.0,
        },
        &mut rng,
        0,
    )
}

// Visit the complete learned continuous block; empty memory has no trainable
// storage and consolidated adapter weights remain zero throughout this module.
fn block_matrices(block: &PSSAContinuousBlockV2) -> [&ParamMatrix; 12] {
    [
        &block.a_mat,
        &block.w_delta,
        &block.w_b,
        &block.w_c,
        &block.w_qx,
        &block.w_qh,
        &block.w_gate,
        &block.w_proj,
        &block.mlp_w1,
        &block.mlp_w2,
        &block.adapters[0].down_proj,
        &block.adapters[0].up_proj,
    ]
}

fn block_matrices_mut(block: &mut PSSAContinuousBlockV2) -> [&mut ParamMatrix; 12] {
    let adapter = &mut block.adapters[0];
    [
        &mut block.a_mat,
        &mut block.w_delta,
        &mut block.w_b,
        &mut block.w_c,
        &mut block.w_qx,
        &mut block.w_qh,
        &mut block.w_gate,
        &mut block.w_proj,
        &mut block.mlp_w1,
        &mut block.mlp_w2,
        &mut adapter.down_proj,
        &mut adapter.up_proj,
    ]
}

fn block_snapshot(block: &PSSAContinuousBlockV2) -> Vec<f32> {
    block_matrices(block)
        .into_iter()
        .flat_map(|p| p.data.iter().copied())
        .chain(block.norm_gamma.data.iter().copied())
        .chain(block.norm_beta.data.iter().copied())
        .collect()
}

fn clip_gradients(
    block: &mut PSSAContinuousBlockV2,
    heads: &mut [&mut Linear],
    limit: f32,
) -> Result<(), String> {
    let mut squared = 0.0f64;
    let mut visit = |grad: &[f32]| {
        for &g in grad {
            squared += (g as f64).powi(2);
        }
    };
    for matrix in block_matrices(block) {
        visit(&matrix.grad);
    }
    visit(&block.norm_gamma.grad);
    visit(&block.norm_beta.grad);
    for head in heads.iter() {
        visit(&head.weight.grad);
        visit(&head.bias.grad);
    }
    if !squared.is_finite() {
        return Err("non-finite world-model gradient; no optimizer update applied".into());
    }
    let scale = (limit as f64 / squared.sqrt().max(1e-30)).min(1.0) as f32;
    for matrix in block_matrices_mut(block) {
        for g in &mut matrix.grad {
            *g *= scale;
        }
    }
    for g in block
        .norm_gamma
        .grad
        .iter_mut()
        .chain(&mut block.norm_beta.grad)
    {
        *g *= scale;
    }
    for head in heads.iter_mut() {
        for g in head.weight.grad.iter_mut().chain(&mut head.bias.grad) {
            *g *= scale;
        }
    }
    Ok(())
}

fn update_delta(before: &[f32], block: &PSSAContinuousBlockV2) -> f64 {
    before
        .iter()
        .zip(block_snapshot(block))
        .map(|(&a, b)| (b as f64 - a as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn hidden(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|&x| (0.5 * x).tanh()).collect()
}

fn hidden_vjp(features: &[f32], gradient: &[f32]) -> Vec<f32> {
    features
        .iter()
        .zip(gradient)
        .map(|(&h, &g)| 0.5 * (1.0 - h * h) * g)
        .collect()
}

fn encode(encoder: &Linear, features: &[f32]) -> Vec<f32> {
    encoder
        .forward(features)
        .into_iter()
        .map(f32::tanh)
        .collect()
}

fn decode(
    output: &Linear,
    features: &[f32],
    cells: usize,
    category: Option<usize>,
) -> WorldPrediction {
    let logits = output.forward(features);
    let probabilities: Vec<Vec<f32>> = logits[..cells * FIELDS]
        .chunks_exact(cells)
        .map(softmax)
        .collect();
    let positions: Vec<usize> = probabilities.iter().map(|p| mode(p)).collect();
    WorldPrediction {
        observation: BoxesObservation {
            agent: positions[0],
            box_cell: positions[1],
            goal: positions[2],
        },
        observation_probabilities: probabilities,
        reward_probability: sigmoid(logits[cells * FIELDS]),
        continue_probability: sigmoid(logits[cells * FIELDS + 1]),
        latent_category: category,
    }
}

fn mode(probabilities: &[f32]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

// Per-frame field-mean CE and auxiliary binary CE. Initial frames have no
// reward/continue target; a horizon cut is never labeled task termination.
fn reconstruction(
    output: &mut Linear,
    features: &[f32],
    observation: BoxesObservation,
    target: Option<&BoxesTransition>,
    cells: usize,
    scale: f32,
    auxiliary: f32,
) -> (f32, Vec<f32>) {
    let logits = output.forward(features);
    let mut gradient = vec![0.0; logits.len()];
    let mut loss = 0.0;
    for (field, position) in observation.fields().into_iter().enumerate() {
        let range = field * cells..(field + 1) * cells;
        let mut probability = softmax(&logits[range.clone()]);
        loss -= log_probability(probability[position]) / FIELDS as f32;
        probability[position] -= 1.0;
        for (g, p) in gradient[range].iter_mut().zip(probability) {
            *g = p * scale / FIELDS as f32;
        }
    }
    if let Some(target) = target {
        for (offset, value) in [
            (0, target.reward),
            (1, if target.continued { 1.0 } else { 0.0 }),
        ] {
            let index = FIELDS * cells + offset;
            let logit = logits[index];
            // Stable BCE from logits; no clamped-probability derivative.
            loss += auxiliary * (logit.max(0.0) - value * logit + (-logit.abs()).exp().ln_1p());
            gradient[index] = auxiliary * scale * (sigmoid(logit) - value);
        }
    }
    (loss, output.backward(features, &gradient))
}

struct StochasticFrame {
    input_features: Vec<f32>,
    input: Vec<f32>,
    hidden: Vec<f32>,
    observation_features: Vec<f32>,
    prior: Vec<f32>,
    posterior: Vec<f32>,
    latent: Vec<f32>,
}

/// Reusable learned stochastic model. `reset` observes an initial frame;
/// `predict(action)` advances with only a prior sample; optional `observe`
/// assimilates the resulting real frame without advancing dynamics again.
/// `imagine` observes only the initial frame and always returns the requested
/// bounded number of steps (continuation probabilities do not censor scoring).
pub struct WorldModel {
    config: WorldModelConfig,
    block: PSSAContinuousBlockV2,
    input: Linear,
    prior: Linear,
    posterior: Linear,
    output: Linear,
    rng: SimpleRng,
    current_hidden: Vec<f32>,
    current_latent: Vec<f32>,
    initialized: bool,
    steps: usize,
    updates: usize,
}

impl WorldModel {
    pub fn new(config: &WorldModelConfig, seed: u64) -> Result<Self, String> {
        config.validate()?;
        let block = backbone(config, seed);
        let mut rng = SimpleRng::new(seed ^ 0xda43_671b_9ef2_0851);
        let observations = FIELDS * config.cells();
        Ok(Self {
            config: config.clone(),
            block,
            input: Linear::new(
                config.hidden,
                config.categories + ACTIONS + observations,
                &mut rng,
            ),
            prior: Linear::new(config.categories, config.hidden, &mut rng),
            posterior: Linear::new(config.categories, config.hidden + observations, &mut rng),
            output: Linear::new(
                observations + 2,
                config.hidden + config.categories,
                &mut rng,
            ),
            rng: SimpleRng::new(seed ^ 0x13c6_fa8d_504b_972e),
            current_hidden: vec![0.0; config.hidden],
            current_latent: vec![0.0; config.categories],
            initialized: false,
            steps: 0,
            updates: 0,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.block.parameter_count()
            + self.input.parameters()
            + self.prior.parameters()
            + self.posterior.parameters()
            + self.output.parameters()
    }

    fn input_features(
        &self,
        initial: Option<BoxesObservation>,
        action: Option<BoxAction>,
    ) -> Vec<f32> {
        let mut features = vec![0.0; self.input.weight.cols];
        if let Some(observation) = initial {
            features[self.config.categories + ACTIONS..]
                .copy_from_slice(&observation.features(self.config.cells()));
        } else {
            features[..self.config.categories].copy_from_slice(&self.current_latent);
            if let Some(action) = action {
                features[self.config.categories + action.index()] = 1.0;
            }
        }
        features
    }

    fn advance(&mut self, features: &[f32]) -> Vec<f32> {
        let input = encode(&self.input, features);
        let mut raw = vec![0.0; self.config.hidden];
        self.block.forward_continuous_inference(&input, &mut raw);
        self.current_hidden = hidden(&raw);
        input
    }

    pub fn reset(&mut self, observation: BoxesObservation) -> Result<(), String> {
        observation.validate(self.config.side)?;
        self.block.reset_recurrent_state();
        self.current_latent.fill(0.0);
        let features = self.input_features(Some(observation), None);
        self.advance(&features);
        self.initialized = true;
        self.steps = 0;
        self.observe(observation)
    }

    pub fn observe(&mut self, observation: BoxesObservation) -> Result<(), String> {
        if !self.initialized {
            return Err("reset the world model before observe".into());
        }
        observation.validate(self.config.side)?;
        let features: Vec<f32> = self
            .current_hidden
            .iter()
            .copied()
            .chain(observation.features(self.config.cells()))
            .collect();
        let posterior = softmax(&self.posterior.forward(&features));
        self.current_latent = sample(&posterior, &mut self.rng).1;
        Ok(())
    }

    pub fn predict(&mut self, action: BoxAction) -> Result<WorldPrediction, String> {
        if !self.initialized {
            return Err("reset the world model before predict".into());
        }
        if self.steps >= MAX_HORIZON {
            return Err("world-model prediction horizon exceeded; reset first".into());
        }
        let features = self.input_features(None, Some(action));
        self.advance(&features);
        let probabilities = softmax(&self.prior.forward(&self.current_hidden));
        let (category, latent) = sample(&probabilities, &mut self.rng);
        self.current_latent = latent;
        self.steps += 1;
        // Exact marginal over the one categorical latent, instead of crediting
        // an observation-conditioned posterior or one lucky decoder sample.
        let cells = self.config.cells();
        let mut prediction = WorldPrediction {
            observation: BoxesObservation {
                agent: 0,
                box_cell: 0,
                goal: 0,
            },
            observation_probabilities: vec![vec![0.0; cells]; FIELDS],
            reward_probability: 0.0,
            continue_probability: 0.0,
            latent_category: Some(category),
        };
        for (category, &probability) in probabilities.iter().enumerate() {
            let mut features = self.current_hidden.clone();
            features.resize(self.config.hidden + self.config.categories, 0.0);
            features[self.config.hidden + category] = 1.0;
            let decoded = decode(&self.output, &features, cells, Some(category));
            for (result, decoded) in prediction
                .observation_probabilities
                .iter_mut()
                .zip(decoded.observation_probabilities)
            {
                for (value, p) in result.iter_mut().zip(decoded) {
                    *value += probability * p;
                }
            }
            prediction.reward_probability += probability * decoded.reward_probability;
            prediction.continue_probability += probability * decoded.continue_probability;
        }
        prediction.observation = BoxesObservation {
            agent: mode(&prediction.observation_probabilities[0]),
            box_cell: mode(&prediction.observation_probabilities[1]),
            goal: mode(&prediction.observation_probabilities[2]),
        };
        ensure_prediction(&prediction)?;
        Ok(prediction)
    }

    pub fn imagine(
        &mut self,
        initial: BoxesObservation,
        actions: &[BoxAction],
        seed: u64,
    ) -> Result<Vec<WorldPrediction>, String> {
        if actions.len() > MAX_HORIZON {
            return Err("imagined rollout exceeds bounded horizon".into());
        }
        self.rng = SimpleRng::new(seed);
        self.reset(initial)?;
        actions.iter().map(|&action| self.predict(action)).collect()
    }

    /// One full-episode Adam update, bounded by this configuration's total
    /// update budget. Supplied transitions must match the deterministic world.
    pub fn train_episode(&mut self, episode: &BoxesEpisode) -> Result<f32, String> {
        validate_episode(episode, &self.config)?;
        if self.updates >= self.config.train_episodes * self.config.epochs {
            return Err("world-model optimizer update budget exhausted".into());
        }
        let loss = self.episode_gradients(episode)?;
        clip_gradients(
            &mut self.block,
            &mut [
                &mut self.input,
                &mut self.prior,
                &mut self.posterior,
                &mut self.output,
            ],
            self.config.gradient_clip,
        )?;
        self.updates += 1;
        self.block.apply_adamw(
            self.config.learning_rate,
            0.9,
            0.999,
            0.0,
            1e-8,
            self.updates,
        );
        self.input.step(self.config.learning_rate, self.updates);
        self.prior.step(self.config.learning_rate, self.updates);
        self.posterior.step(self.config.learning_rate, self.updates);
        self.output.step(self.config.learning_rate, self.updates);
        self.initialized = false;
        ensure_weights(
            &self.block,
            &[&self.input, &self.prior, &self.posterior, &self.output],
        )?;
        Ok(loss)
    }

    fn episode_gradients(&mut self, episode: &BoxesEpisode) -> Result<f32, String> {
        let count = episode.transitions.len() + 1;
        let width = self.config.hidden;
        let categories = self.config.categories;
        let cells = self.config.cells();
        self.block.reset_recurrent_state();
        self.current_latent.fill(0.0);
        let mut frames = Vec::with_capacity(count);
        for t in 0..count {
            let observation = if t == 0 {
                episode.initial
            } else {
                episode.transitions[t - 1].observation
            };
            let features = self.input_features(
                (t == 0).then_some(episode.initial),
                if t == 0 {
                    None
                } else {
                    Some(episode.transitions[t - 1].action)
                },
            );
            let input = self.advance(&features);
            let observation_features = observation.features(cells);
            let posterior_features: Vec<f32> = self
                .current_hidden
                .iter()
                .copied()
                .chain(observation_features.iter().copied())
                .collect();
            let prior = softmax(&self.prior.forward(&self.current_hidden));
            let posterior = softmax(&self.posterior.forward(&posterior_features));
            self.current_latent = sample(&posterior, &mut self.rng).1;
            frames.push(StochasticFrame {
                input_features: features,
                input,
                hidden: self.current_hidden.clone(),
                observation_features,
                prior,
                posterior,
                latent: self.current_latent.clone(),
            });
        }
        // Replay the already sampled causal input sequence into ONE full tape.
        // This exposes SSM carry VJPs across all episode transitions.
        let inputs: Vec<f32> = frames
            .iter()
            .flat_map(|f| f.input.iter().copied())
            .collect();
        self.block.reset_recurrent_state();
        self.block.forward_train_chunk(&inputs, count);
        self.block.zero_gradients();
        self.input.zero();
        self.prior.zero();
        self.posterior.zero();
        self.output.zero();
        let scale = 1.0 / count as f32;
        let mut h_grad = vec![vec![0.0; width]; count];
        let mut z_grad = vec![vec![0.0; categories]; count];
        let mut q_kl_grad = vec![vec![0.0; categories]; count];
        let mut loss = 0.0;
        for t in 0..count {
            let frame = &frames[t];
            let target = if t == 0 {
                None
            } else {
                Some(&episode.transitions[t - 1])
            };
            let observation = target.map_or(episode.initial, |t| t.observation);
            let features: Vec<f32> = frame
                .hidden
                .iter()
                .copied()
                .chain(frame.latent.iter().copied())
                .collect();
            let (recon, gradient) = reconstruction(
                &mut self.output,
                &features,
                observation,
                target,
                cells,
                scale,
                self.config.auxiliary_weight,
            );
            loss += recon;
            add_into(&mut h_grad[t], &gradient[..width]);
            add_into(&mut z_grad[t], &gradient[width..]);
            let log_ratio: Vec<f32> = frame
                .posterior
                .iter()
                .zip(&frame.prior)
                .map(|(&q, &p)| log_probability(q) - log_probability(p))
                .collect();
            let kl: f32 = frame
                .posterior
                .iter()
                .zip(&log_ratio)
                .map(|(&q, &r)| q * r)
                .sum();
            loss += self.config.kl_weight * kl;
            let factor = self.config.kl_weight * scale;
            let prior_gradient: Vec<f32> = frame
                .prior
                .iter()
                .zip(&frame.posterior)
                .map(|(&p, &q)| factor * (p - q))
                .collect();
            add_into(
                &mut h_grad[t],
                &self.prior.backward(&frame.hidden, &prior_gradient),
            );
            for k in 0..categories {
                q_kl_grad[t][k] = factor * frame.posterior[k] * (log_ratio[k] - kl);
            }
        }
        // A prefix VJP at t propagates its output loss through every earlier SSM
        // input. Each such input also depends on z_(s-1); schedule those latent
        // adjoints before processing its posterior. Prefixes add parameter VJPs
        // exactly once per output loss, not repeatedly for accumulated losses.
        for t in (0..count).rev() {
            let frame = &frames[t];
            let mut q_gradient = softmax_vjp(&frame.posterior, &z_grad[t]);
            add_into(&mut q_gradient, &q_kl_grad[t]);
            let posterior_features: Vec<f32> = frame
                .hidden
                .iter()
                .copied()
                .chain(frame.observation_features.iter().copied())
                .collect();
            let gradient = self.posterior.backward(&posterior_features, &q_gradient);
            add_into(&mut h_grad[t], &gradient[..width]);
            let mut output_adjoints = vec![0.0; (t + 1) * width];
            output_adjoints[t * width..].copy_from_slice(&hidden_vjp(&frame.hidden, &h_grad[t]));
            let mut input_adjoints = vec![0.0; output_adjoints.len()];
            self.block
                .backward_chunk(&output_adjoints, t + 1, &mut input_adjoints);
            for s in 0..=t {
                let frame = &frames[s];
                let gradient: Vec<f32> = frame
                    .input
                    .iter()
                    .zip(&input_adjoints[s * width..(s + 1) * width])
                    .map(|(&x, &g)| (1.0 - x * x) * g)
                    .collect();
                let input_gradient = self.input.backward(&frame.input_features, &gradient);
                if s > 0 {
                    add_into(&mut z_grad[s - 1], &input_gradient[..categories]);
                }
            }
        }
        if !loss.is_finite() {
            return Err("non-finite stochastic world-model objective".into());
        }
        Ok(loss * scale)
    }
}

struct AutoregressiveModel {
    config: WorldModelConfig,
    block: PSSAContinuousBlockV2,
    input: Linear,
    output: Linear,
    observation: BoxesObservation,
    initialized: bool,
    steps: usize,
    updates: usize,
}

impl AutoregressiveModel {
    fn new(config: &WorldModelConfig, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed ^ 0xda43_671b_9ef2_0851);
        Self {
            config: config.clone(),
            block: backbone(config, seed),
            input: Linear::new(config.hidden, FIELDS * config.cells() + ACTIONS, &mut rng),
            output: Linear::new(FIELDS * config.cells() + 2, config.hidden, &mut rng),
            observation: BoxesObservation {
                agent: 0,
                box_cell: 1,
                goal: 2,
            },
            initialized: false,
            steps: 0,
            updates: 0,
        }
    }

    fn features(&self, observation: BoxesObservation, action: BoxAction) -> Vec<f32> {
        let mut features = observation.features(self.config.cells());
        features.resize(features.len() + ACTIONS, 0.0);
        features[FIELDS * self.config.cells() + action.index()] = 1.0;
        features
    }

    fn reset(&mut self, observation: BoxesObservation) {
        self.block.reset_recurrent_state();
        self.observation = observation;
        self.initialized = true;
        self.steps = 0;
    }

    fn predict(&mut self, action: BoxAction) -> Result<WorldPrediction, String> {
        if !self.initialized || self.steps >= MAX_HORIZON {
            return Err("autoregressive model needs reset or bounded horizon".into());
        }
        let input = encode(&self.input, &self.features(self.observation, action));
        let mut raw = vec![0.0; self.config.hidden];
        self.block.forward_continuous_inference(&input, &mut raw);
        let prediction = decode(&self.output, &hidden(&raw), self.config.cells(), None);
        self.observation = prediction.observation;
        self.steps += 1;
        ensure_prediction(&prediction)?;
        Ok(prediction)
    }

    fn train_episode(&mut self, episode: &BoxesEpisode) -> Result<f32, String> {
        let count = episode.transitions.len();
        let width = self.config.hidden;
        let mut observation = episode.initial;
        let mut features = Vec::with_capacity(count);
        let mut inputs = Vec::with_capacity(count * width);
        for transition in &episode.transitions {
            let row = self.features(observation, transition.action);
            inputs.extend(encode(&self.input, &row));
            features.push(row);
            observation = transition.observation;
        }
        self.block.reset_recurrent_state();
        self.block.forward_train_chunk(&inputs, count);
        self.block.zero_gradients();
        self.input.zero();
        self.output.zero();
        let mut loss = 0.0;
        let mut adjoints = vec![0.0; inputs.len()];
        for (t, target) in episode.transitions.iter().enumerate() {
            let h = hidden(&self.block.tape.z_final[t * width..(t + 1) * width]);
            let (recon, gradient) = reconstruction(
                &mut self.output,
                &h,
                target.observation,
                Some(target),
                self.config.cells(),
                1.0 / count as f32,
                self.config.auxiliary_weight,
            );
            loss += recon;
            adjoints[t * width..(t + 1) * width].copy_from_slice(&hidden_vjp(&h, &gradient));
        }
        let mut input_adjoints = vec![0.0; adjoints.len()];
        self.block
            .backward_chunk(&adjoints, count, &mut input_adjoints);
        for t in 0..count {
            let gradient: Vec<f32> = inputs[t * width..(t + 1) * width]
                .iter()
                .zip(&input_adjoints[t * width..(t + 1) * width])
                .map(|(&x, &g)| (1.0 - x * x) * g)
                .collect();
            self.input.backward(&features[t], &gradient);
        }
        if !loss.is_finite() {
            return Err("non-finite autoregressive objective".into());
        }
        clip_gradients(
            &mut self.block,
            &mut [&mut self.input, &mut self.output],
            self.config.gradient_clip,
        )?;
        self.updates += 1;
        self.block.apply_adamw(
            self.config.learning_rate,
            0.9,
            0.999,
            0.0,
            1e-8,
            self.updates,
        );
        self.input.step(self.config.learning_rate, self.updates);
        self.output.step(self.config.learning_rate, self.updates);
        self.initialized = false;
        ensure_weights(&self.block, &[&self.input, &self.output])?;
        Ok(loss / count as f32)
    }

    fn parameter_count(&self) -> usize {
        self.block.parameter_count() + self.input.parameters() + self.output.parameters()
    }
}

fn validate_episode(episode: &BoxesEpisode, config: &WorldModelConfig) -> Result<(), String> {
    if episode.transitions.is_empty() || episode.transitions.len() > config.horizon {
        return Err("training episode must contain 1..=configured horizon transitions".into());
    }
    let mut world = BoxesWorld::from_observation(config.side, episode.initial)?;
    if !world.step(BoxAction::Wait).continued {
        return Err("training episode must start before task completion".into());
    }
    for (index, transition) in episode.transitions.iter().enumerate() {
        if world.step(transition.action) != *transition {
            return Err(
                "training transition does not match deterministic boxes-world dynamics".into(),
            );
        }
        if !transition.continued && index + 1 != episode.transitions.len() {
            return Err("training episode must end on task completion".into());
        }
    }
    Ok(())
}

fn ensure_weights(block: &PSSAContinuousBlockV2, heads: &[&Linear]) -> Result<(), String> {
    let finite = block_matrices(block)
        .iter()
        .all(|p| p.data.iter().all(|x| x.is_finite()))
        && block
            .norm_gamma
            .data
            .iter()
            .chain(&block.norm_beta.data)
            .all(|x| x.is_finite())
        && heads.iter().all(|h| {
            h.weight
                .data
                .iter()
                .chain(&h.bias.data)
                .all(|x| x.is_finite())
        });
    if finite {
        Ok(())
    } else {
        Err("non-finite world-model weights after update".into())
    }
}

fn ensure_prediction(prediction: &WorldPrediction) -> Result<(), String> {
    let normalized = prediction.observation_probabilities.iter().all(|p| {
        p.iter()
            .all(|x| x.is_finite() && (0.0..=1.00001).contains(x))
            && (p.iter().sum::<f32>() - 1.0).abs() < 0.0001
    });
    if normalized
        && prediction.reward_probability.is_finite()
        && prediction.continue_probability.is_finite()
        && (0.0..=1.00001).contains(&prediction.reward_probability)
        && (0.0..=1.00001).contains(&prediction.continue_probability)
    {
        Ok(())
    } else {
        Err("non-finite or unnormalized world-model prediction".into())
    }
}

#[derive(Default)]
struct Score {
    count: usize,
    nll: f64,
    correct: usize,
    exact: usize,
    reward: f64,
    continued: f64,
}

impl Score {
    fn add(&mut self, prediction: &WorldPrediction, target: &BoxesTransition) {
        self.count += 1;
        let mut correct = 0;
        for (field, position) in target.observation.fields().into_iter().enumerate() {
            self.nll -=
                log_probability(prediction.observation_probabilities[field][position]) as f64;
            correct += usize::from(prediction.observation.fields()[field] == position);
        }
        self.correct += correct;
        self.exact += usize::from(correct == FIELDS);
        self.reward += (prediction.reward_probability as f64 - target.reward as f64).powi(2);
        self.continued += (prediction.continue_probability as f64
            - if target.continued { 1.0 } else { 0.0 })
        .powi(2);
    }

    fn nll(&self) -> f64 {
        self.nll / (self.count.max(1) * FIELDS) as f64
    }
    fn accuracy(&self) -> f64 {
        self.correct as f64 / (self.count.max(1) * FIELDS) as f64
    }
    fn exact(&self) -> f64 {
        self.exact as f64 / self.count.max(1) as f64
    }
}

trait EvaluatedModel {
    fn start(&mut self, initial: BoxesObservation, seed: u64) -> Result<(), String>;
    fn next(&mut self, action: BoxAction) -> Result<WorldPrediction, String>;
    fn assimilate(&mut self, observation: BoxesObservation) -> Result<(), String>;
}

impl EvaluatedModel for WorldModel {
    fn start(&mut self, initial: BoxesObservation, seed: u64) -> Result<(), String> {
        self.rng = SimpleRng::new(seed);
        self.reset(initial)
    }
    fn next(&mut self, action: BoxAction) -> Result<WorldPrediction, String> {
        self.predict(action)
    }
    fn assimilate(&mut self, observation: BoxesObservation) -> Result<(), String> {
        self.observe(observation)
    }
}

impl EvaluatedModel for AutoregressiveModel {
    fn start(&mut self, initial: BoxesObservation, _seed: u64) -> Result<(), String> {
        self.reset(initial);
        Ok(())
    }
    fn next(&mut self, action: BoxAction) -> Result<WorldPrediction, String> {
        self.predict(action)
    }
    fn assimilate(&mut self, observation: BoxesObservation) -> Result<(), String> {
        self.observation = observation;
        Ok(())
    }
}

fn evaluate(
    model: &mut impl EvaluatedModel,
    episodes: &[BoxesEpisode],
    horizon: usize,
) -> Result<WorldModelMetrics, String> {
    let mut one = Score::default();
    let mut rollout = Score::default();
    let mut distances: Vec<Score> = (0..horizon).map(|_| Score::default()).collect();
    for episode in episodes {
        let seed = episode.seed ^ 0x9d60_eb83_741a_2cf5;
        model.start(episode.initial, seed)?;
        for target in &episode.transitions {
            // Score BEFORE assimilation: targets never enter the prior or the
            // autoregressive next-frame predictor for their own timestep.
            one.add(&model.next(target.action)?, target);
            model.assimilate(target.observation)?;
        }
        model.start(episode.initial, seed)?;
        for (t, target) in episode.transitions.iter().take(horizon).enumerate() {
            let prediction = model.next(target.action)?;
            rollout.add(&prediction, target);
            distances[t].add(&prediction, target);
            // Deliberately no assimilation, even if predictions are invalid.
        }
    }
    Ok(WorldModelMetrics {
        transitions: one.count,
        one_step_nll: one.nll(),
        one_step_field_accuracy: one.accuracy(),
        one_step_exact_accuracy: one.exact(),
        reward_brier: one.reward / one.count.max(1) as f64,
        continue_brier: one.continued / one.count.max(1) as f64,
        rollout_transitions: rollout.count,
        rollout_nll: rollout.nll(),
        rollout_field_accuracy: rollout.accuracy(),
        rollout_exact_accuracy: rollout.exact(),
        rollout_accuracy_by_step: distances.iter().map(Score::accuracy).collect(),
        rollout_counts_by_step: distances.iter().map(|s| s.count).collect(),
    })
}

/// Train fresh tiny models and compare prior/next-frame and open-loop metrics
/// before/after on disjoint held-out episode seeds. No external data or device
/// access. Resource validation occurs before model/dataset allocations.
pub fn run_world_model(config: &WorldModelConfig) -> Result<WorldModelResult, String> {
    run_world_model_with_progress(config, |_, _, _| {})
}

/// As `run_world_model`, with one bounded progress event per completed epoch.
/// Objectives are model-specific training losses, not held-out language CE.
pub fn run_world_model_with_progress(
    config: &WorldModelConfig,
    mut progress: impl FnMut(usize, f64, f64),
) -> Result<WorldModelResult, String> {
    config.validate()?;
    let train_seeds: Vec<u64> = (0..config.train_episodes)
        .map(|i| config.seed.wrapping_add(1 + i as u64))
        .collect();
    let heldout_seeds: Vec<u64> = (0..config.heldout_episodes)
        .map(|i| {
            config
                .seed
                .wrapping_add(1 + config.train_episodes as u64 + i as u64)
        })
        .collect();
    let train: Vec<BoxesEpisode> = train_seeds
        .iter()
        .map(|&seed| boxes_episode(config.side, config.horizon, seed))
        .collect::<Result<_, _>>()?;
    let heldout: Vec<BoxesEpisode> = heldout_seeds
        .iter()
        .map(|&seed| boxes_episode(config.side, config.horizon, seed))
        .collect::<Result<_, _>>()?;
    let training_transitions =
        train.iter().map(|e| e.transitions.len()).sum::<usize>() * config.epochs;
    let mut stochastic = WorldModel::new(config, config.seed)?;
    let mut autoregressive = AutoregressiveModel::new(config, config.seed);
    let stochastic_before = evaluate(&mut stochastic, &heldout, config.rollout_horizon)?;
    let autoregressive_before = evaluate(&mut autoregressive, &heldout, config.rollout_horizon)?;
    // Evaluation sampling must not influence training's random stream.
    stochastic.rng = SimpleRng::new(config.seed ^ 0x13c6_fa8d_504b_972e);
    let stochastic_snapshot = block_snapshot(&stochastic.block);
    let autoregressive_snapshot = block_snapshot(&autoregressive.block);
    let mut stochastic_objectives = Vec::with_capacity(config.epochs);
    let mut autoregressive_objectives = Vec::with_capacity(config.epochs);
    let mut stochastic_ms = 0.0;
    let mut autoregressive_ms = 0.0;
    for epoch in 0..config.epochs {
        let mut stochastic_loss = 0.0;
        let mut autoregressive_loss = 0.0;
        for index in 0..train.len() {
            // Same deterministic ordering, rotated each epoch, for both models.
            let episode = &train[(index + epoch) % train.len()];
            let started = Instant::now();
            stochastic_loss += stochastic.train_episode(episode)? as f64;
            stochastic_ms += started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            autoregressive_loss += autoregressive.train_episode(episode)? as f64;
            autoregressive_ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        stochastic_objectives.push(stochastic_loss / train.len() as f64);
        autoregressive_objectives.push(autoregressive_loss / train.len() as f64);
        progress(
            epoch + 1,
            stochastic_objectives[epoch],
            autoregressive_objectives[epoch],
        );
    }
    let stochastic_heldout = evaluate(&mut stochastic, &heldout, config.rollout_horizon)?;
    let autoregressive_heldout = evaluate(&mut autoregressive, &heldout, config.rollout_horizon)?;
    Ok(WorldModelResult {
        config: config.clone(),
        train_seeds,
        heldout_seeds,
        stochastic: ModelComparisonResult {
            parameters: stochastic.parameter_count(),
            optimizer_updates: stochastic.updates,
            training_transitions,
            first_epoch_objective: stochastic_objectives[0],
            last_epoch_objective: *stochastic_objectives.last().unwrap(),
            backbone_update_l2: update_delta(&stochastic_snapshot, &stochastic.block),
            training_ms: stochastic_ms,
            before: stochastic_before,
            heldout: stochastic_heldout,
        },
        autoregressive: ModelComparisonResult {
            parameters: autoregressive.parameter_count(),
            optimizer_updates: autoregressive.updates,
            training_transitions,
            first_epoch_objective: autoregressive_objectives[0],
            last_epoch_objective: *autoregressive_objectives.last().unwrap(),
            backbone_update_l2: update_delta(&autoregressive_snapshot, &autoregressive.block),
            training_ms: autoregressive_ms,
            before: autoregressive_before,
            heldout: autoregressive_heldout,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> WorldModelConfig {
        WorldModelConfig {
            side: 3,
            hidden: 6,
            state: 2,
            categories: 4,
            train_episodes: 4,
            heldout_episodes: 2,
            horizon: 4,
            rollout_horizon: 4,
            epochs: 2,
            ..WorldModelConfig::default()
        }
    }

    fn push_episode() -> BoxesEpisode {
        let initial = BoxesObservation {
            agent: 3,
            box_cell: 4,
            goal: 2,
        };
        let mut world = BoxesWorld::from_observation(3, initial).unwrap();
        let transitions = [
            BoxAction::Right,
            BoxAction::Down,
            BoxAction::Right,
            BoxAction::Up,
        ]
        .into_iter()
        .map(|action| world.step(action))
        .collect();
        BoxesEpisode {
            seed: 9,
            initial,
            transitions,
        }
    }

    #[test]
    fn movement_pushes_walls_and_absorbing_terminal() {
        let mut world = BoxesWorld::from_observation(
            3,
            BoxesObservation {
                agent: 1,
                box_cell: 2,
                goal: 8,
            },
        )
        .unwrap();
        let before = world.observation();
        assert_eq!(world.step(BoxAction::Right).observation, before); // Cannot push through wall.
        assert_eq!(world.step(BoxAction::Up).observation, before);
        assert_eq!(world.step(BoxAction::Wait).observation, before);
        assert_eq!(world.step(BoxAction::Down).observation.agent, 4);
        let episode = push_episode();
        assert_eq!(
            episode.transitions[0].observation,
            BoxesObservation {
                agent: 4,
                box_cell: 5,
                goal: 2
            }
        );
        assert_eq!(episode.transitions[3].reward, 1.0);
        assert!(!episode.transitions[3].continued);
        let mut solved =
            BoxesWorld::from_observation(3, episode.transitions[3].observation).unwrap();
        let next = solved.step(BoxAction::Left);
        assert_eq!(next.observation, episode.transitions[3].observation);
        assert_eq!(next.reward, 0.0);
        assert!(!next.continued);
        assert!(
            BoxesWorld::from_observation(
                3,
                BoxesObservation {
                    agent: 2,
                    box_cell: 2,
                    goal: 1
                }
            )
            .is_err()
        );
    }

    #[test]
    fn generation_is_reproducible_and_truncation_is_not_termination() {
        assert_eq!(
            boxes_episode(4, 12, 91).unwrap(),
            boxes_episode(4, 12, 91).unwrap()
        );
        let episode = boxes_episode(4, 1, 91).unwrap();
        assert_eq!(episode.transitions.len(), 1);
        assert!(episode.transitions[0].continued);
        for seed in 0..12 {
            let episode = boxes_episode(3, 4, seed).unwrap();
            validate_episode(&episode, &tiny_config()).unwrap();
        }
    }

    #[test]
    fn reject_resource_and_transition_errors_before_updates() {
        let mut config = tiny_config();
        config.learning_rate = f32::NAN;
        assert!(
            run_world_model_with_progress(&config, |_, _, _| panic!(
                "invalid config must not start"
            ))
            .is_err()
        );
        config = tiny_config();
        config.hidden = usize::MAX;
        assert!(config.validate().is_err());
        config = tiny_config();
        config.train_episodes = 128;
        config.epochs = 12;
        config.horizon = 24;
        config.hidden = 48;
        config.state = 8;
        config.categories = 32;
        assert!(config.validate().is_err());
        config = tiny_config();
        config.train_episodes = 1;
        config.epochs = 1;
        let mut model = WorldModel::new(&config, 10).unwrap();
        assert!(model.predict(BoxAction::Up).is_err());
        let mut invalid = push_episode();
        invalid.transitions[0].reward = 1.0;
        assert!(model.train_episode(&invalid).is_err());
        assert_eq!(model.updates, 0);
        assert!(model.train_episode(&push_episode()).unwrap().is_finite());
        assert!(model.train_episode(&push_episode()).is_err());
        assert_eq!(model.updates, 1);
    }

    #[test]
    fn prior_prediction_cannot_read_the_frame_it_is_scored_against() {
        let config = tiny_config();
        let episode = push_episode();
        let mut a = WorldModel::new(&config, 101).unwrap();
        let mut b = WorldModel::new(&config, 101).unwrap();
        a.reset(episode.initial).unwrap();
        b.reset(episode.initial).unwrap();
        // Prediction takes ONLY an action. Changing the yet-unseen target has
        // no route into it; posterior conditioning is a separate operation.
        let original = episode.transitions[0].observation;
        let changed = BoxesObservation {
            agent: 0,
            box_cell: 7,
            goal: 1,
        };
        assert_eq!(
            a.predict(BoxAction::Right).unwrap(),
            b.predict(BoxAction::Right).unwrap()
        );
        let q_a: Vec<f32> = a
            .current_hidden
            .iter()
            .copied()
            .chain(original.features(config.cells()))
            .collect();
        let q_b: Vec<f32> = b
            .current_hidden
            .iter()
            .copied()
            .chain(changed.features(config.cells()))
            .collect();
        assert_ne!(a.posterior.forward(&q_a), b.posterior.forward(&q_b));
        a.observe(original).unwrap();
        b.observe(changed).unwrap();
    }

    #[test]
    fn imagined_rollouts_are_seeded_prior_only_and_fixed_length() {
        let config = tiny_config();
        let mut model = WorldModel::new(&config, 12).unwrap();
        let episode = push_episode();
        let actions: Vec<_> = episode.transitions.iter().map(|t| t.action).collect();
        let predictions = model.imagine(episode.initial, &actions, 3).unwrap();
        assert_eq!(predictions.len(), actions.len());
        assert_eq!(
            predictions,
            model.imagine(episode.initial, &actions, 3).unwrap()
        );
        for prediction in predictions {
            ensure_prediction(&prediction).unwrap();
            assert!(prediction.latent_category.unwrap() < config.categories);
        }
        assert!(
            model
                .imagine(episode.initial, &vec![BoxAction::Wait; MAX_HORIZON + 1], 3)
                .is_err()
        );
    }

    #[test]
    fn stochastic_and_autoregressive_train_recurrent_weights_not_just_heads() {
        let config = tiny_config();
        let episode = push_episode();
        let mut stochastic = WorldModel::new(&config, 30).unwrap();
        let mut autoregressive = AutoregressiveModel::new(&config, 30);
        // Same deterministic recurrent backbone initialization for the control.
        assert_eq!(
            block_snapshot(&stochastic.block),
            block_snapshot(&autoregressive.block)
        );
        let before = block_snapshot(&stochastic.block);
        let before_a = stochastic.block.a_mat.data.clone();
        let before_prior = stochastic.prior.weight.data.clone();
        let before_posterior = stochastic.posterior.weight.data.clone();
        let before_input = stochastic.input.weight.data.clone();
        assert!(stochastic.train_episode(&episode).unwrap().is_finite());
        assert!(update_delta(&before, &stochastic.block) > 0.0);
        assert_ne!(before_a, stochastic.block.a_mat.data);
        assert_ne!(before_prior, stochastic.prior.weight.data);
        assert_ne!(before_posterior, stochastic.posterior.weight.data);
        assert_ne!(before_input, stochastic.input.weight.data);
        let before = block_snapshot(&autoregressive.block);
        assert!(autoregressive.train_episode(&episode).unwrap().is_finite());
        assert!(update_delta(&before, &autoregressive.block) > 0.0);
        assert_eq!(stochastic.block.memory.count, 0);
        assert!(
            stochastic.block.adapters[0]
                .consolidated_up
                .iter()
                .all(|&x| x == 0.0)
        );
    }

    // Finite differences of a local straight-through surrogate, NOT of the
    // discontinuous categorical sample. Hold sampled one-hots fixed at the
    // reference trajectory and add q(theta)-q(theta_reference) at each frame.
    // This checks SSM temporal carry plus posterior-to-next-input feedback.
    fn reference_latents(
        model: &mut WorldModel,
        episode: &BoxesEpisode,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        model.block.reset_recurrent_state();
        model.current_latent.fill(0.0);
        let mut result = Vec::new();
        for t in 0..=episode.transitions.len() {
            let observation = if t == 0 {
                episode.initial
            } else {
                episode.transitions[t - 1].observation
            };
            let features = model.input_features(
                (t == 0).then_some(episode.initial),
                if t == 0 {
                    None
                } else {
                    Some(episode.transitions[t - 1].action)
                },
            );
            model.advance(&features);
            let features: Vec<f32> = model
                .current_hidden
                .iter()
                .copied()
                .chain(observation.features(model.config.cells()))
                .collect();
            let q = softmax(&model.posterior.forward(&features));
            model.current_latent = sample(&q, &mut model.rng).1;
            result.push((q, model.current_latent.clone()));
        }
        result
    }

    fn surrogate_objective(
        model: &mut WorldModel,
        episode: &BoxesEpisode,
        reference: &[(Vec<f32>, Vec<f32>)],
    ) -> f32 {
        model.block.reset_recurrent_state();
        model.current_latent.fill(0.0);
        let mut loss = 0.0;
        for (t, (reference_q, reference_latent)) in reference.iter().enumerate() {
            let target = if t == 0 {
                None
            } else {
                Some(&episode.transitions[t - 1])
            };
            let observation = target.map_or(episode.initial, |t| t.observation);
            let features = model.input_features(
                (t == 0).then_some(episode.initial),
                target.map(|t| t.action),
            );
            model.advance(&features);
            let features: Vec<f32> = model
                .current_hidden
                .iter()
                .copied()
                .chain(observation.features(model.config.cells()))
                .collect();
            let q = softmax(&model.posterior.forward(&features));
            let p = softmax(&model.prior.forward(&model.current_hidden));
            model.current_latent = reference_latent
                .iter()
                .zip(&q)
                .zip(reference_q)
                .map(|((&z, &q), &r)| z + q - r)
                .collect();
            let decoded: Vec<f32> = model
                .current_hidden
                .iter()
                .copied()
                .chain(model.current_latent.iter().copied())
                .collect();
            loss += reconstruction(
                &mut model.output,
                &decoded,
                observation,
                target,
                model.config.cells(),
                0.0,
                model.config.auxiliary_weight,
            )
            .0;
            loss += model.config.kl_weight
                * q.iter()
                    .zip(p)
                    .map(|(&q, p)| q * (log_probability(q) - log_probability(p)))
                    .sum::<f32>();
        }
        loss / reference.len() as f32
    }

    fn gradient_family(model: &WorldModel, family: usize) -> &ParamMatrix {
        match family {
            0 => &model.block.a_mat,
            1 => &model.block.w_b,
            2 => &model.input.weight,
            3 => &model.posterior.weight,
            4 => &model.prior.weight,
            _ => &model.output.weight,
        }
    }

    fn gradient_family_mut(model: &mut WorldModel, family: usize) -> &mut ParamMatrix {
        match family {
            0 => &mut model.block.a_mat,
            1 => &mut model.block.w_b,
            2 => &mut model.input.weight,
            3 => &mut model.posterior.weight,
            4 => &mut model.prior.weight,
            _ => &mut model.output.weight,
        }
    }

    #[test]
    fn recurrent_straight_through_vjp_matches_local_surrogate_finite_differences() {
        let config = tiny_config();
        let episode = push_episode();
        let mut reference_model = WorldModel::new(&config, 47).unwrap();
        let reference = reference_latents(&mut reference_model, &episode);
        let mut analytic_model = WorldModel::new(&config, 47).unwrap();
        let objective = analytic_model.episode_gradients(&episode).unwrap();
        assert!(
            (objective - surrogate_objective(&mut reference_model, &episode, &reference)).abs()
                < 1e-5
        );
        let epsilon = 0.003;
        for family in 0..6 {
            let gradients = &gradient_family(&analytic_model, family).grad;
            let index = gradients
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap()
                .0;
            let analytic = gradients[index];
            assert!(
                analytic.abs() > 1e-7,
                "family {family} must receive a real gradient"
            );
            let original = gradient_family(&reference_model, family).data[index];
            gradient_family_mut(&mut reference_model, family).data[index] = original + epsilon;
            let plus = surrogate_objective(&mut reference_model, &episode, &reference);
            gradient_family_mut(&mut reference_model, family).data[index] = original - epsilon;
            let minus = surrogate_objective(&mut reference_model, &episode, &reference);
            gradient_family_mut(&mut reference_model, family).data[index] = original;
            let numerical = (plus - minus) / (2.0 * epsilon);
            let tolerance = 0.0003 + 0.03 * analytic.abs().max(numerical.abs());
            assert!(
                (analytic - numerical).abs() < tolerance,
                "family={family} analytic={analytic} numerical={numerical}"
            );
        }
    }

    struct EvaluationSpy {
        phase: usize,
        assimilations: usize,
        calls: Vec<(usize, usize)>,
        initial: BoxesObservation,
        cells: usize,
    }

    impl EvaluatedModel for EvaluationSpy {
        fn start(&mut self, initial: BoxesObservation, _seed: u64) -> Result<(), String> {
            self.phase += 1;
            self.assimilations = 0;
            self.initial = initial;
            Ok(())
        }
        fn next(&mut self, _action: BoxAction) -> Result<WorldPrediction, String> {
            self.calls.push((self.phase, self.assimilations));
            Ok(WorldPrediction {
                observation: self.initial,
                observation_probabilities: vec![vec![1.0 / self.cells as f32; self.cells]; FIELDS],
                reward_probability: 0.0,
                continue_probability: 0.0,
                latent_category: None,
            })
        }
        fn assimilate(&mut self, _observation: BoxesObservation) -> Result<(), String> {
            self.assimilations += 1;
            Ok(())
        }
    }

    #[test]
    fn evaluation_scores_before_assimilation_and_never_assimilates_rollouts() {
        let episode = push_episode();
        let mut spy = EvaluationSpy {
            phase: 0,
            assimilations: 0,
            calls: Vec::new(),
            initial: episode.initial,
            cells: 9,
        };
        let metrics = evaluate(&mut spy, &[episode], 4).unwrap();
        assert_eq!(&spy.calls[..4], &[(1, 0), (1, 1), (1, 2), (1, 3)]);
        assert_eq!(&spy.calls[4..], &[(2, 0), (2, 0), (2, 0), (2, 0)]);
        assert_eq!(metrics.transitions, 4);
        assert_eq!(metrics.rollout_transitions, 4); // Even with continuation=0.
        assert_eq!(metrics.rollout_counts_by_step, vec![1, 1, 1, 1]);
    }

    #[test]
    fn comparison_uses_disjoint_heldout_seeds_and_equal_training_budgets() {
        let mut config = tiny_config();
        config.seed = u64::MAX - 2;
        let mut events = Vec::new();
        let result = run_world_model_with_progress(&config, |epoch, stochastic, plain| {
            assert!(stochastic.is_finite() && plain.is_finite());
            events.push(epoch);
        })
        .unwrap();
        assert_eq!(events, (1..=config.epochs).collect::<Vec<_>>());
        assert!(
            result
                .train_seeds
                .iter()
                .all(|s| !result.heldout_seeds.contains(s))
        );
        assert_eq!(
            result.stochastic.optimizer_updates,
            config.train_episodes * config.epochs
        );
        assert_eq!(
            result.stochastic.optimizer_updates,
            result.autoregressive.optimizer_updates
        );
        assert_eq!(
            result.stochastic.training_transitions,
            result.autoregressive.training_transitions
        );
        assert_eq!(
            result.stochastic.heldout.transitions,
            result.autoregressive.heldout.transitions
        );
        assert_eq!(
            result.stochastic.heldout.rollout_counts_by_step,
            result.autoregressive.heldout.rollout_counts_by_step
        );
        for model in [&result.stochastic, &result.autoregressive] {
            assert!(model.first_epoch_objective.is_finite());
            assert!(model.last_epoch_objective.is_finite());
            assert!(model.heldout.one_step_nll.is_finite());
            assert!(model.heldout.rollout_nll.is_finite());
            assert!(model.backbone_update_l2 > 0.0);
        }
        assert!(result.report().contains("NOT parameter- or time-matched"));
    }
}
