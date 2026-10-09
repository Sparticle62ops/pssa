//! Opt-in boxes-world entry point. No text-training options or checkpoint I/O.
use super::Parsed;
use crate::world_model::{WorldModelConfig, run_world_model_with_progress};
use std::io::{self, Write};

pub(crate) fn parse(args: &[String]) -> Result<WorldModelConfig, String> {
    let p = Parsed::parse(
        args,
        &[
            "--seed",
            "--side",
            "--hidden",
            "--state",
            "--categories",
            "--train-episodes",
            "--heldout-episodes",
            "--horizon",
            "--rollout-horizon",
            "--epochs",
            "--lr",
            "--kl-weight",
            "--auxiliary-weight",
            "--grad-clip",
        ],
    )?;
    if !p.positional.is_empty() {
        return Err("world-model takes no corpus or positional arguments".into());
    }
    let d = WorldModelConfig::default();
    let config = WorldModelConfig {
        seed: p.string("--seed", "").map_or(Ok(d.seed), |s| {
            s.parse::<u64>()
                .map_err(|_| "--seed must be an unsigned 64-bit integer".to_string())
        })?,
        side: p.usize_nonzero("--side", "", d.side)?,
        hidden: p.usize_nonzero("--hidden", "", d.hidden)?,
        state: p.usize_nonzero("--state", "", d.state)?,
        categories: p.usize_nonzero("--categories", "", d.categories)?,
        train_episodes: p.usize_nonzero("--train-episodes", "", d.train_episodes)?,
        heldout_episodes: p.usize_nonzero("--heldout-episodes", "", d.heldout_episodes)?,
        horizon: p.usize_nonzero("--horizon", "", d.horizon)?,
        rollout_horizon: p.usize_nonzero("--rollout-horizon", "", d.rollout_horizon)?,
        epochs: p.usize_nonzero("--epochs", "", d.epochs)?,
        learning_rate: p.f32("--lr", "", d.learning_rate)?,
        kl_weight: p.f32("--kl-weight", "", d.kl_weight)?,
        auxiliary_weight: p.f32("--auxiliary-weight", "", d.auxiliary_weight)?,
        gradient_clip: p.f32("--grad-clip", "", d.gradient_clip)?,
    };
    config.validate()?;
    Ok(config)
}

pub(super) fn help() {
    println!("Usage: pssa world-model [OPTIONS]");
    println!(
        "Bounded CPU boxes-world comparison: stochastic RSSM-style PSSA vs plain autoregressive PSSA."
    );
    println!("No corpus, checkpoint, actor/critic, diffusion, or changes to default training.");
    println!("  --seed N                 unsigned seed (73)");
    println!("  --side N                 grid side, 3..6 (4)");
    println!("  --hidden N --state N     recurrent widths, 4..48 / 1..8 (16 / 2)");
    println!("  --categories N           categorical latent size, 2..32 (16)");
    println!("  --train-episodes N        1..128 (32)");
    println!("  --heldout-episodes N      1..64 (12), disjoint episode seeds");
    println!("  --horizon N              episode limit, 1..24 (12)");
    println!("  --rollout-horizon N      open-loop limit, 1..horizon (8)");
    println!("  --epochs N               1..12 (3); combined CPU work is also capped");
    println!("  --lr F                   0.00001..0.03 (0.003)");
    println!("  --kl-weight F            0.001..2 (0.1)");
    println!("  --auxiliary-weight F     reward/continuation BCE weight, 0..2 (0.25)");
    println!("  --grad-clip F            global norm limit, 0.1..20 (5)");
    println!("Example: pssa world-model --seed 73 | pssa tui");
}

pub(super) fn execute(args: &[String]) -> Result<(), String> {
    let config = parse(args)?;
    println!(
        "world_model=start seed={} side={} epochs={} train_episodes={} heldout_episodes={} horizon={} rollout_horizon={} backend=cpu checkpoint=none",
        config.seed,
        config.side,
        config.epochs,
        config.train_episodes,
        config.heldout_episodes,
        config.horizon,
        config.rollout_horizon
    );
    let _ = io::stdout().flush();
    let result = run_world_model_with_progress(&config, |epoch, stochastic, plain| {
        println!(
            "world_model=epoch epoch={epoch} epochs={} stochastic_objective={stochastic:.6} autoregressive_objective={plain:.6}",
            config.epochs
        );
        let _ = io::stdout().flush();
    });
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            println!("world_model=failed checkpoint=none");
            return Err(error);
        }
    };
    for (name, model) in [
        ("stochastic", &result.stochastic),
        ("autoregressive", &result.autoregressive),
    ] {
        let m = &model.heldout;
        println!(
            "world_model=result variant={name} parameters={} updates={} transitions={} train_ms={:.3} one_step_nll={:.6} rollout_nll={:.6} field_accuracy={:.6} rollout_field_accuracy={:.6} exact_accuracy={:.6} reward_brier={:.6} continue_brier={:.6}",
            model.parameters,
            model.optimizer_updates,
            model.training_transitions,
            model.training_ms,
            m.one_step_nll,
            m.rollout_nll,
            m.one_step_field_accuracy,
            m.rollout_field_accuracy,
            m.one_step_exact_accuracy,
            m.reward_brier,
            m.continue_brier
        );
    }
    print!("{}", result.report());
    println!("world_model=complete checkpoint=none");
    Ok(())
}
