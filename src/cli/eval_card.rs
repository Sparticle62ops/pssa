//! Opt-in JSON card for the paired runner; existing score output is unchanged.
use super::Parsed;
use crate::evaluation::EvaluationSlice;

fn parse(args: &[String]) -> Result<(String, String, EvaluationSlice, usize), String> {
    let parsed = Parsed::parse(args, &["--model", "--skip-tokens", "--max-tokens", "--loops"])?;
    if parsed.positional.len() != 1 {
        return Err("eval-card requires exactly one local corpus path".into());
    }
    for name in ["--skip-tokens", "--max-tokens"] {
        parsed.string(name, "").ok_or_else(|| format!("eval-card requires {name}"))?;
    }
    Ok((
        parsed.positional[0].clone(),
        parsed.string("--model", "").ok_or("eval-card requires --model PATH")?.into(),
        EvaluationSlice {
            skip_tokens: parsed.required_usize("--skip-tokens", "", 0)?,
            max_tokens: Some(parsed.usize_nonzero("--max-tokens", "", 1)?),
        },
        parsed.loops()?,
    ))
}

pub(super) fn execute(args: &[String]) -> Result<(), String> {
    let (data, model, slice, loops) = parse(args)?;
    let card = crate::eval_harness::checkpoint_card(&data, &model, slice, loops)?;
    println!("{}", serde_json::to_string_pretty(&card).map_err(|e| e.to_string())?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_explicit_split_model_and_valid_options() {
        let args = ["corpus", "--model", "m.pssa", "--skip-tokens", "0", "--max-tokens", "16"]
            .map(str::to_owned);
        assert_eq!(parse(&args).unwrap().2.max_tokens, Some(16));
        for start in [1, 3, 5] {
            let mut missing = args.to_vec();
            missing.drain(start..start + 2);
            assert!(parse(&missing).is_err());
        }
        for suffix in [vec!["--loops", "0"], vec!["--backend", "cuda"], vec!["extra"]] {
            let mut bad = args.to_vec();
            bad.extend(suffix.into_iter().map(str::to_owned));
            assert!(parse(&bad).is_err());
        }
    }
}
