//! Opt-in process/workload limits. No flags means no pool or process changes.
//! RAM is Linux RLIMIT_AS (virtual address space), not RSS or GPU VRAM.
use std::process::Command;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    pub threads: Option<usize>,
    pub ram_mib: Option<usize>,
    /// Independent document lanes, forwarded to the existing --batch-size flag.
    pub batch_size: Option<usize>,
    /// Global corpus cap, forwarded to the existing --max-tokens flag.
    pub max_tokens: Option<usize>,
}

impl ResourceLimits {
    /// Blank inputs preserve existing defaults. Never silently coerce bad input.
    pub fn from_inputs(inputs: [&str; 4]) -> Result<Self, String> {
        let parse = |value: &str, flag: &str, max: usize| -> Result<Option<usize>, String> {
            if value.is_empty() {
                return Ok(None);
            }
            value
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=max).contains(n))
                .map(Some)
                .ok_or_else(|| format!("{flag} must be an integer in 1..={max}, or blank"))
        };
        let limits = Self {
            threads: parse(inputs[0], "--threads", 65_536)?,
            ram_mib: parse(inputs[1], "--ram-mib", usize::MAX / (1024 * 1024))?,
            batch_size: parse(inputs[2], "--batch-size", 65_536)?,
            max_tokens: parse(inputs[3], "--max-tokens", usize::MAX)?,
        };
        limits.validate()?;
        Ok(limits)
    }

    pub fn validate(self) -> Result<(), String> {
        if self.threads.is_some_and(|n| n == 0 || n > 65_536) {
            return Err("--threads must be in 1..=65536".into());
        }
        if self.batch_size.is_some_and(|n| n == 0 || n > 65_536) {
            return Err("--batch-size must be in 1..=65536".into());
        }
        if self.max_tokens == Some(0) {
            return Err("--max-tokens must be positive".into());
        }
        if let Some(mib) = self.ram_mib {
            if mib == 0 || mib.checked_mul(1024 * 1024).is_none() {
                return Err("--ram-mib must be a positive MiB budget without overflow".into());
            }
            if !cfg!(target_os = "linux") {
                return Err("--ram-mib requires Linux prlimit; no RAM limit was applied".into());
            }
        }
        Ok(())
    }

    pub fn append_args(self, args: &mut Vec<String>) {
        for (flag, value) in [
            ("--threads", self.threads),
            ("--ram-mib", self.ram_mib),
            ("--batch-size", self.batch_size),
            ("--max-tokens", self.max_tokens),
        ] {
            if let Some(value) = value {
                args.extend([flag.to_owned(), value.to_string()]);
            }
        }
    }

    /// A local Rayon pool also covers nested parallel work in gpu_batch. Do not
    /// mutate process environment or the global pool in a multithreaded TUI.
    /// Batch/tokens are enforced separately by the existing trainer options.
    /// A RAM budget must first be applied by the CLI process wrapper.
    pub fn run<T: Send>(
        self,
        work: impl FnOnce() -> Result<T, String> + Send,
    ) -> Result<T, String> {
        self.validate()?;
        if self.ram_mib.is_some() {
            return Err(
                "RAM limits require a CLI child with --ram-mib, not an in-process worker".into(),
            );
        }
        match self.threads {
            None => work(),
            Some(threads) => rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("cannot create --threads pool: {e}"))?
                .install(work),
        }
    }

    /// Replace the CLI with prlimit + the same command, removing only the RAM
    /// flag to avoid recursion. Argument boundaries (including spaces) survive.
    /// This runs BEFORE loading a checkpoint, tokenizing or allocating a model.
    pub(super) fn apply_process_budget(self, args: &[String]) -> Result<(), String> {
        self.validate()?;
        if self.ram_mib.is_none() {
            return Ok(());
        }
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut command = self.budget_command(&executable, args)?;
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::process::CommandExt;
            let error = command.exec();
            Err(format!(
                "cannot apply --ram-mib: {error}; install util-linux prlimit or omit the flag"
            ))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = &mut command;
            Err("--ram-mib requires Linux prlimit".into())
        }
    }

    fn budget_command(
        self,
        executable: &std::path::Path,
        args: &[String],
    ) -> Result<Command, String> {
        self.validate()?;
        let bytes = self
            .ram_mib
            .ok_or("missing RAM budget")?
            .checked_mul(1024 * 1024)
            .ok_or("RAM budget overflow")?;
        let mut command = Command::new("prlimit");
        command
            .arg(format!("--as={bytes}:{bytes}"))
            .arg("--")
            .arg(executable);
        // The validated parser accepts flag/value pairs, with --no-tui as the
        // only boolean. Walk pairs so a value equal to a flag is never stripped.
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--ram-mib" {
                i += 2;
            } else if args[i].starts_with('-') && args[i] != "--no-tui" {
                command.arg(&args[i]);
                if let Some(value) = args.get(i + 1) {
                    command.arg(value);
                }
                i += 2;
            } else {
                command.arg(&args[i]);
                i += 1;
            }
        }
        Ok(command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_limits_keep_flags_and_default_pool_unchanged() {
        let limits = ResourceLimits::from_inputs([""; 4]).unwrap();
        assert_eq!(limits, ResourceLimits::default());
        let mut args = vec!["train".to_owned(), "--no-tui".to_owned()];
        limits.append_args(&mut args);
        assert_eq!(args, ["train", "--no-tui"]);
        let before = rayon::current_num_threads();
        assert_eq!(
            limits.run(|| Ok(rayon::current_num_threads())).unwrap(),
            before
        );
    }

    #[test]
    fn limits_reject_zero_negative_overflow_and_invalid_numbers() {
        for index in 0..4 {
            for bad in ["0", "-1", "NaN", "1.5", " 2", "184467440737095516160"] {
                let mut inputs = [""; 4];
                inputs[index] = bad;
                assert!(
                    ResourceLimits::from_inputs(inputs).is_err(),
                    "{index}: {bad}"
                );
            }
        }
        assert!(ResourceLimits::from_inputs(["65537", "", "", ""]).is_err());
        assert!(ResourceLimits::from_inputs(["", "", "65537", ""]).is_err());
        assert!(ResourceLimits::from_inputs(["", &usize::MAX.to_string(), "", ""]).is_err());
    }

    #[test]
    fn scoped_thread_limit_is_enforced_and_does_not_change_global_pool() {
        let before = rayon::current_num_threads();
        let limits = ResourceLimits::from_inputs(["2", "", "3", "77"]).unwrap();
        assert_eq!(limits.run(|| Ok(rayon::current_num_threads())).unwrap(), 2);
        assert_eq!(rayon::current_num_threads(), before);
        let mut args = Vec::new();
        limits.append_args(&mut args);
        assert_eq!(
            args,
            ["--threads", "2", "--batch-size", "3", "--max-tokens", "77"]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ram_wrapper_keeps_paths_flags_and_process_budget() {
        let limits = ResourceLimits {
            ram_mib: Some(512),
            ..Default::default()
        };
        let args: Vec<_> = [
            "pssa",
            "train",
            "--ram-mib",
            "512",
            "--resume",
            "some path's/checkpoint.pssa",
            "--threads",
            "2",
            "--no-tui",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let command = limits
            .budget_command(std::path::Path::new("/a path/pssa"), &args)
            .unwrap();
        let actual: Vec<_> = command
            .get_args()
            .map(|s| s.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            actual,
            [
                "--as=536870912:536870912",
                "--",
                "/a path/pssa",
                "train",
                "--resume",
                "some path's/checkpoint.pssa",
                "--threads",
                "2",
                "--no-tui"
            ]
        );
        // A real child verifies the kernel-enforced soft+hard limits, without
        // changing this test runner's budget or provoking an OOM.
        let output = Command::new("prlimit")
            .args([
                "--as=536870912:536870912",
                "--",
                "prlimit",
                "--noheadings",
                "--raw",
                "--as",
                "--output",
                "SOFT,HARD",
            ])
            .output();
        let output = match output {
            Ok(output) => output,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("prlimit unavailable; kernel budget smoke test skipped");
                return;
            }
            Err(e) => panic!("prlimit failed: {e}"),
        };
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stdout).trim(),
            "536870912 536870912"
        );
    }
}
