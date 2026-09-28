//! Append-safe, dependency-free loss logging shared by training runtimes.
//!
//! `tokens` passed to [`LossCsv::record`] is the number of supervised next-token
//! targets in that optimizer update, not a cumulative count or input length.
//! `loss_sum` is their summed loss, and `updates` is the global optimizer count.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

const HEADER: &str = "tokens_seen,updates,loss,tokens_per_second";

#[derive(Debug)]
pub struct LossCsv {
    file: File,
    path: String,
    every: usize,
    tokens_seen: usize,
    updates: usize,
    pending_tokens: usize,
    pending_loss: f64,
    interval_start: Instant,
    // A failed write may have reached disk partially or completely. Never retry
    // it automatically, since that could duplicate a row or extend a torn row.
    write_error: Option<String>,
}

impl LossCsv {
    /// Open a new CSV or validate every row before appending to an existing one.
    ///
    /// `every` is a positive interval in supervised targets. Existing rows must
    /// end at `prior_updates`; an explicit `tokens_seen` must match their tail.
    /// A new, empty, or header-only file defaults to zero targets only for a
    /// fresh run. Resuming into one requires an explicit cumulative count.
    /// Use only one writer per CSV file.
    pub fn open(
        path: &str,
        every: usize,
        prior_updates: usize,
        tokens_seen: Option<usize>,
    ) -> Result<Self, String> {
        if every == 0 {
            return Err("loss CSV interval must be greater than zero".into());
        }
        let mut file = match OpenOptions::new().read(true).append(true).open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                // Validate before creating a file, even when the origin is
                // missing. create_new also avoids overwriting a racing creator.
                Self::new_origin(prior_updates, tokens_seen)?;
                OpenOptions::new()
                    .read(true)
                    .append(true)
                    .create_new(true)
                    .open(path)
                    .map_err(|error| Self::open_error(path, error))?
            }
            Err(error) => return Err(Self::open_error(path, error)),
        };
        let (has_header, tail) = Self::validate(&file, path)?;
        let origin = match tail {
            Some((tail_tokens, tail_updates)) => {
                if tail_updates != prior_updates {
                    return Err(format!(
                        "loss CSV '{path}' ends at update {tail_updates}, not checkpoint update \
                         {prior_updates}; choose the matching CSV or a new path with explicit tokens_seen (--tokens-seen)"
                    ));
                }
                if let Some(requested) = tokens_seen
                    && requested != tail_tokens
                {
                    return Err(format!(
                        "loss CSV '{path}' ends at tokens_seen={tail_tokens}, not {requested}; \
                         use the CSV's token count or choose a new path"
                    ));
                }
                tail_tokens
            }
            None => Self::new_origin(prior_updates, tokens_seen)?,
        };
        if !has_header {
            writeln!(file, "{HEADER}")
                .and_then(|()| file.flush())
                .map_err(|error| format!("cannot write loss CSV header to '{path}': {error}"))?;
        }
        Ok(Self {
            file,
            path: path.into(),
            every,
            tokens_seen: origin,
            updates: prior_updates,
            pending_tokens: 0,
            pending_loss: 0.0,
            interval_start: Instant::now(),
            write_error: None,
        })
    }

    /// Record one completed optimizer update. Emit at most one row: the first
    /// completed update crossing the next global multiple of `every` targets.
    /// Loss and throughput cover only targets accumulated since this process's
    /// previous row (or `open`), never an earlier process's measurements.
    pub fn record(&mut self, tokens: usize, updates: usize, loss_sum: f64) -> Result<(), String> {
        self.record_at(tokens, updates, loss_sum, Instant::now())
    }

    /// Emit any residual interval. Repeated calls without new records are no-ops.
    /// Call explicitly on successful shutdown; dropping the logger does not emit.
    pub fn finish(&mut self) -> Result<(), String> {
        self.finish_at(Instant::now())
    }

    fn record_at(
        &mut self,
        tokens: usize,
        updates: usize,
        loss_sum: f64,
        now: Instant,
    ) -> Result<(), String> {
        self.check_writer()?;
        if tokens == 0 {
            return Err("loss CSV record requires at least one supervised target".into());
        }
        if updates <= self.updates {
            return Err(format!(
                "loss CSV update {updates} must be greater than the previous global update {}",
                self.updates
            ));
        }
        if !loss_sum.is_finite() || loss_sum < 0.0 {
            return Err("loss CSV loss_sum must be finite and nonnegative".into());
        }
        let tokens_seen = self
            .tokens_seen
            .checked_add(tokens)
            .ok_or("loss CSV tokens_seen counter overflow")?;
        let pending_tokens = self
            .pending_tokens
            .checked_add(tokens)
            .ok_or("loss CSV interval token counter overflow")?;
        let pending_loss = self.pending_loss + loss_sum;
        if !pending_loss.is_finite() {
            return Err("loss CSV accumulated loss_sum overflow; use a shorter interval".into());
        }
        // Comparing buckets avoids overflowing a fabricated next-boundary
        // counter, and skips every boundary crossed by the same update.
        let emit = tokens_seen / self.every > self.tokens_seen / self.every;
        if emit {
            self.emit(tokens_seen, updates, pending_tokens, pending_loss, now)?;
        }
        self.tokens_seen = tokens_seen;
        self.updates = updates;
        self.pending_tokens = if emit { 0 } else { pending_tokens };
        self.pending_loss = if emit { 0.0 } else { pending_loss };
        Ok(())
    }

    fn finish_at(&mut self, now: Instant) -> Result<(), String> {
        self.check_writer()?;
        if self.pending_tokens != 0 {
            self.emit(
                self.tokens_seen,
                self.updates,
                self.pending_tokens,
                self.pending_loss,
                now,
            )?;
            self.pending_tokens = 0;
            self.pending_loss = 0.0;
        }
        Ok(())
    }

    fn emit(
        &mut self,
        tokens_seen: usize,
        updates: usize,
        tokens: usize,
        loss_sum: f64,
        now: Instant,
    ) -> Result<(), String> {
        let elapsed = now
            .checked_duration_since(self.interval_start)
            .ok_or("loss CSV elapsed clock moved backwards")?;
        // A zero-duration interval can occur at the clock's resolution. Bound
        // it to one nanosecond rather than writing an infinite throughput.
        let seconds = elapsed.as_secs_f64().max(1e-9);
        let loss = loss_sum / tokens as f64;
        let rate = tokens as f64 / seconds;
        if !loss.is_finite() || !rate.is_finite() {
            return Err("loss CSV interval produced a nonfinite loss or throughput".into());
        }
        let row = format!("{tokens_seen},{updates},{loss},{rate}\n");
        if let Err(error) = self
            .file
            .write_all(row.as_bytes())
            .and_then(|()| self.file.flush())
        {
            let message = format!(
                "cannot append loss CSV '{}': {error}; inspect the file before reopening it",
                self.path
            );
            self.write_error = Some(message.clone());
            return Err(message);
        }
        self.interval_start = now;
        Ok(())
    }

    fn check_writer(&self) -> Result<(), String> {
        match &self.write_error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn new_origin(prior_updates: usize, tokens_seen: Option<usize>) -> Result<usize, String> {
        if prior_updates != 0 && tokens_seen.is_none() {
            return Err(
                "resuming loss CSV logging into a new, empty, or header-only file requires \
                 explicit tokens_seen; supply --tokens-seen with the cumulative supervised target count"
                    .into(),
            );
        }
        Ok(tokens_seen.unwrap_or(0))
    }

    fn open_error(path: &str, error: io::Error) -> String {
        format!(
            "cannot open loss CSV '{path}': {error}; create the parent directory and choose \
             a writable file path (or check its permissions)"
        )
    }

    fn validate(file: &File, path: &str) -> Result<(bool, Option<(usize, usize)>), String> {
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut line_number = 0;
        let mut tail = None;
        loop {
            line.clear();
            let count = reader
                .read_line(&mut line)
                .map_err(|error| format!("cannot read loss CSV '{path}': {error}"))?;
            if count == 0 {
                return Ok((line_number != 0, tail));
            }
            line_number += 1;
            let invalid = |reason: &str| {
                format!(
                    "invalid loss CSV '{path}' at line {line_number}: {reason}; \
                     restore a complete compatible CSV or choose a new path"
                )
            };
            let content = line
                .strip_suffix('\n')
                .ok_or_else(|| invalid("truncated tail (missing final newline)"))?;
            let content = content.strip_suffix('\r').unwrap_or(content);
            if line_number == 1 {
                if content != HEADER {
                    return Err(invalid(&format!("expected header '{HEADER}'")));
                }
                continue;
            }
            let columns: Vec<&str> = content.split(',').collect();
            if columns.len() != 4 {
                return Err(invalid("expected exactly four columns"));
            }
            let integer = |value: &str| -> Result<usize, String> {
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(invalid("tokens_seen and updates must be unsigned integers"));
                }
                value
                    .parse()
                    .map_err(|_| invalid("integer counter overflow"))
            };
            let tokens = integer(columns[0])?;
            let updates = integer(columns[1])?;
            for value in &columns[2..] {
                let value: f64 = value
                    .parse()
                    .map_err(|_| invalid("loss and tokens_per_second must be numbers"))?;
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid(
                        "loss and tokens_per_second must be finite and nonnegative",
                    ));
                }
            }
            if let Some((previous_tokens, previous_updates)) = tail
                && (tokens <= previous_tokens || updates <= previous_updates)
            {
                return Err(invalid(
                    "tokens_seen and updates must both strictly increase",
                ));
            }
            tail = Some((tokens, updates));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct TestFile {
        dir: PathBuf,
        path: PathBuf,
    }

    impl TestFile {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            loop {
                let dir = std::env::temp_dir().join(format!(
                    "oxide-loss-csv-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&dir) {
                    Ok(()) => {
                        return Self {
                            path: dir.join("loss.csv"),
                            dir,
                        };
                    }
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("cannot create test directory: {error}"),
                }
            }
        }

        fn path(&self) -> &str {
            self.path.to_str().unwrap()
        }

        fn seed(&self, contents: &str) {
            fs::write(&self.path, contents).unwrap();
        }

        fn contents(&self) -> String {
            fs::read_to_string(&self.path).unwrap()
        }

        fn rows(&self) -> Vec<(usize, usize, f64, f64)> {
            let contents = self.contents();
            assert!(contents.ends_with('\n'));
            let mut lines = contents.lines();
            assert_eq!(lines.next(), Some(HEADER));
            lines
                .map(|line| {
                    let columns: Vec<_> = line.split(',').collect();
                    assert_eq!(columns.len(), 4);
                    (
                        columns[0].parse().unwrap(),
                        columns[1].parse().unwrap(),
                        columns[2].parse().unwrap(),
                        columns[3].parse().unwrap(),
                    )
                })
                .collect()
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn near(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn new_file_has_exact_flushed_header_and_empty_finish_is_idempotent() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 0, None).unwrap();
        assert_eq!(file.contents(), format!("{HEADER}\n"));
        csv.finish().unwrap();
        csv.finish().unwrap();
        assert!(file.rows().is_empty());
    }

    #[test]
    fn cadence_uses_global_targets_and_weighted_interval_loss() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 0, None).unwrap();
        let start = csv.interval_start;
        csv.record_at(3, 1, 3.0, start + Duration::from_secs(1))
            .unwrap();
        assert!(file.rows().is_empty());
        csv.record_at(8, 2, 32.0, start + Duration::from_secs(4))
            .unwrap();
        let first = file.rows()[0];
        assert_eq!((first.0, first.1), (11, 2));
        near(first.2, 35.0 / 11.0);
        near(first.3, 11.0 / 4.0);
        csv.record_at(9, 3, 18.0, start + Duration::from_secs(7))
            .unwrap();
        assert_eq!(file.rows()[1], (20, 3, 2.0, 3.0));
        csv.finish_at(start + Duration::from_secs(9)).unwrap();
        assert_eq!(file.rows().len(), 2);
    }

    #[test]
    fn one_large_update_emits_one_row_and_skips_crossed_boundaries() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 0, None).unwrap();
        let start = csv.interval_start;
        csv.record_at(35, 1, 70.0, start + Duration::from_secs(7))
            .unwrap();
        assert_eq!(file.rows(), vec![(35, 1, 2.0, 5.0)]);
        csv.record_at(4, 2, 4.0, start + Duration::from_secs(8))
            .unwrap();
        assert_eq!(file.rows().len(), 1);
        csv.record_at(1, 3, 6.0, start + Duration::from_secs(9))
            .unwrap();
        assert_eq!(file.rows()[1], (40, 3, 2.0, 2.5));
    }

    #[test]
    fn finish_flushes_only_residual_interval_and_does_not_change_global_cadence() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 0, None).unwrap();
        let start = csv.interval_start;
        csv.record_at(4, 1, 12.0, start + Duration::from_secs(2))
            .unwrap();
        csv.finish_at(start + Duration::from_secs(5)).unwrap();
        assert_eq!(file.rows(), vec![(4, 1, 3.0, 0.8)]);
        csv.finish_at(start + Duration::from_secs(8)).unwrap();
        assert_eq!(file.rows().len(), 1);
        csv.record_at(6, 2, 6.0, start + Duration::from_secs(11))
            .unwrap();
        assert_eq!(file.rows()[1], (10, 2, 1.0, 1.0));
        csv.record_at(2, 3, 8.0, start + Duration::from_secs(12))
            .unwrap();
        csv.finish_at(start + Duration::from_secs(15)).unwrap();
        assert_eq!(file.rows()[2], (12, 3, 4.0, 0.5));
        csv.finish().unwrap();
        assert_eq!(file.rows().len(), 3);
    }

    #[test]
    fn resume_appends_without_repeating_header_or_historical_measurements() {
        let file = TestFile::new();
        let original = format!("{HEADER}\n10,2,99,100\n17,5,88,200\n");
        file.seed(&original);
        let mut csv = LossCsv::open(file.path(), 10, 5, Some(17)).unwrap();
        let start = csv.interval_start;
        csv.record_at(2, 6, 2.0, start + Duration::from_secs(1))
            .unwrap();
        assert_eq!(file.contents(), original);
        csv.record_at(1, 7, 4.0, start + Duration::from_secs(3))
            .unwrap();
        assert!(file.contents().starts_with(&original));
        assert_eq!(file.rows()[2], (20, 7, 2.0, 1.0));
        drop(csv);
        let mut csv = LossCsv::open(file.path(), 10, 7, None).unwrap();
        csv.finish().unwrap();
        assert_eq!(file.rows().len(), 3);
    }

    #[test]
    fn resume_without_rows_requires_explicit_origin_before_writing() {
        for initial in [None, Some(String::new()), Some(format!("{HEADER}\n"))] {
            let file = TestFile::new();
            if let Some(contents) = &initial {
                file.seed(contents);
            }
            let error = LossCsv::open(file.path(), 10, 7, None).unwrap_err();
            assert!(error.contains("explicit tokens_seen"), "{error}");
            match &initial {
                Some(contents) => assert_eq!(&file.contents(), contents),
                None => assert!(!file.path.exists()),
            }
            let mut csv = LossCsv::open(file.path(), 10, 7, Some(123)).unwrap();
            let start = csv.interval_start;
            csv.record_at(7, 8, 14.0, start + Duration::from_secs(2))
                .unwrap();
            assert_eq!(file.rows(), vec![(130, 8, 2.0, 3.5)]);
        }
    }

    #[test]
    fn empty_and_header_only_fresh_files_default_to_zero() {
        for contents in [String::new(), format!("{HEADER}\n")] {
            let file = TestFile::new();
            file.seed(&contents);
            let mut csv = LossCsv::open(file.path(), 1, 0, None).unwrap();
            let start = csv.interval_start;
            csv.record_at(1, 1, 0.0, start + Duration::from_secs(1))
                .unwrap();
            assert_eq!(file.rows(), vec![(1, 1, 0.0, 1.0)]);
        }
    }

    #[test]
    fn explicit_zero_resume_origin_and_fresh_nonzero_origin_are_supported() {
        for (prior_updates, origin, targets) in [(7, 0, 10), (0, 17, 3)] {
            let file = TestFile::new();
            let mut csv = LossCsv::open(file.path(), 10, prior_updates, Some(origin)).unwrap();
            let start = csv.interval_start;
            csv.record_at(
                targets,
                prior_updates + 1,
                0.0,
                start + Duration::from_secs(1),
            )
            .unwrap();
            let row = file.rows()[0];
            assert_eq!((row.0, row.1), (origin + targets, prior_updates + 1));
        }
    }

    #[test]
    fn mismatched_checkpoint_or_explicit_tokens_does_not_modify_csv() {
        let file = TestFile::new();
        let original = format!("{HEADER}\n17,5,2,3\n");
        file.seed(&original);
        for (updates, tokens) in [
            (4, None),
            (6, None),
            (0, None),
            (5, Some(16)),
            (5, Some(18)),
        ] {
            assert!(LossCsv::open(file.path(), 10, updates, tokens).is_err());
            assert_eq!(file.contents(), original);
        }
    }

    #[test]
    fn zero_interval_is_rejected_without_creating_or_modifying_files() {
        let file = TestFile::new();
        assert!(
            LossCsv::open(file.path(), 0, 0, None)
                .unwrap_err()
                .contains("greater than zero")
        );
        assert!(!file.path.exists());
        file.seed("leave this alone");
        assert!(LossCsv::open(file.path(), 0, 0, None).is_err());
        assert_eq!(file.contents(), "leave this alone");
    }

    #[test]
    fn missing_parent_has_actionable_error_and_is_not_created() {
        let file = TestFile::new();
        let path = file.dir.join("missing").join("loss.csv");
        let error = LossCsv::open(path.to_str().unwrap(), 10, 0, None).unwrap_err();
        assert!(error.contains("create the parent directory"), "{error}");
        assert!(!path.parent().unwrap().exists());
        assert!(LossCsv::open(file.dir.to_str().unwrap(), 10, 0, None).is_err());
    }

    #[test]
    fn rejects_invalid_schema_and_truncated_header_without_modifying_them() {
        for contents in [
            HEADER.to_string(),
            format!("{HEADER}\r"),
            "\n".into(),
            "tokens,updates,loss,tokens_per_second\n".into(),
            "updates,tokens_seen,loss,tokens_per_second\n".into(),
            format!("{HEADER},extra\n"),
            format!(" {HEADER}\n"),
            format!("\u{feff}{HEADER}\n"),
        ] {
            let file = TestFile::new();
            file.seed(&contents);
            assert!(
                LossCsv::open(file.path(), 10, 0, None).is_err(),
                "{contents:?}"
            );
            assert_eq!(file.contents(), contents);
        }
    }

    #[test]
    fn validates_all_rows_not_just_tail_and_preserves_invalid_files() {
        let overflow = format!("{}0", usize::MAX);
        let bad_rows = [
            "1,1,1,1".to_string(), // A parseable but unterminated tail is unsafe.
            "1,1,1,1\r".into(),
            "\n".into(),
            "1,1,1\n".into(),
            "1,1,1,1,1\n".into(),
            ",1,1,1\n".into(),
            "1,,1,1\n".into(),
            "1,1,,1\n".into(),
            "1,1,1,\n".into(),
            "-1,1,1,1\n".into(),
            "+1,1,1,1\n".into(),
            "1,-1,1,1\n".into(),
            "1.5,1,1,1\n".into(),
            "1,1.5,1,1\n".into(),
            " 1,1,1,1\n".into(),
            format!("{overflow},1,1,1\n"),
            format!("1,{overflow},1,1\n"),
            "1,1,NaN,1\n".into(),
            "1,1,inf,1\n".into(),
            "1,1,-inf,1\n".into(),
            "1,1,-0.5,1\n".into(),
            "1,1,1,NaN\n".into(),
            "1,1,1,inf\n".into(),
            "1,1,1,-inf\n".into(),
            "1,1,1,-0.5\n".into(),
            "1,1,1e999,1\n".into(),
            "1,1,1,not-a-number\n".into(),
            "1,1,1,1\n1,2,1,1\n".into(),
            "1,1,1,1\n2,1,1,1\n".into(),
            "2,1,1,1\n1,2,1,1\n".into(),
            "1,2,1,1\n2,1,1,1\n".into(),
            "1,1,1,1\n\n".into(),
            // Invalid earlier row followed by an otherwise valid final row.
            "1,1,NaN,1\n3,3,1,1\n".into(),
        ];
        for rows in bad_rows {
            let file = TestFile::new();
            let contents = format!("{HEADER}\n{rows}");
            file.seed(&contents);
            let error = LossCsv::open(file.path(), 10, 3, None).unwrap_err();
            assert!(error.contains("invalid loss CSV"), "{rows:?}: {error}");
            assert_eq!(file.contents(), contents);
        }
    }

    #[test]
    fn rejects_non_utf8_without_modifying_file() {
        let file = TestFile::new();
        let contents = [format!("{HEADER}\n").as_bytes(), &[0xff, b'\n']].concat();
        fs::write(&file.path, &contents).unwrap();
        assert!(LossCsv::open(file.path(), 10, 0, None).is_err());
        assert_eq!(fs::read(&file.path).unwrap(), contents);
    }

    #[test]
    fn accepts_crlf_and_finite_scientific_notation() {
        let file = TestFile::new();
        let original = format!("{HEADER}\r\n0,0,0,0\r\n10,1,2.5e-1,1e2\r\n");
        file.seed(&original);
        let mut csv = LossCsv::open(file.path(), 10, 1, None).unwrap();
        let start = csv.interval_start;
        csv.record_at(10, 2, 5.0, start + Duration::from_secs(2))
            .unwrap();
        assert!(file.contents().starts_with(&original));
        assert_eq!(file.rows()[2], (20, 2, 0.5, 5.0));
    }

    #[test]
    fn rejects_bad_records_without_consuming_counters_or_loss() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 2, Some(20)).unwrap();
        let start = csv.interval_start;
        for (tokens, updates, loss) in [
            (0, 3, 0.0),
            (1, 2, 1.0),
            (1, 1, 1.0),
            (1, 0, 1.0),
            (1, 3, f64::NAN),
            (1, 3, f64::INFINITY),
            (1, 3, f64::NEG_INFINITY),
            (1, 3, -1.0),
        ] {
            assert!(csv.record_at(tokens, updates, loss, start).is_err());
            assert!(file.rows().is_empty());
            assert_eq!(
                (csv.tokens_seen, csv.updates, csv.pending_tokens),
                (20, 2, 0)
            );
        }
        csv.record_at(4, 3, 8.0, start + Duration::from_secs(1))
            .unwrap();
        assert!(csv.record_at(1, 3, 1.0, start).is_err());
        assert!(csv.record_at(1, 2, 1.0, start).is_err());
        csv.record_at(6, 4, 18.0, start + Duration::from_secs(2))
            .unwrap();
        assert_eq!(file.rows(), vec![(30, 4, 2.6, 5.0)]);
    }

    #[test]
    fn rejects_accumulated_loss_overflow_and_allows_corrected_record() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 0, None).unwrap();
        let start = csv.interval_start;
        csv.record_at(1, 1, f64::MAX, start).unwrap();
        assert!(
            csv.record_at(1, 2, f64::MAX, start)
                .unwrap_err()
                .contains("overflow")
        );
        assert_eq!(
            (csv.tokens_seen, csv.updates, csv.pending_tokens),
            (1, 1, 1)
        );
        csv.record_at(1, 2, 0.0, start + Duration::from_secs(1))
            .unwrap();
        csv.finish_at(start + Duration::from_secs(2)).unwrap();
        assert_eq!(file.rows(), vec![(2, 2, f64::MAX / 2.0, 1.0)]);
    }

    #[test]
    fn rejects_token_overflow_but_handles_last_representable_boundary() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), usize::MAX, 1, Some(usize::MAX - 1)).unwrap();
        let start = csv.interval_start;
        assert!(
            csv.record_at(2, 2, 2.0, start)
                .unwrap_err()
                .contains("overflow")
        );
        csv.record_at(1, 2, 2.0, start + Duration::from_secs(1))
            .unwrap();
        assert_eq!(file.rows(), vec![(usize::MAX, 2, 2.0, 1.0)]);
        assert!(
            csv.record_at(1, 3, 1.0, start)
                .unwrap_err()
                .contains("overflow")
        );
        csv.finish().unwrap();
        assert_eq!(file.rows().len(), 1);
    }

    #[test]
    fn final_partial_interval_needs_no_representable_next_boundary() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 10, 1, Some(usize::MAX - 1)).unwrap();
        let start = csv.interval_start;
        csv.record_at(1, 2, 3.0, start + Duration::from_secs(1))
            .unwrap();
        csv.finish_at(start + Duration::from_secs(2)).unwrap();
        assert_eq!(file.rows(), vec![(usize::MAX, 2, 3.0, 0.5)]);
    }

    #[test]
    fn update_counter_cannot_wrap_or_repeat_at_usize_max() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 1, usize::MAX - 1, Some(0)).unwrap();
        let start = csv.interval_start;
        csv.record_at(1, usize::MAX, 1.0, start + Duration::from_secs(1))
            .unwrap();
        assert!(csv.record_at(1, 0, 1.0, start).is_err());
        assert!(csv.record_at(1, usize::MAX, 1.0, start).is_err());
        assert_eq!(file.rows(), vec![(1, usize::MAX, 1.0, 1.0)]);
    }

    #[test]
    fn zero_elapsed_time_still_writes_finite_throughput() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 1, 0, None).unwrap();
        csv.record_at(1, 1, 1.0, csv.interval_start).unwrap();
        let rate = file.rows()[0].3;
        assert!(rate.is_finite());
        near(rate, 1e9);
    }

    #[test]
    fn backwards_test_clock_is_rejected_without_consuming_record() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 1, 0, None).unwrap();
        let start = csv.interval_start;
        csv.interval_start = start + Duration::from_secs(1);
        assert!(csv.record_at(1, 1, 1.0, start).is_err());
        assert!(file.rows().is_empty());
        csv.record_at(1, 1, 1.0, start + Duration::from_secs(2))
            .unwrap();
        assert_eq!(file.rows(), vec![(1, 1, 1.0, 1.0)]);
    }

    #[test]
    fn write_errors_are_reported_and_poison_writer_against_duplicate_retries() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 1, 0, None).unwrap();
        // Replace the append handle with a read-only handle for portable I/O
        // failure injection, independent of permissions or root privileges.
        csv.file = File::open(file.path()).unwrap();
        let error = csv.record(1, 1, 1.0).unwrap_err();
        assert!(error.contains("cannot append loss CSV"));
        assert_eq!(csv.record(1, 1, 1.0).unwrap_err(), error);
        assert_eq!(csv.finish().unwrap_err(), error);
        assert_eq!(file.contents(), format!("{HEADER}\n"));
    }

    #[test]
    fn public_clock_api_flushes_rows_immediately_and_can_resume() {
        let file = TestFile::new();
        let mut csv = LossCsv::open(file.path(), 2, 0, None).unwrap();
        csv.record(2, 1, 6.0).unwrap();
        assert_eq!(file.rows().len(), 1);
        csv.record(1, 2, 4.0).unwrap();
        csv.finish().unwrap();
        csv.finish().unwrap();
        let rows = file.rows();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].0, rows[0].1, rows[0].2), (2, 1, 3.0));
        assert_eq!((rows[1].0, rows[1].1, rows[1].2), (3, 2, 4.0));
        assert!(rows.iter().all(|row| row.3.is_finite() && row.3 >= 0.0));
        drop(csv);
        LossCsv::open(file.path(), 2, 2, None).unwrap();
    }
}
