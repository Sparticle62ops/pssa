use pssa::cli::CLIHandler;
use pssa::dataset::Tokenizer;
use rayon::ThreadPoolBuilder;
use std::fs;
use std::path::PathBuf;

fn legacy_documents(
    raw: &str,
    tokenizer: &Tokenizer,
    limit: Option<usize>,
    skip: usize,
) -> Result<Vec<Vec<usize>>, String> {
    let encoded: Vec<Vec<usize>> = raw
        .lines()
        .map(|line| tokenizer.try_encode(line, true))
        .collect::<Result<_, _>>()?;
    let nonempty: Vec<&[usize]> = encoded
        .iter()
        .map(Vec::as_slice)
        .filter(|ids| !ids.is_empty())
        .collect();
    let total = nonempty.iter().try_fold(0usize, |sum, ids| {
        sum.checked_add(ids.len())
            .ok_or_else(|| "dataset token count overflow".to_string())
    })?;
    if total < 2 {
        return Err("dataset has no token transitions".into());
    }
    let mut remaining = limit.unwrap_or(total.saturating_sub(skip % total));
    if remaining == 0 {
        return Err("dataset has no token transitions in the selected window".into());
    }
    let mut offset = skip % total;
    let mut doc_index = 0;
    while offset >= nonempty[doc_index].len() {
        offset -= nonempty[doc_index].len();
        doc_index = (doc_index + 1) % nonempty.len();
    }
    let mut docs = Vec::new();
    while remaining > 0 {
        let ids = nonempty[doc_index];
        let take = (ids.len() - offset).min(remaining);
        if take >= 2 {
            docs.push(ids[offset..offset + take].to_vec());
        }
        remaining -= take;
        doc_index = (doc_index + 1) % nonempty.len();
        offset = 0;
        if limit.is_none() && doc_index == 0 {
            break;
        }
    }
    if docs.is_empty() {
        Err("dataset has no token transitions in the selected window".into())
    } else {
        Ok(docs)
    }
}

fn scratch(name: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("pssa-token-cache-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let cache = root.join("corpus.pssatok");
    (root, cache)
}

#[test]
fn lazy_windows_match_the_old_selection_for_wraps_empty_lines_and_many_pairs() {
    let raw = "alpha beta\n\ngamma delta epsilon\n \nzeta\neta theta iota kappa\n";
    let tokenizer = Tokenizer::from_corpus(raw, true).unwrap();
    let limits = [None, Some(0), Some(1), Some(2), Some(3), Some(5), Some(11)];
    for skip in 0..24 {
        for limit in limits {
            let expected = legacy_documents(raw, &tokenizer, limit, skip);
            let actual = CLIHandler::documents(raw, &tokenizer, limit, skip);
            assert_eq!(actual, expected, "limit={limit:?} skip={skip}");
        }
    }
    for (limit, skip) in [(Some(4), 24), (Some(8), 19), (Some(13), 40), (None, 40)] {
        assert_eq!(
            CLIHandler::documents(raw, &tokenizer, limit, skip),
            legacy_documents(raw, &tokenizer, limit, skip),
            "wrap limit={limit:?} skip={skip}"
        );
    }
}

#[test]
fn token_cache_round_trip_reuses_the_same_window() {
    let (root, cache) = scratch("round-trip");
    let raw = "alpha beta gamma\n\ndelta epsilon\nzeta eta theta\n";
    let tokenizer = Tokenizer::from_corpus(raw, true).unwrap();
    let expected = legacy_documents(raw, &tokenizer, Some(7), 3).unwrap();
    let first =
        CLIHandler::documents_with_cache(raw, &tokenizer, Some(7), 3, Some(&cache), None).unwrap();
    let bytes = fs::read(&cache).unwrap();
    let second =
        CLIHandler::documents_with_cache(raw, &tokenizer, Some(7), 3, Some(&cache), None).unwrap();
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(
        fs::read(&cache).unwrap(),
        bytes,
        "reuse must not rewrite the cache"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn stale_and_corrupt_caches_are_rebuilt_atomically() {
    let (root, cache) = scratch("rebuild");
    let raw = "alpha beta gamma\ndelta epsilon zeta\n";
    let changed = "alpha beta gamma\ndelta epsilon zeta eta\n";
    let tokenizer = Tokenizer::from_corpus(changed, true).unwrap();
    let _ = CLIHandler::documents_with_cache(raw, &tokenizer, Some(5), 0, Some(&cache), None);
    let stale_bytes = fs::read(&cache).unwrap();
    let expected_changed = legacy_documents(changed, &tokenizer, Some(5), 0).unwrap();
    let actual_changed =
        CLIHandler::documents_with_cache(changed, &tokenizer, Some(5), 0, Some(&cache), None)
            .unwrap();
    assert_eq!(actual_changed, expected_changed);
    assert_ne!(
        fs::read(&cache).unwrap(),
        stale_bytes,
        "changed input must rebuild"
    );

    fs::write(&cache, b"not a token cache").unwrap();
    let rebuilt =
        CLIHandler::documents_with_cache(changed, &tokenizer, Some(5), 0, Some(&cache), None)
            .unwrap();
    assert_eq!(rebuilt, expected_changed);
    assert_ne!(fs::read(&cache).unwrap(), b"not a token cache");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn large_bpe_training_window_is_safe_with_and_without_cache_on_rayon_threads() {
    const LINE_COUNT: usize = 250_000;
    let mut raw = String::with_capacity(LINE_COUNT * 30);
    for _ in 0..LINE_COUNT {
        raw.push_str("alpha beta gamma delta\n");
    }
    let tokenizer = Tokenizer::from_corpus_bpe("alpha beta gamma delta epsilon\n", 300).unwrap();
    let (root, cache) = scratch("large-bpe");
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let (uncached, cached) = pool.install(|| {
        assert_eq!(rayon::current_num_threads(), 4);
        let uncached =
            CLIHandler::documents_with_cache(&raw, &tokenizer, None, 37, None, None).unwrap();
        let cached =
            CLIHandler::documents_with_cache(&raw, &tokenizer, None, 37, Some(&cache), None)
                .unwrap();
        (uncached, cached)
    });

    // Skipping 37 tokens drops the first few documents, so expect almost every line.
    assert!(uncached.len() > LINE_COUNT - 64 && uncached.len() <= LINE_COUNT);
    assert_eq!(cached, uncached);
    fs::remove_dir_all(root).unwrap();
}
