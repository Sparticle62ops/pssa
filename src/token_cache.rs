use crate::dataset::Tokenizer;
use rayon::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 8] = b"PSSATOK\0";
const VERSION: u32 = 1;
const HASH_PREFIX_BYTES: usize = 1024 * 1024;
const TOKEN_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheStatus {
    Disabled,
    Built,
    Reused,
}

impl CacheStatus {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Built => "built",
            Self::Reused => "reused",
        }
    }
}

pub(crate) struct WindowResult {
    pub(crate) docs: Vec<Vec<usize>>,
    pub(crate) status: CacheStatus,
    pub(crate) elapsed: Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CacheKey {
    dataset_size: u64,
    mtime_secs: u64,
    mtime_nanos: u64,
    first_hash: u64,
    last_hash: u64,
    tokenizer_hash: u64,
}

/// Select training documents, using an optional persistent cache. Without a
/// cache this intentionally stops tokenizing as soon as a non-wrapping window
/// is known to be complete. A cache build is the one full, parallel pass that
/// makes subsequent chained windows independent of the corpus size.
pub(crate) fn documents(
    raw: &str,
    tokenizer: &Tokenizer,
    limit: Option<usize>,
    skip: usize,
    cache_path: Option<&Path>,
    source_path: Option<&Path>,
) -> Result<WindowResult, String> {
    let started = Instant::now();
    let result = match cache_path {
        Some(path) => {
            let key = cache_key(raw, tokenizer, source_path);
            if let Some(encoded) = read_cache(path, key) {
                let docs = select_documents(&encoded, limit, skip)?;
                WindowResult {
                    docs,
                    status: CacheStatus::Reused,
                    elapsed: started.elapsed(),
                }
            } else {
                let encoded = tokenize_all_parallel(raw, tokenizer)?;
                write_cache(path, key, &encoded)?;
                let docs = select_documents(&encoded, limit, skip)?;
                WindowResult {
                    docs,
                    status: CacheStatus::Built,
                    elapsed: started.elapsed(),
                }
            }
        }
        None => {
            let collected = tokenize_until_window(raw, tokenizer, limit, skip)?;
            let docs = select_documents(&collected, limit, skip)?;
            WindowResult {
                docs,
                status: CacheStatus::Disabled,
                elapsed: started.elapsed(),
            }
        }
    };
    Ok(result)
}

/// Tokenize lines in input order. Rayon gives each line an independent work
/// item, while its indexed collection preserves the historical document order.
fn tokenize_all_parallel(raw: &str, tokenizer: &Tokenizer) -> Result<Vec<Vec<usize>>, String> {
    raw.lines()
        .collect::<Vec<_>>()
        .par_iter()
        .map(|line| tokenizer.try_encode(line, true))
        .collect()
}

fn tokenize_until_window(
    raw: &str,
    tokenizer: &Tokenizer,
    limit: Option<usize>,
    skip: usize,
) -> Result<Vec<Vec<usize>>, String> {
    // A zero-length or unbounded window needs the full token count to preserve
    // the historical empty-window and EOF-wrap decisions. Overflow likewise
    // forces the full pass because the old implementation accepted usize-sized
    // skip and limit independently.
    let Some(limit) = limit else {
        return tokenize_all_serial(raw, tokenizer);
    };
    let Some(target) = skip.checked_add(limit) else {
        return tokenize_all_serial(raw, tokenizer);
    };
    if limit == 0 || target == 0 {
        return tokenize_all_serial(raw, tokenizer);
    }

    let mut encoded = Vec::new();
    let mut total = 0usize;
    for line in raw.lines() {
        let ids = tokenizer.try_encode(line, true)?;
        if ids.is_empty() {
            continue;
        }
        total = total
            .checked_add(ids.len())
            .ok_or_else(|| "dataset token count overflow".to_string())?;
        encoded.push(ids);
        if total >= target {
            break;
        }
    }
    Ok(encoded)
}

fn tokenize_all_serial(raw: &str, tokenizer: &Tokenizer) -> Result<Vec<Vec<usize>>, String> {
    raw.lines()
        .map(|line| tokenizer.try_encode(line, true))
        .collect()
}

/// This is deliberately the old selection algorithm. Keeping it isolated and
/// unchanged makes the lazy and cached input representations share exactly the
/// same boundary and cyclic-wrap behavior as the historical implementation.
fn select_documents(
    encoded: &[Vec<usize>],
    limit: Option<usize>,
    skip: usize,
) -> Result<Vec<Vec<usize>>, String> {
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

fn cache_key(raw: &str, tokenizer: &Tokenizer, source_path: Option<&Path>) -> CacheKey {
    let metadata = source_path.and_then(|path| fs::metadata(path).ok());
    let dataset_size = metadata
        .as_ref()
        .filter(|metadata| metadata.is_file())
        .map_or(raw.len() as u64, |metadata| metadata.len());
    let (mtime_secs, mtime_nanos) = metadata
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or((0, 0), |duration| {
            (duration.as_secs(), u64::from(duration.subsec_nanos()))
        });
    let bytes = raw.as_bytes();
    CacheKey {
        dataset_size,
        mtime_secs,
        mtime_nanos,
        first_hash: hash_bytes(&bytes[..bytes.len().min(HASH_PREFIX_BYTES)]),
        last_hash: hash_bytes(&bytes[bytes.len().saturating_sub(HASH_PREFIX_BYTES)..]),
        tokenizer_hash: tokenizer.cache_identity_hash(),
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn read_cache(path: &Path, expected: CacheKey) -> Option<Vec<Vec<usize>>> {
    let bytes = fs::read(path).ok()?;
    let mut cursor = 0usize;
    if take(&bytes, &mut cursor, MAGIC.len())? != MAGIC {
        return None;
    }
    if take_u32(&bytes, &mut cursor)? != VERSION {
        return None;
    }
    if take_u32(&bytes, &mut cursor)? != 0 {
        return None;
    }
    let key = CacheKey {
        dataset_size: take_u64(&bytes, &mut cursor)?,
        mtime_secs: take_u64(&bytes, &mut cursor)?,
        mtime_nanos: take_u64(&bytes, &mut cursor)?,
        first_hash: take_u64(&bytes, &mut cursor)?,
        last_hash: take_u64(&bytes, &mut cursor)?,
        tokenizer_hash: take_u64(&bytes, &mut cursor)?,
    };
    if key != expected {
        return None;
    }
    let doc_count = usize::try_from(take_u64(&bytes, &mut cursor)?).ok()?;
    let total = usize::try_from(take_u64(&bytes, &mut cursor)?).ok()?;
    if doc_count > total || (doc_count == 0 && total != 0) {
        return None;
    }
    if doc_count > bytes.len().saturating_sub(cursor) / 16 {
        return None;
    }
    let mut index = Vec::with_capacity(doc_count);
    let mut next_offset = 0usize;
    for _ in 0..doc_count {
        let offset = usize::try_from(take_u64(&bytes, &mut cursor)?).ok()?;
        let len = usize::try_from(take_u64(&bytes, &mut cursor)?).ok()?;
        if len == 0 || offset != next_offset || offset.checked_add(len)? > total {
            return None;
        }
        next_offset = offset + len;
        index.push((offset, len));
    }
    if next_offset != total || total > bytes.len().saturating_sub(cursor) / TOKEN_BYTES {
        return None;
    }
    let token_bytes = total.checked_mul(TOKEN_BYTES)?;
    if bytes.len() - cursor != token_bytes {
        return None;
    }
    let mut tokens = Vec::with_capacity(total);
    for chunk in bytes[cursor..].chunks_exact(TOKEN_BYTES) {
        tokens.push(u32::from_le_bytes(chunk.try_into().ok()?) as usize);
    }
    index
        .into_iter()
        .map(|(offset, len)| Some(tokens[offset..offset + len].to_vec()))
        .collect()
}

fn write_cache(path: &Path, key: CacheKey, encoded: &[Vec<usize>]) -> Result<(), String> {
    let nonempty: Vec<&[usize]> = encoded
        .iter()
        .map(Vec::as_slice)
        .filter(|ids| !ids.is_empty())
        .collect();
    let total = nonempty.iter().try_fold(0usize, |sum, ids| {
        sum.checked_add(ids.len())
            .ok_or_else(|| "dataset token count overflow".to_string())
    })?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("token cache path has no valid filename: {}", path.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let temp_path = parent.join(format!(".{name}.tmp-{}-{stamp}", std::process::id()));
    let result = write_cache_file(&temp_path, key, &nonempty, total)
        .and_then(|()| fs::rename(&temp_path, path).map_err(|error| io_error(path, error)));
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn write_cache_file(
    path: &Path,
    key: CacheKey,
    docs: &[&[usize]],
    total: usize,
) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| io_error(path, error))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(MAGIC)
        .map_err(|error| io_error(path, error))?;
    write_u32(&mut writer, VERSION, path)?;
    write_u32(&mut writer, 0, path)?;
    for value in [
        key.dataset_size,
        key.mtime_secs,
        key.mtime_nanos,
        key.first_hash,
        key.last_hash,
        key.tokenizer_hash,
        docs.len() as u64,
        total as u64,
    ] {
        write_u64(&mut writer, value, path)?;
    }
    let mut offset = 0usize;
    for ids in docs {
        write_u64(&mut writer, offset as u64, path)?;
        write_u64(&mut writer, ids.len() as u64, path)?;
        offset += ids.len();
    }
    for ids in docs {
        for &id in *ids {
            let id = u32::try_from(id)
                .map_err(|_| "token ID does not fit the token cache format".to_string())?;
            writer
                .write_all(&id.to_le_bytes())
                .map_err(|error| io_error(path, error))?;
        }
    }
    writer.flush().map_err(|error| io_error(path, error))?;
    writer
        .into_inner()
        .map_err(|error| io_error(path, error.into_error()))?
        .sync_all()
        .map_err(|error| io_error(path, error))
}

fn write_u32(writer: &mut BufWriter<File>, value: u32, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|error| io_error(path, error))
}

fn write_u64(writer: &mut BufWriter<File>, value: u64, path: &Path) -> Result<(), String> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|error| io_error(path, error))
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = cursor.checked_add(len)?;
    let out = bytes.get(*cursor..end)?;
    *cursor = end;
    Some(out)
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Option<u32> {
    Some(u32::from_le_bytes(take(bytes, cursor, 4)?.try_into().ok()?))
}

fn take_u64(bytes: &[u8], cursor: &mut usize) -> Option<u64> {
    Some(u64::from_le_bytes(take(bytes, cursor, 8)?.try_into().ok()?))
}

fn io_error(path: &Path, error: io::Error) -> String {
    format!("cannot write token cache '{}': {error}", path.display())
}
