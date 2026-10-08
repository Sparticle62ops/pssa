use crate::dataset::Tokenizer;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: &[u8; 8] = b"PSSATOK\0";
const VERSION: u32 = 2;
const HASH_PREFIX_BYTES: usize = 1024 * 1024;
const TOKEN_BYTES: usize = 4;
const CACHE_BUILD_CHUNK_LINES: usize = 4096;

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

/// Origins only for the selected documents, not a full-corpus offset map.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DocumentSelection {
    pub(crate) source_doc: usize,
    pub(crate) token_start: usize,
    pub(crate) byte_start: Option<usize>,
}

pub(crate) struct WindowResult {
    pub(crate) docs: Vec<Vec<usize>>,
    pub(crate) selections: Vec<DocumentSelection>,
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
/// is known to be complete. A cache build is the one full pass that makes
/// subsequent chained windows independent of the corpus size.
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
            if let Some(encoded) = read_cache(path, key, tokenizer.vocab_size) {
                let (docs, selections) = select_documents(&encoded, tokenizer, limit, skip)?;
                WindowResult {
                    docs,
                    selections,
                    status: CacheStatus::Reused,
                    elapsed: started.elapsed(),
                }
            } else {
                // The Hugging Face tokenizer contains mutable added-vocabulary
                // state despite its shared API. Do not call it concurrently:
                // tokenizers 0.22 can corrupt its allocation state when one
                // instance is shared by rayon workers. A cache is opt-in, so a
                // bounded serial build is safer than changing the uncached path
                // or cloning a tokenizer per line.
                write_cache_serial(path, key, raw, tokenizer)?;
                let encoded = read_cache(path, key, tokenizer.vocab_size).ok_or_else(|| {
                    "token cache could not be read after it was built".to_string()
                })?;
                let (docs, selections) = select_documents(&encoded, tokenizer, limit, skip)?;
                WindowResult {
                    docs,
                    selections,
                    status: CacheStatus::Built,
                    elapsed: started.elapsed(),
                }
            }
        }
        None => {
            let collected = tokenize_until_window(raw, tokenizer, limit, skip)?;
            let (docs, selections) = select_documents(&collected, tokenizer, limit, skip)?;
            WindowResult {
                docs,
                selections,
                status: CacheStatus::Disabled,
                elapsed: started.elapsed(),
            }
        }
    };
    Ok(result)
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
    tokenizer: &Tokenizer,
    limit: Option<usize>,
    skip: usize,
) -> Result<(Vec<Vec<usize>>, Vec<DocumentSelection>), String> {
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
    let mut selections = Vec::new();
    while remaining > 0 {
        let ids = nonempty[doc_index];
        let take = (ids.len() - offset).min(remaining);
        if take >= 2 {
            docs.push(ids[offset..offset + take].to_vec());
            let byte_start = if tokenizer.kind() == crate::dataset::TokenizerKind::Bpe {
                ids[..offset].iter().try_fold(0usize, |sum, &id| {
                    sum.checked_add(tokenizer.token_bytes(id)?.len())
                })
            } else {
                None
            };
            selections.push(DocumentSelection {
                source_doc: doc_index,
                token_start: offset,
                byte_start,
            });
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
        Ok((docs, selections))
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
    hash_update(&mut hash, bytes);
    hash
}

fn hash_update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = (*hash).wrapping_mul(0x100000001b3);
    }
}

fn cache_content_hash(index: &[(usize, usize)], payload: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &(offset, len) in index {
        hash_update(&mut hash, &(offset as u64).to_le_bytes());
        hash_update(&mut hash, &(len as u64).to_le_bytes());
    }
    hash_update(&mut hash, payload);
    hash
}

fn read_cache(path: &Path, expected: CacheKey, vocab_size: usize) -> Option<Vec<Vec<usize>>> {
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
    let content_hash = take_u64(&bytes, &mut cursor)?;
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
    let payload = &bytes[cursor..];
    if cache_content_hash(&index, payload) != content_hash {
        return None;
    }
    let mut tokens = Vec::with_capacity(total);
    for chunk in payload.chunks_exact(TOKEN_BYTES) {
        let id = u32::from_le_bytes(chunk.try_into().ok()?);
        if usize::try_from(id).ok()? >= vocab_size {
            return None;
        }
        tokens.push(id as usize);
    }
    index
        .into_iter()
        .map(|(offset, len)| Some(tokens[offset..offset + len].to_vec()))
        .collect()
}

/// Build the v2 cache without retaining the whole encoded corpus. The index
/// and token payload are staged separately because v2 places the complete index
/// before the payload; only a bounded line chunk and buffered file writes stay
/// live during tokenization.
fn write_cache_serial(
    path: &Path,
    key: CacheKey,
    raw: &str,
    tokenizer: &Tokenizer,
) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("token cache path has no valid filename: {}", path.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let suffix = format!("{}-{stamp}", std::process::id());
    let temp_path = parent.join(format!(".{name}.tmp-{suffix}"));
    let index_path = parent.join(format!(".{name}.index-{suffix}"));
    let payload_path = parent.join(format!(".{name}.payload-{suffix}"));

    let result = (|| {
        let index_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&index_path)
            .map_err(|error| io_error(&index_path, error))?;
        let payload_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&payload_path)
            .map_err(|error| io_error(&payload_path, error))?;
        let mut index = BufWriter::new(index_file);
        let mut payload = BufWriter::new(payload_file);
        let mut doc_count = 0usize;
        let mut total = 0usize;
        let mut lines = Vec::with_capacity(CACHE_BUILD_CHUNK_LINES);

        let mut append_chunk = |lines: &mut Vec<&str>| -> Result<(), String> {
            for line in lines.drain(..) {
                let ids = tokenizer.try_encode(line, true)?;
                if ids.is_empty() {
                    continue;
                }
                let offset = total;
                let len = ids.len();
                let new_total = total
                    .checked_add(len)
                    .ok_or_else(|| "dataset token count overflow".to_string())?;
                write_u64(&mut index, offset as u64, &index_path)?;
                write_u64(&mut index, len as u64, &index_path)?;
                for &id in &ids {
                    let id = u32::try_from(id)
                        .map_err(|_| "token ID does not fit the token cache format".to_string())?;
                    let bytes = id.to_le_bytes();
                    payload
                        .write_all(&bytes)
                        .map_err(|error| io_error(&payload_path, error))?;
                }
                doc_count = doc_count
                    .checked_add(1)
                    .ok_or_else(|| "token cache document count overflow".to_string())?;
                total = new_total;
            }
            Ok(())
        };

        for line in raw.lines() {
            lines.push(line);
            if lines.len() == CACHE_BUILD_CHUNK_LINES {
                append_chunk(&mut lines)?;
            }
        }
        append_chunk(&mut lines)?;
        index
            .flush()
            .map_err(|error| io_error(&index_path, error))?;
        payload
            .flush()
            .map_err(|error| io_error(&payload_path, error))?;
        index
            .into_inner()
            .map_err(|error| io_error(&index_path, error.into_error()))?
            .sync_all()
            .map_err(|error| io_error(&index_path, error))?;
        payload
            .into_inner()
            .map_err(|error| io_error(&payload_path, error.into_error()))?
            .sync_all()
            .map_err(|error| io_error(&payload_path, error))?;

        // The v2 content hash covers the whole index, then the whole payload,
        // exactly as read_cache verifies it, so hash the staged files in order.
        let mut content_hash = 0xcbf29ce484222325u64;
        for staged in [&index_path, &payload_path] {
            let mut file = File::open(staged).map_err(|error| io_error(staged, error))?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n =
                    io::Read::read(&mut file, &mut buf).map_err(|error| io_error(staged, error))?;
                if n == 0 {
                    break;
                }
                hash_update(&mut content_hash, &buf[..n]);
            }
        }

        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .map_err(|error| io_error(&temp_path, error))?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(MAGIC)
            .map_err(|error| io_error(&temp_path, error))?;
        write_u32(&mut writer, VERSION, &temp_path)?;
        write_u32(&mut writer, 0, &temp_path)?;
        for value in [
            key.dataset_size,
            key.mtime_secs,
            key.mtime_nanos,
            key.first_hash,
            key.last_hash,
            key.tokenizer_hash,
            u64::try_from(doc_count)
                .map_err(|_| "token cache document count does not fit the format".to_string())?,
            u64::try_from(total)
                .map_err(|_| "token cache token count does not fit the format".to_string())?,
            content_hash,
        ] {
            write_u64(&mut writer, value, &temp_path)?;
        }
        let mut index_file =
            File::open(&index_path).map_err(|error| io_error(&index_path, error))?;
        io::copy(&mut index_file, &mut writer).map_err(|error| io_error(&temp_path, error))?;
        let mut payload_file =
            File::open(&payload_path).map_err(|error| io_error(&payload_path, error))?;
        io::copy(&mut payload_file, &mut writer).map_err(|error| io_error(&temp_path, error))?;
        writer
            .flush()
            .map_err(|error| io_error(&temp_path, error))?;
        writer
            .into_inner()
            .map_err(|error| io_error(&temp_path, error.into_error()))?
            .sync_all()
            .map_err(|error| io_error(&temp_path, error))?;
        fs::rename(&temp_path, path).map_err(|error| io_error(path, error))
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    let _ = fs::remove_file(&index_path);
    let _ = fs::remove_file(&payload_path);
    result
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
