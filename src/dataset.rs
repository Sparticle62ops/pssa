use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{NormalizerWrapper, PostProcessorWrapper};
use tokenizers::{AddedToken, Tokenizer as HfTokenizer, TokenizerBuilder};

/// Zero-copy sequence chunking iterator for TBPTT training batches.
pub struct TokenChunkIterator<'a> {
    tokens: &'a [usize],
    chunk_len: usize,
    cursor: usize,
}
impl<'a> TokenChunkIterator<'a> {
    pub fn new(tokens: &'a [usize], chunk_len: usize) -> Self {
        Self {
            tokens,
            chunk_len: chunk_len.max(1),
            cursor: 0,
        }
    }
}
impl<'a> Iterator for TokenChunkIterator<'a> {
    type Item = (&'a [usize], &'a [usize]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor + 1 >= self.tokens.len() {
            return None;
        }
        let len = self.chunk_len.min(self.tokens.len() - 1 - self.cursor);
        let start = self.cursor;
        self.cursor += len;
        Some((
            &self.tokens[start..start + len],
            &self.tokens[start + 1..start + 1 + len],
        ))
    }
}

/// The checked tokenization policy persisted with V7 models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    Word,
    Bpe,
}

enum TokenizerBackend {
    Word,
    Bpe(HfTokenizer),
}

/// Either the legacy lowercase word tokenizer or a maintained, standard
/// Hugging Face `tokenizers` byte-level BPE tokenizer. ID zero is `<unk>`.
/// BPE has all 256 bytes in its alphabet and therefore never emits it for
/// ordinary UTF-8 input.
pub struct Tokenizer {
    pub vocab_size: usize,
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub synsets: Vec<HashSet<String>>,
    pub token_counts: Vec<usize>,
    pub unigram_table: Vec<usize>,
    kind: TokenizerKind,
    backend: TokenizerBackend,
    /// Decoded byte representation of every BPE vocabulary entry. This is
    /// deliberately cached: byte-level vocabulary labels are not display text.
    token_bytes: Vec<Vec<u8>>,
}

impl Tokenizer {
    const MAX_BPE_VOCAB: usize = 65_536;

    fn clean_and_tokenize(text: &str, lower: bool) -> Vec<String> {
        let mut tokens = Vec::new();
        for raw in text.split_whitespace() {
            let processed = if lower {
                raw.to_lowercase()
            } else {
                raw.to_string()
            };
            let mut word = String::new();
            for c in processed.chars() {
                if c.is_alphanumeric() || c == '-' || c == '\'' {
                    word.push(c);
                } else if matches!(c, '.' | ',' | '?' | '!') {
                    if !word.is_empty() {
                        tokens.push(std::mem::take(&mut word));
                    }
                    tokens.push(c.to_string());
                }
            }
            if !word.is_empty() {
                tokens.push(word);
            }
        }
        tokens
    }

    fn unigram_table(counts: &[usize]) -> Vec<usize> {
        let mut table = Vec::with_capacity(100_000);
        let total: f64 = counts.iter().map(|&n| (n.max(1) as f64).powf(0.75)).sum();
        let mut current = 0usize;
        let mut cumulative = (counts[0].max(1) as f64).powf(0.75) / total;
        for i in 0..100_000 {
            let p = i as f64 / 100_000.0;
            while p > cumulative && current + 1 < counts.len() {
                current += 1;
                cumulative += (counts[current].max(1) as f64).powf(0.75) / total;
            }
            table.push(current);
        }
        table
    }

    fn word_with_ordered_tokens(tokens: Vec<String>, counts: Vec<usize>) -> Result<Self, String> {
        if tokens.len() < 2 {
            return Err("vocabulary must contain <unk> and at least one token".into());
        }
        if tokens[0] != "<unk>" {
            return Err("vocabulary ID 0 must be <unk>".into());
        }
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            if token.trim().is_empty() {
                return Err(format!("vocabulary token {id} is empty"));
            }
            if token_to_id.insert(token.clone(), id).is_some() {
                return Err(format!("duplicate vocabulary token: {token}"));
            }
            id_to_token.insert(id, token.clone());
        }
        let mut synonyms = HashSet::new();
        synonyms.insert("fast".into());
        synonyms.insert("rapid".into());
        Ok(Self {
            vocab_size: tokens.len(),
            token_to_id,
            id_to_token,
            synsets: vec![synonyms],
            unigram_table: Self::unigram_table(&counts),
            token_counts: counts,
            kind: TokenizerKind::Word,
            backend: TokenizerBackend::Word,
            token_bytes: Vec::new(),
        })
    }

    /// Restores an ordered legacy checkpoint vocabulary exactly; it never relearns IDs.
    pub fn from_vocabulary(ordered: &[String]) -> Result<Self, String> {
        Self::word_with_ordered_tokens(ordered.to_vec(), vec![1; ordered.len()])
    }

    /// Returns checkpoint-safe ID order, rejecting a corrupted/gapped map.
    pub fn ordered_vocabulary(&self) -> Result<Vec<String>, String> {
        if self.vocab_size < 2 {
            return Err("vocabulary must contain at least two entries".into());
        }
        let mut ordered = Vec::with_capacity(self.vocab_size);
        for id in 0..self.vocab_size {
            let token = self
                .id_to_token
                .get(&id)
                .ok_or_else(|| format!("missing vocabulary ID {id}"))?;
            if self.token_to_id.get(token) != Some(&id) {
                return Err(format!("inconsistent vocabulary ID {id}"));
            }
            ordered.push(token.clone());
        }
        if ordered[0] != "<unk>" {
            return Err("vocabulary ID 0 must be <unk>".into());
        }
        Ok(ordered)
    }

    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// Stable identity for artifacts derived from this tokenizer. BPE caches
    /// include the serialized merge table, not only the visible vocabulary.
    pub(crate) fn cache_identity_hash(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        let mut feed = |bytes: &[u8]| {
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        };
        feed(match self.kind {
            TokenizerKind::Word => b"word",
            TokenizerKind::Bpe => b"bpe",
        });
        feed(&(self.vocab_size as u64).to_le_bytes());
        match &self.backend {
            TokenizerBackend::Word => {
                for id in 0..self.vocab_size {
                    if let Some(token) = self.id_to_token.get(&id) {
                        feed(token.as_bytes());
                    }
                    feed(&[0]);
                }
            }
            TokenizerBackend::Bpe(backend) => {
                if let Ok(json) = backend.to_string(false) {
                    feed(json.as_bytes());
                } else {
                    for id in 0..self.vocab_size {
                        if let Some(token) = self.id_to_token.get(&id) {
                            feed(token.as_bytes());
                        }
                        feed(&[0]);
                    }
                }
            }
        }
        hash
    }

    pub fn from_corpus(corpus: &str, lower: bool) -> Result<Self, String> {
        let raw = Self::clean_and_tokenize(corpus, lower);
        let mut freq = HashMap::<String, usize>::new();
        for token in &raw {
            *freq.entry(token.clone()).or_default() += 1;
        }
        let mut words: Vec<_> = freq.into_iter().collect();
        words.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let min_freq = if raw.len() < 1000 { 1 } else { 2 };
        let mut tokens = vec!["<unk>".into()];
        let mut counts = vec![1usize];
        let mut unknown = 0usize;
        for (word, count) in words {
            if tokens.len() >= 10_000 || (count < min_freq && word != "." && word != ",") {
                unknown += count;
            } else {
                tokens.push(word);
                counts.push(count);
            }
        }
        counts[0] = unknown.max(1);
        Self::word_with_ordered_tokens(tokens, counts)
    }

    /// Trains a deterministic byte-level BPE from precisely the supplied
    /// corpus. The complete ByteLevel alphabet covers arbitrary UTF-8 bytes.
    pub fn from_corpus_bpe(corpus: &str, vocab_size: usize) -> Result<Self, String> {
        if !(257..=Self::MAX_BPE_VOCAB).contains(&vocab_size) {
            return Err(format!(
                "BPE vocabulary size must be in 257..={}",
                Self::MAX_BPE_VOCAB
            ));
        }
        if corpus.is_empty() {
            return Err("cannot train BPE on an empty corpus".into());
        }
        let byte_level = ByteLevel::new(false, false, true);
        let mut backend = TokenizerBuilder::<
            BPE,
            NormalizerWrapper,
            ByteLevel,
            PostProcessorWrapper,
            ByteLevel,
        >::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(byte_level))
        .with_decoder(Some(byte_level))
        .build()
        .map_err(|e| format!("BPE tokenizer setup failed: {e}"))?;
        let mut trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
            .build();
        // lines retain document order and eliminate train-time implicit corpus lookup.
        backend
            .train(&mut trainer, corpus.split_inclusive('\n'))
            .map_err(|e| format!("BPE training failed: {e}"))?;
        Self::from_backend(backend.into())
    }

    fn from_backend(backend: HfTokenizer) -> Result<Self, String> {
        let vocab_size = backend.get_vocab_size(true);
        if vocab_size < 257 {
            return Err("byte-level BPE vocabulary omitted part of the byte alphabet".into());
        }
        let mut ordered = vec![String::new(); vocab_size];
        for (token, id) in backend.get_vocab(true) {
            let index = usize::try_from(id).map_err(|_| "token ID does not fit usize")?;
            if index >= ordered.len() || !ordered[index].is_empty() {
                return Err("BPE vocabulary IDs are not contiguous and unique".into());
            }
            ordered[index] = token;
        }
        if ordered.iter().any(String::is_empty) || ordered[0] != "<unk>" {
            return Err("BPE vocabulary must have contiguous IDs and <unk> at ID 0".into());
        }
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        let mut id_to_token = HashMap::with_capacity(vocab_size);
        for (id, token) in ordered.iter().enumerate() {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }
        let token_bytes = ordered
            .iter()
            .map(|token| {
                if token == "<unk>" {
                    Ok(Vec::new())
                } else {
                    byte_level_token_bytes(token)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Verify the declared base alphabet is complete, not merely that a BPE happened to train.
        let mut base = [false; 256];
        for bytes in &token_bytes {
            if bytes.len() == 1 {
                base[bytes[0] as usize] = true;
            }
        }
        if base.iter().any(|present| !present) {
            return Err("BPE tokenizer lacks complete 256-byte fallback alphabet".into());
        }
        Ok(Self {
            vocab_size,
            token_to_id,
            id_to_token,
            synsets: Vec::new(),
            token_counts: vec![1; vocab_size],
            unigram_table: Self::unigram_table(&vec![1; vocab_size]),
            kind: TokenizerKind::Bpe,
            backend: TokenizerBackend::Bpe(backend),
            token_bytes,
        })
    }

    /// Restores and validates a serialized maintained BPE tokenizer.
    pub fn from_serialized(json: &str) -> Result<Self, String> {
        if json.len() > 16 * 1024 * 1024 {
            return Err("tokenizer JSON exceeds 16 MiB cap".into());
        }
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("invalid tokenizer JSON: {e}"))?;
        let byte_level = value.get("pre_tokenizer").and_then(|x| x.as_object());
        if value
            .get("model")
            .and_then(|x| x.get("type"))
            .and_then(serde_json::Value::as_str)
            != Some("BPE")
            || byte_level
                .and_then(|x| x.get("type"))
                .and_then(serde_json::Value::as_str)
                != Some("ByteLevel")
            || byte_level
                .and_then(|x| x.get("add_prefix_space"))
                .and_then(serde_json::Value::as_bool)
                != Some(false)
            || value
                .get("decoder")
                .and_then(|x| x.get("type"))
                .and_then(serde_json::Value::as_str)
                != Some("ByteLevel")
            || !value
                .get("normalizer")
                .is_none_or(serde_json::Value::is_null)
        {
            return Err(
                "serialized tokenizer is not the required unnormalized ByteLevel BPE policy".into(),
            );
        }
        let backend = HfTokenizer::from_bytes(json.as_bytes())
            .map_err(|e| format!("invalid serialized BPE tokenizer: {e}"))?;
        let out = Self::from_backend(backend)?;
        if out.kind != TokenizerKind::Bpe {
            return Err("serialized tokenizer is not byte-level BPE".into());
        }
        Ok(out)
    }

    pub fn serialized_metadata(&self) -> Option<String> {
        match &self.backend {
            TokenizerBackend::Word => None,
            TokenizerBackend::Bpe(backend) => backend.to_string(false).ok(),
        }
    }

    /// Returns distinct word-level tokens that map to `<unk>`, in prompt order.
    /// BPE tokenization has byte fallback and therefore has no unknown words.
    pub fn unknown_words(&self, text: &str, lower: bool) -> Vec<String> {
        if self.kind != TokenizerKind::Word {
            return Vec::new();
        }
        let mut unknown = Vec::new();
        for token in Self::clean_and_tokenize(text, lower) {
            if self.token_to_id.contains_key(&token)
                || unknown.iter().any(|known: &String| known == &token)
            {
                continue;
            }
            unknown.push(token);
            if unknown.len() == 8 {
                break;
            }
        }
        unknown
    }

    pub fn try_encode(&self, text: &str, lower: bool) -> Result<Vec<usize>, String> {
        match &self.backend {
            TokenizerBackend::Word => Ok(Self::clean_and_tokenize(text, lower)
                .into_iter()
                .map(|x| self.token_to_id.get(&x).copied().unwrap_or(0))
                .collect()),
            TokenizerBackend::Bpe(backend) => {
                let encoding = backend
                    .encode(text, false)
                    .map_err(|e| format!("BPE encode failed: {e}"))?;
                let ids = encoding
                    .get_ids()
                    .iter()
                    .map(|&id| usize::try_from(id).map_err(|_| "token ID does not fit usize"))
                    .collect::<Result<Vec<_>, _>>()?;
                if ids.iter().any(|&id| id == 0 || id >= self.vocab_size) {
                    return Err(
                        "BPE emitted <unk> or an out-of-range token despite byte fallback".into(),
                    );
                }
                Ok(ids)
            }
        }
    }

    /// Compatibility wrapper; new runtime paths use `try_encode` to surface errors.
    pub fn encode(&self, text: &str, lower: bool) -> Vec<usize> {
        self.try_encode(text, lower).unwrap_or_default()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match &self.backend {
            TokenizerBackend::Word => {
                let mut out = String::new();
                for &id in ids {
                    let Some(token) = self.id_to_token.get(&id) else {
                        continue;
                    };
                    if matches!(token.as_str(), "." | "," | "?" | "!") {
                        out.push_str(token);
                    } else {
                        if !out.is_empty() {
                            out.push(' ');
                        }
                        out.push_str(token);
                    }
                }
                out
            }
            TokenizerBackend::Bpe(backend) => {
                let ids = ids
                    .iter()
                    .filter_map(|&id| u32::try_from(id).ok())
                    .collect::<Vec<_>>();
                backend.decode(&ids, true).unwrap_or_default()
            }
        }
    }

    pub fn token_bytes(&self, id: usize) -> Option<&[u8]> {
        if self.kind == TokenizerKind::Bpe {
            self.token_bytes.get(id).map(Vec::as_slice)
        } else {
            None
        }
    }

    pub fn are_synonyms(&self, a: usize, b: usize) -> bool {
        self.id_to_token
            .get(&a)
            .zip(self.id_to_token.get(&b))
            .is_some_and(|(x, y)| self.synsets.iter().any(|s| s.contains(x) && s.contains(y)))
    }
}

/// Inverse GPT-2/ByteLevel Unicode alphabet. This is used only to cache each
/// individual vocabulary token's raw bytes for allocation-free streaming.
fn byte_level_token_bytes(token: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(token.len());
    for ch in token.chars() {
        let code = ch as u32;
        let byte = if (33..=126).contains(&code)
            || (161..=172).contains(&code)
            || (174..=255).contains(&code)
        {
            code as u8
        } else if (256..=323).contains(&code) {
            // GPT-2 maps the missing bytes, in byte order, onto U+0100 onward.
            let target = (code - 256) as usize;
            (0u16..=255)
                .filter(|b| {
                    !((33..=126).contains(b) || (161..=172).contains(b) || (174..=255).contains(b))
                })
                .nth(target)
                .ok_or_else(|| format!("invalid ByteLevel character U+{code:04X}"))?
                as u8
        } else {
            return Err(format!(
                "token contains non-ByteLevel character U+{code:04X}"
            ));
        };
        out.push(byte);
    }
    Ok(out)
}

/// Clean a UTF-8 WikiText raw dump without buffering the whole corpus.
///
/// Removes balanced `= Title =` / `= = Section = =` heading lines and `<unk>`,
/// joins `@-@`, `@.@`, and `@,@` to their neighbors, and removes whitespace
/// before `.`, `,`, `)` and after `(`. Other inline whitespace becomes one
/// space, with no leading/trailing spaces. Consecutive blank lines (including
/// whitespace-only or `<unk>`-only lines) become one blank line; heading lines
/// do not introduce blanks. Leading/trailing blank runs also retain one blank.
///
/// Output uses LF line endings, including on a final unterminated input line.
/// Memory use is proportional to the longest line, not the file size. Read,
/// UTF-8, and write errors are returned; callers must flush buffered writers.
/// Dataset loading and tokenization do not implicitly apply this opt-in pass.
pub fn clean_wikitext<R: BufRead, W: Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    let mut line = String::new();
    let mut cleaned = String::new();
    let mut previous_blank = false;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let text = line.trim();
        if is_wikitext_heading(text) {
            continue;
        }
        clean_wikitext_line(text, &mut cleaned);
        let blank = cleaned.is_empty();
        if !blank || !previous_blank {
            writer.write_all(cleaned.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        previous_blank = blank;
    }
}

fn is_wikitext_heading(line: &str) -> bool {
    let marker = |ch: char| ch == '=' || ch.is_whitespace();
    let body = line.trim_start_matches(marker);
    let opening = line[..line.len() - body.len()]
        .bytes()
        .filter(|&b| b == b'=')
        .count();
    if opening == 0 {
        return false;
    }
    let title = body.trim_end_matches(marker);
    let closing = body[title.len()..].bytes().filter(|&b| b == b'=').count();
    !title.is_empty() && opening == closing
}

fn clean_wikitext_line(mut text: &str, out: &mut String) {
    out.clear();
    let mut pending_space = false;
    let mut join_next = false;
    while let Some(ch) = text.chars().next() {
        if let Some(rest) = text.strip_prefix("<unk>") {
            text = rest;
            continue;
        }
        let joined = match text.as_bytes() {
            [b'@', b'-', b'@', ..] => Some('-'),
            [b'@', b'.', b'@', ..] => Some('.'),
            [b'@', b',', b'@', ..] => Some(','),
            _ => None,
        };
        if let Some(punctuation) = joined {
            out.push(punctuation);
            text = &text[3..];
            pending_space = false;
            join_next = true;
            continue;
        }
        text = &text[ch.len_utf8()..];
        if ch.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space
            && !join_next
            && !out.is_empty()
            && !out.ends_with('(')
            && !matches!(ch, '.' | ',' | ')')
        {
            out.push(' ');
        }
        out.push(ch);
        pending_space = false;
        join_next = false;
    }
}

fn encode_query_component(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn field_value<'a>(value: &'a serde_json::Value, field: &str) -> Option<&'a serde_json::Value> {
    if let Some(value) = value.get(field) {
        return Some(value);
    }
    field
        .split('.')
        .try_fold(value, |current, part| current.get(part))
}

// Keep each HF row on one cache line, including null/blank rows. Arrays of
// strings belong to the same row, rather than becoming additional feed items.
fn append_text_value(value: &serde_json::Value, output: &mut String) -> Result<(), String> {
    match value {
        serde_json::Value::String(text) => {
            for part in text
                .split(['\r', '\n'])
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                if !output.is_empty() {
                    output.push(' ');
                }
                output.push_str(part);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                append_text_value(value, output)?;
            }
        }
        serde_json::Value::Null => {}
        _ => return Err("expected text, an array of text, or null".into()),
    }
    Ok(())
}

static HF_CACHE_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A streamed Hugging Face dataset request. The rows API is paged into a
/// local text cache, so a training run can be resumed without downloading the
/// same corpus again. This type intentionally contains no model state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HuggingFaceDatasetOptions {
    pub dataset: String,
    pub config: Option<String>,
    pub split: String,
    pub field: String,
}

impl HuggingFaceDatasetOptions {
    pub fn new(dataset: impl Into<String>) -> Self {
        Self {
            dataset: dataset.into(),
            config: None,
            split: "train".into(),
            field: "text".into(),
        }
    }
}

pub struct DatasetManager;
impl DatasetManager {
    pub const SCIENCE_REFERENCE_CORPUS: &'static str = "the solar system consists of the sun and the planetary objects orbiting it .\nthe four inner terrestrial planets are mercury , venus , earth , and mars , composed primarily of rock and metal .\nquantum mechanics is the branch of physics studying the behavior of matter and light at atomic scale .\ncomputer science is the study of computation , information , and the theoretical foundations of computation .\nartificial intelligence focuses on building computational models and software that learn .\nphotosynthesis is the biological process used by plants to convert light into energy .\ndna contains the genetic instructions necessary for the development and reproduction of living organisms .\nsocial science is the study of human societies and interconnected relationships .\ndata science involves analyzing large volumes of information to extract patterns .\n";

    pub fn try_load_dataset(sources_arg: Option<&str>) -> Result<String, String> {
        let raw = match sources_arg {
            None | Some("") => return Ok(Self::SCIENCE_REFERENCE_CORPUS.into()),
            Some(x) => x,
        };
        let sources: Vec<_> = raw
            .split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .collect();
        if sources.is_empty() {
            return Err("dataset source list is empty".into());
        }
        let mut combined = String::new();
        for source in sources {
            let text = if source == "science" {
                Self::SCIENCE_REFERENCE_CORPUS.into()
            } else if let Some(repo) = source.strip_prefix("hf:") {
                if repo.is_empty() {
                    return Err("hf: requires a repository".into());
                }
                Self::download_huggingface_dataset(repo)?
            } else if source.starts_with("http://") || source.starts_with("https://") {
                Self::download_url_dataset(source)?
            } else {
                let path = source
                    .strip_prefix("file:")
                    .or_else(|| source.strip_prefix("dir:"))
                    .or_else(|| source.strip_prefix("local:"))
                    .unwrap_or(source);
                Self::read_local(Path::new(path))?
            };
            if text.trim().is_empty() {
                return Err(format!("dataset source '{source}' is empty"));
            }
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&text);
            if !combined.ends_with('\n') {
                combined.push('\n');
            }
        }
        if combined.trim().is_empty() {
            Err("dataset is empty".into())
        } else {
            Ok(combined)
        }
    }
    fn read_local(path: &Path) -> Result<String, String> {
        if !path.exists() {
            return Err(format!(
                "local dataset path does not exist: {}",
                path.display()
            ));
        }
        if path.is_file() {
            return fs::read_to_string(path)
                .map_err(|e| format!("cannot read UTF-8 dataset {}: {e}", path.display()));
        }
        if !path.is_dir() {
            return Err(format!(
                "dataset path is not a regular file or directory: {}",
                path.display()
            ));
        }
        let mut paths = Vec::new();
        for entry in fs::read_dir(path)
            .map_err(|e| format!("cannot read dataset directory {}: {e}", path.display()))?
        {
            let entry = entry.map_err(|e| format!("cannot read dataset directory entry: {e}"))?;
            let ty = entry
                .file_type()
                .map_err(|e| format!("cannot inspect {}: {e}", entry.path().display()))?;
            if ty.is_file() {
                paths.push(entry.path());
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err(format!(
                "dataset directory contains no regular files: {}",
                path.display()
            ));
        }
        let mut out = String::new();
        for file in paths {
            let text = fs::read_to_string(&file)
                .map_err(|e| format!("cannot read UTF-8 dataset {}: {e}", file.display()))?;
            if text.trim().is_empty() {
                return Err(format!("dataset file is empty: {}", file.display()));
            }
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&text);
            if !out.ends_with('\n') {
                out.push('\n');
            }
        }
        Ok(out)
    }
    /// Fallible dataset loader.  Callers must not silently train on the
    /// built-in corpus when an explicitly requested source fails.
    pub fn load_dataset(sources_arg: Option<&str>) -> Result<String, String> {
        Self::try_load_dataset(sources_arg)
    }

    /// Compatibility helper for callers that explicitly want the historical
    /// built-in fallback.  New code should prefer [`Self::load_dataset`].
    pub fn load_dataset_or_default(sources_arg: Option<&str>) -> String {
        Self::try_load_dataset(sources_arg)
            .unwrap_or_else(|_| Self::SCIENCE_REFERENCE_CORPUS.into())
    }
    fn download_url_raw(url: &str) -> Result<String, String> {
        ureq::get(url)
            .set("User-Agent", concat!("pssa/", env!("CARGO_PKG_VERSION")))
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .map_err(|e| format!("HTTP request failed: {e}"))?
            .into_string()
            .map_err(|e| format!("failed to read response body: {e}"))
    }

    pub fn download_url_dataset(url: &str) -> Result<String, String> {
        let body = Self::download_url_raw(url)?;
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
            return Self::extract_json_text(&body).ok_or_else(|| {
                let kind = value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .map(|message| format!(": {message}"))
                    .unwrap_or_default();
                format!("URL response contains no supported text fields{kind}")
            });
        }
        Ok(body.trim().to_string())
    }
    /// Validate the explicit `owner/name` spelling before making a network
    /// request. This turns common typos into an immediate actionable error,
    /// including on Kaggle where a failed remote request is otherwise noisy.
    pub fn validate_huggingface_dataset_name(repo: &str) -> Result<(), String> {
        let mut parts = repo.split('/');
        let owner = parts.next().unwrap_or_default();
        let name = parts.next().unwrap_or_default();
        if owner.is_empty()
            || name.is_empty()
            || parts.next().is_some()
            || !owner
                .bytes()
                .chain(name.bytes())
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
            || owner == "."
            || name == "."
            || owner == ".."
            || name == ".."
        {
            return Err(format!(
                "invalid Hugging Face dataset '{repo}'; expected an owner/name repository"
            ));
        }
        Ok(())
    }

    /// Download a Hugging Face rows API request a page at a time and cache the
    /// newline-delimited selected field. The API is used instead of a heavy
    /// parquet dependency, which keeps the default Kaggle binary small.
    pub fn load_huggingface_dataset(options: &HuggingFaceDatasetOptions) -> Result<String, String> {
        Self::load_huggingface_cached(
            options,
            &Self::huggingface_cache_path(options),
            &mut Self::request_huggingface_json,
        )
    }

    // Inject only the HTTP boundary and cache location for offline fixtures.
    fn load_huggingface_cached(
        options: &HuggingFaceDatasetOptions,
        cache_path: &Path,
        fetch: &mut impl FnMut(&str) -> Result<serde_json::Value, String>,
    ) -> Result<String, String> {
        Self::validate_huggingface_dataset_name(&options.dataset)?;
        if options.split.trim().is_empty() {
            return Err("Hugging Face split must not be empty".into());
        }
        if options.field.trim().is_empty() {
            return Err("Hugging Face field must not be empty".into());
        }
        if options
            .config
            .as_deref()
            .is_some_and(|config| config.trim().is_empty())
        {
            return Err("Hugging Face config must not be empty".into());
        }

        match fs::read_to_string(cache_path) {
            Ok(cached) if !cached.trim().is_empty() => return Ok(cached),
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "cannot read Hugging Face cache {}: {error}",
                    cache_path.display()
                ));
            }
        }

        let mut resolved = options.clone();
        if resolved.config.is_none() {
            resolved.config = Some(Self::discover_huggingface_config(options, fetch)?);
        }
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "cannot create Hugging Face cache directory {}: {e}",
                    parent.display()
                )
            })?;
        }
        let sequence = HF_CACHE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temporary = cache_path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| {
                format!(
                    "cannot create Hugging Face cache {}: {e}",
                    temporary.display()
                )
            })?;
        let result = (|| {
            let mut output = io::BufWriter::new(file);
            Self::fetch_huggingface_rows(&resolved, &mut output, fetch)?;
            output
                .flush()
                .and_then(|()| output.get_ref().sync_all())
                .map_err(|e| {
                    format!(
                        "cannot flush Hugging Face cache {}: {e}",
                        temporary.display()
                    )
                })?;
            drop(output);
            fs::rename(&temporary, cache_path).map_err(|e| {
                format!(
                    "cannot commit Hugging Face cache {}: {e}",
                    cache_path.display()
                )
            })?;
            // Preserve the existing corpus String API, but only allocate the full
            // corpus after the complete download is safely on disk.
            fs::read_to_string(cache_path).map_err(|e| {
                format!(
                    "cannot read Hugging Face cache {}: {e}",
                    cache_path.display()
                )
            })
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn huggingface_cache_path(options: &HuggingFaceDatasetOptions) -> PathBuf {
        let root = crate::env_var_os("PSSA_HF_CACHE", "OXIDE_PSSA_HF_CACHE")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("XDG_CACHE_HOME")
                    .map(|p| PathBuf::from(p).join("pssa/huggingface"))
            })
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|p| PathBuf::from(p).join(".cache/pssa/huggingface"))
            })
            .unwrap_or_else(|| std::env::temp_dir().join("pssa/huggingface"));
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Invalidate older caches that split one source row across several lines.
        "hf-row-lines-v2".hash(&mut hasher);
        options.dataset.hash(&mut hasher);
        options.config.hash(&mut hasher);
        options.split.hash(&mut hasher);
        options.field.hash(&mut hasher);
        root.join(format!("{:016x}.txt", hasher.finish()))
    }

    fn request_huggingface_json(endpoint: &str) -> Result<serde_json::Value, String> {
        crate::tui::hf::dataset_json(endpoint)
    }

    fn huggingface_response(
        options: &HuggingFaceDatasetOptions,
        endpoint: &str,
        fetch: &mut impl FnMut(&str) -> Result<serde_json::Value, String>,
    ) -> Result<serde_json::Value, String> {
        let context = |error: String| {
            format!(
                "cannot load Hugging Face dataset '{}' (config '{}', split '{}'): {error}; check the repository name, access permissions, --hf-config and --hf-split",
                options.dataset,
                options.config.as_deref().unwrap_or("auto"),
                options.split
            )
        };
        let value = fetch(endpoint).map_err(&context)?;
        if let Some(error) = value.get("error") {
            return Err(context(
                error
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| error.to_string()),
            ));
        }
        Ok(value)
    }

    fn discover_huggingface_config(
        options: &HuggingFaceDatasetOptions,
        fetch: &mut impl FnMut(&str) -> Result<serde_json::Value, String>,
    ) -> Result<String, String> {
        let endpoint = format!(
            "https://datasets-server.huggingface.co/splits?dataset={}",
            encode_query_component(&options.dataset)
        );
        let value = Self::huggingface_response(options, &endpoint, fetch)?;
        let splits = value.get("splits").and_then(serde_json::Value::as_array)
            .ok_or_else(|| format!("Hugging Face dataset '{}' returned no splits; specify --hf-config explicitly if discovery is unavailable", options.dataset))?;
        let mut configs = Vec::new();
        let mut available_splits = Vec::new();
        for entry in splits {
            let (Some(config), Some(split)) = (
                entry.get("config").and_then(serde_json::Value::as_str),
                entry.get("split").and_then(serde_json::Value::as_str),
            ) else {
                continue;
            };
            available_splits.push(split);
            if split == options.split {
                configs.push(config);
            }
        }
        configs.sort_unstable();
        configs.dedup();
        available_splits.sort_unstable();
        available_splits.dedup();
        if configs.contains(&"default") {
            return Ok("default".into());
        }
        match configs.as_slice() {
            [config] => Ok((*config).into()),
            [] => Err(format!(
                "Hugging Face dataset '{}' has no available split '{}'; available splits: {}; check --hf-split or specify --hf-config if the dataset is still processing",
                options.dataset,
                options.split,
                available_splits.join(", ")
            )),
            _ => Err(format!(
                "Hugging Face dataset '{}' has multiple configs for split '{}': {}; select one with --hf-config",
                options.dataset,
                options.split,
                configs.join(", ")
            )),
        }
    }

    fn fetch_huggingface_rows(
        options: &HuggingFaceDatasetOptions,
        output: &mut impl Write,
        fetch: &mut impl FnMut(&str) -> Result<serde_json::Value, String>,
    ) -> Result<(), String> {
        const PAGE_SIZE: u64 = 100; // datasets-server /rows maximum
        let mut offset = 0u64;
        let mut total_rows = None;
        let mut has_text = false;
        let config = options
            .config
            .as_deref()
            .ok_or("Hugging Face config was not resolved")?;
        loop {
            let endpoint = format!(
                "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={offset}&length={PAGE_SIZE}",
                encode_query_component(&options.dataset),
                encode_query_component(config),
                encode_query_component(&options.split)
            );
            let value = Self::huggingface_response(options, &endpoint, fetch)?;
            let rows = value
                .get("rows")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    format!(
                        "Hugging Face dataset '{}' returned invalid rows JSON: missing rows array",
                        options.dataset
                    )
                })?;
            if let Some(total) = value
                .get("num_rows_total")
                .and_then(serde_json::Value::as_u64)
            {
                if total_rows.is_some_and(|previous| previous != total) {
                    return Err("Hugging Face row count changed during download; retry to avoid an inconsistent cache".into());
                }
                total_rows = Some(total);
            }
            if rows.is_empty() {
                if total_rows.is_some_and(|total| offset < total) {
                    return Err(format!(
                        "Hugging Face dataset '{}' ended early at row {offset} of {}; incomplete download was not cached",
                        options.dataset,
                        total_rows.unwrap()
                    ));
                }
                break;
            }
            if rows.len() as u64 > PAGE_SIZE
                || total_rows.is_some_and(|total| offset + rows.len() as u64 > total)
            {
                return Err(
                    "Hugging Face rows response exceeds the requested page or advertised total"
                        .into(),
                );
            }
            for (index, row) in rows.iter().enumerate() {
                let row_index = offset + index as u64;
                if row
                    .get("row_idx")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|idx| idx != row_index)
                {
                    return Err(format!(
                        "Hugging Face dataset '{}' returned an unexpected row index at {row_index}; incomplete download was not cached",
                        options.dataset
                    ));
                }
                let record = row.get("row").unwrap_or(row);
                let field = field_value(record, &options.field).ok_or_else(|| {
                    let available = record.as_object().map(|object| object.keys().cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default();
                    format!("Hugging Face dataset '{}' field '{}' is missing at row {row_index}; available fields: {available}; select a text column with --hf-field", options.dataset, options.field)
                })?;
                if row
                    .get("truncated_cells")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|cells| {
                        cells.iter().any(|cell| {
                            cell.as_str().is_some_and(|name| {
                                name == options.field
                                    || options.field.starts_with(&format!("{name}."))
                            })
                        })
                    })
                {
                    return Err(format!(
                        "Hugging Face field '{}' at row {row_index} is truncated by the rows API; use a local dataset export instead",
                        options.field
                    ));
                }
                let mut text = String::new();
                append_text_value(field, &mut text).map_err(|e| format!(
                    "Hugging Face dataset '{}' field '{}' at row {row_index}: {e}; select a text column with --hf-field",
                    options.dataset, options.field
                ))?;
                has_text |= !text.is_empty();
                writeln!(output, "{text}")
                    .map_err(|e| format!("cannot write Hugging Face cache: {e}"))?;
            }
            // Make each completed page visible on disk before fetching the next.
            output
                .flush()
                .map_err(|e| format!("cannot flush Hugging Face cache: {e}"))?;
            offset += rows.len() as u64;
            if total_rows.is_some_and(|total| offset == total)
                || (total_rows.is_none() && (rows.len() as u64) < PAGE_SIZE)
            {
                break;
            }
        }
        if !has_text {
            return Err(format!(
                "Hugging Face dataset '{}' split '{}' has no text values in field '{}'; check --hf-config, --hf-split and --hf-field",
                options.dataset, options.split, options.field
            ));
        }
        Ok(())
    }

    /// Preserve the legacy download/hf: source behavior (including automatic
    /// common-field extraction). Explicit --hf-* training uses the paged,
    /// selected-field cache above without changing existing script inputs.
    pub fn download_huggingface_dataset(repo: &str) -> Result<String, String> {
        let endpoint = format!(
            "https://datasets-server.huggingface.co/rows?dataset={repo}&split=train&offset=0&limit=1000"
        );
        let body = Self::request_huggingface_json(&endpoint)?.to_string();
        let text = Self::extract_json_text(&body).ok_or_else(|| {
            format!("dataset '{repo}' response contains no supported text fields")
        })?;
        if text.trim().is_empty() {
            Err(format!("dataset '{repo}' is empty"))
        } else {
            Ok(text)
        }
    }

    /// Extract text columns from structured API responses while leaving plain
    /// text downloads unchanged.  The Hugging Face rows API nests records as
    /// `rows[].row`, so extraction walks all objects and recognizes common
    /// corpus fields at any depth.
    fn extract_json_text(raw: &str) -> Option<String> {
        const TEXT_FIELDS: &[&str] = &[
            "text",
            "content",
            "article",
            "story",
            "instruction",
            "output",
            "sentence",
            "summary",
        ];
        let value: serde_json::Value = serde_json::from_str(raw).ok()?;
        let mut parts = Vec::new();
        fn collect_strings(value: &serde_json::Value, out: &mut Vec<String>) {
            match value {
                serde_json::Value::String(s) if !s.trim().is_empty() => out.push(s.clone()),
                serde_json::Value::Array(values) => {
                    for value in values {
                        collect_strings(value, out);
                    }
                }
                _ => {}
            }
        }
        fn collect_text(value: &serde_json::Value, fields: &[&str], out: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(object) => {
                    for (key, value) in object {
                        if fields.contains(&key.as_str()) {
                            collect_strings(value, out);
                        } else {
                            collect_text(value, fields, out);
                        }
                    }
                }
                serde_json::Value::Array(values) => {
                    for value in values {
                        collect_text(value, fields, out);
                    }
                }
                _ => {}
            }
        }
        collect_text(&value, TEXT_FIELDS, &mut parts);
        (!parts.is_empty()).then(|| parts.join("\n"))
    }

    pub fn extract_clean_text(raw: &str) -> String {
        Self::extract_json_text(raw).unwrap_or_else(|| raw.trim().to_string())
    }
    pub fn contradiction_stream() -> String {
        "the secret access code is 9988 . system authentication protocol initiated . the secret access code is 1122 . the secret access code is 1122 .".into()
    }
    pub fn mqar_stream(_gap: usize) -> String {
        "query manifold is orthogonal . state space models compute sequential representations . query gradient is convergent . continual learning architectures maintain plastic memory .".into()
    }
    pub fn spam_attack_stream(burst_count: usize) -> String {
        format!(
            "{}{}",
            "the speed of light is 300000 . ".repeat(6),
            "the speed of light is 500 . ".repeat(burst_count)
        )
    }
}

#[cfg(test)]
mod huggingface_tests {
    use super::*;
    use serde_json::{Value, json};

    struct CacheFixture(PathBuf);
    impl CacheFixture {
        fn new() -> Self {
            let sequence = HF_CACHE_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("pssa-hf-test-{}-{sequence}", std::process::id())),
            )
        }
        fn path(&self) -> PathBuf {
            self.0.join("corpus.txt")
        }
    }
    impl Drop for CacheFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn options() -> HuggingFaceDatasetOptions {
        let mut options = HuggingFaceDatasetOptions::new("owner/corpus");
        options.config = Some("default".into());
        options
    }

    fn page(offset: u64, count: u64, total: u64) -> Value {
        json!({
            "num_rows_total": total,
            "rows": (offset..offset + count).map(|idx| json!({
                "row_idx": idx, "row": {"text": format!("row {idx}")}
            })).collect::<Vec<_>>()
        })
    }

    #[test]
    fn pages_at_100_and_flushes_to_cache_before_next_request() {
        let cache = CacheFixture::new();
        let mut requests = 0;
        let corpus =
            DatasetManager::load_huggingface_cached(&options(), &cache.path(), &mut |url| {
                assert!(url.contains("dataset=owner%2Fcorpus&config=default&split=train"));
                assert!(url.ends_with("&length=100"));
                let offset = requests * 100;
                assert!(url.contains(&format!("&offset={offset}&")));
                if requests > 0 {
                    assert!(
                        !cache.path().exists(),
                        "partial cache must not be published"
                    );
                    let files = fs::read_dir(&cache.0)
                        .unwrap()
                        .collect::<Result<Vec<_>, _>>()
                        .unwrap();
                    assert_eq!(files.len(), 1);
                    assert_eq!(
                        fs::read_to_string(files[0].path()).unwrap().lines().count(),
                        offset as usize
                    );
                }
                requests += 1;
                Ok(page(offset, (205 - offset).min(100), 205))
            })
            .unwrap();
        assert_eq!(requests, 3);
        assert_eq!(corpus.lines().count(), 205);
        assert!(corpus.ends_with("row 204\n"));
        assert_eq!(fs::read_to_string(cache.path()).unwrap(), corpus);
        let cached =
            DatasetManager::load_huggingface_cached(&options(), &cache.path(), &mut |_| {
                panic!("cache hit must not perform discovery or fetch")
            })
            .unwrap();
        assert_eq!(cached, corpus);
    }

    #[test]
    fn short_pages_use_reported_total_and_unknown_total_ends_on_empty_page() {
        let mut out = Vec::new();
        let mut calls = 0;
        DatasetManager::fetch_huggingface_rows(&options(), &mut out, &mut |url| {
            assert!(url.contains(&format!("&offset={calls}&")));
            let result = page(calls, 1, 2);
            calls += 1;
            Ok(result)
        })
        .unwrap();
        assert_eq!(out, b"row 0\nrow 1\n");
        assert_eq!(calls, 2);

        calls = 0;
        out.clear();
        DatasetManager::fetch_huggingface_rows(&options(), &mut out, &mut |url| {
            let mut result = if calls == 0 {
                page(0, 100, 100)
            } else {
                assert!(url.contains("&offset=100&"));
                page(100, 0, 100)
            };
            result.as_object_mut().unwrap().remove("num_rows_total");
            calls += 1;
            Ok(result)
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(String::from_utf8(out).unwrap().lines().count(), 100);
    }

    #[test]
    fn discovers_default_or_unique_config_and_explains_ambiguity_and_missing_split() {
        let options = HuggingFaceDatasetOptions::new("owner/corpus");
        let discover = |splits: Value| {
            DatasetManager::discover_huggingface_config(&options, &mut |url| {
                assert!(url.ends_with("/splits?dataset=owner%2Fcorpus"));
                Ok(json!({"splits": splits}))
            })
        };
        assert_eq!(
            discover(json!([
                {"config": "other", "split": "train"},
                {"config": "default", "split": "train"}
            ]))
            .unwrap(),
            "default"
        );
        assert_eq!(
            discover(json!([
                {"config": "default", "split": "test"},
                {"config": "en v1", "split": "train"}
            ]))
            .unwrap(),
            "en v1"
        );
        let error = discover(json!([
            {"config": "en", "split": "train"},
            {"config": "fr", "split": "train"}
        ]))
        .unwrap_err();
        assert!(error.contains("en, fr") && error.contains("--hf-config"));
        let error = discover(json!([{"config": "en", "split": "test"}])).unwrap_err();
        assert!(
            error.contains("no available split 'train'")
                && error.contains("test")
                && error.contains("--hf-split")
        );
    }

    #[test]
    fn discovered_config_is_encoded_and_cached_without_further_discovery() {
        let cache = CacheFixture::new();
        let options = HuggingFaceDatasetOptions::new("owner/corpus");
        let mut calls = 0;
        let corpus = DatasetManager::load_huggingface_cached(&options, &cache.path(), &mut |url| {
            calls += 1;
            if url.contains("/splits?") {
                Ok(json!({"splits": [{"config": "en/v 1", "split": "train"}]}))
            } else {
                assert!(url.contains("&config=en%2Fv%201&"));
                Ok(page(0, 1, 1))
            }
        })
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(corpus, "row 0\n");
        assert_eq!(
            DatasetManager::load_huggingface_cached(&options, &cache.path(), &mut |_| panic!(
                "offline cache hit"
            ))
            .unwrap(),
            corpus
        );
    }

    #[test]
    fn nested_and_literal_fields_preserve_one_line_per_source_row() {
        let mut options = options();
        options.field = "document.text".into();
        let mut out = Vec::new();
        DatasetManager::fetch_huggingface_rows(&options, &mut out, &mut |_| {
            Ok(json!({
                "num_rows_total": 4,
                "rows": [
                    {"row": {"document": {"text": " first\r\nsecond\n第三 "}}},
                    {"row": {"document.text": ["a\nb", "c"], "document": {"text": "wrong"}}},
                    {"row": {"document": {"text": null}}},
                    {"row": {"document": {"text": "  "}}}
                ]
            }))
        })
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "first second 第三\na b c\n\n\n"
        );
    }

    #[test]
    fn field_and_response_errors_are_actionable() {
        for (response, expected) in [
            (
                json!({"rows": [{"row": {"content": "hello"}}]}),
                "available fields: content",
            ),
            (json!({"rows": [{"row": {"text": 42}}]}), "expected text"),
            (json!({"rows": [{"row": {"text": null}}]}), "no text values"),
            (json!({"rows": []}), "no text values"),
            (
                json!({"rows": [{"row": {"text": "short"}, "truncated_cells": ["text"]}]}),
                "truncated",
            ),
            (
                json!({"rows": [{"row_idx": 3, "row": {"text": "hello"}}]}),
                "unexpected row index",
            ),
            (json!({"error": "Dataset not found"}), "Dataset not found"),
            (json!({"unexpected": []}), "missing rows array"),
        ] {
            let error =
                DatasetManager::fetch_huggingface_rows(&options(), &mut Vec::new(), &mut |_| {
                    Ok(response.clone())
                })
                .unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
        let error =
            DatasetManager::fetch_huggingface_rows(&options(), &mut Vec::new(), &mut |_| {
                Err("HTTP 404: split not found".into())
            })
            .unwrap_err();
        for expected in ["owner/corpus", "default", "train", "HTTP 404", "--hf-split"] {
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn failed_downloads_remove_partial_cache_and_can_retry() {
        for fail_with_empty_page in [false, true] {
            let cache = CacheFixture::new();
            let mut calls = 0;
            let error =
                DatasetManager::load_huggingface_cached(&options(), &cache.path(), &mut |_| {
                    calls += 1;
                    match calls {
                        1 => Ok(page(0, 100, 101)),
                        _ if fail_with_empty_page => Ok(page(100, 0, 101)),
                        _ => Err("connection lost".into()),
                    }
                })
                .unwrap_err();
            assert!(error.contains(if fail_with_empty_page {
                "ended early"
            } else {
                "connection lost"
            }));
            assert!(!cache.path().exists());
            assert_eq!(fs::read_dir(&cache.0).unwrap().count(), 0);
            assert_eq!(
                DatasetManager::load_huggingface_cached(&options(), &cache.path(), &mut |_| Ok(
                    page(0, 1, 1)
                ))
                .unwrap(),
                "row 0\n"
            );
        }
    }

    #[test]
    fn invalid_options_fail_before_http_and_cache_keys_include_selection() {
        let cache = CacheFixture::new();
        let mut variants = Vec::new();
        for repo in [
            "",
            "corpus",
            "owner/",
            "../corpus",
            "owner/corpus/extra",
            "owner/bad?name",
        ] {
            variants.push(HuggingFaceDatasetOptions::new(repo));
        }
        let mut invalid = options();
        invalid.split.clear();
        variants.push(invalid);
        let mut invalid = options();
        invalid.field.clear();
        variants.push(invalid);
        let mut invalid = options();
        invalid.config = Some(" ".into());
        variants.push(invalid);
        for invalid in variants {
            assert!(
                DatasetManager::load_huggingface_cached(&invalid, &cache.path(), &mut |_| panic!(
                    "invalid input must not fetch"
                ))
                .is_err()
            );
        }
        let base = options();
        let mut variants = vec![base.clone(); 4];
        variants[0].dataset = "other/corpus".into();
        variants[1].config = None;
        variants[2].split = "test".into();
        variants[3].field = "content".into();
        let base_path = DatasetManager::huggingface_cache_path(&base);
        for variant in variants {
            assert_ne!(DatasetManager::huggingface_cache_path(&variant), base_path);
        }
        assert_eq!(encode_query_component("é +&"), "%C3%A9%20%2B%26");
    }
}

#[cfg(test)]
mod wikitext_tests {
    use super::clean_wikitext;
    use std::io::{self, BufReader, Read};

    fn clean(raw: &str) -> String {
        let mut out = Vec::new();
        clean_wikitext(raw.as_bytes(), &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    // Sentence excerpts and headings from Salesforce/wikitext,
    // wikitext-103-raw-v1 test rows 1, 3, 6, 9, 34, 54, and 119.
    #[test]
    fn restores_hyphenated_words() {
        assert_eq!(
            clean(
                " Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . \n"
            ),
            "Robert Boulter is an English film, television and theatre actor. He had a guest-starring role on the television series The Bill in 2000.\n"
        );
        assert_eq!(
            clean("guest\t@-@  starring co@-@stars\n"),
            "guest-starring co-stars\n"
        );
    }

    #[test]
    fn restores_decimal_points_and_thousands_separators() {
        assert_eq!(
            clean(
                "It caused enormous disruption to Chinese society : the census of 754 recorded 52 @.@ 9 million people , but ten years later , the census counted just 16 @.@ 9 million , the remainder having been displaced or killed ."
            ),
            "It caused enormous disruption to Chinese society : the census of 754 recorded 52.9 million people, but ten years later, the census counted just 16.9 million, the remainder having been displaced or killed.\n"
        );
        assert_eq!(
            clean(
                "The single peaked at number 46 on the US Billboard Hot 100 and has been certified gold by the Recording Industry Association of America ( RIAA ) for shipments of 500 @,@ 000 copies ."
            ),
            "The single peaked at number 46 on the US Billboard Hot 100 and has been certified gold by the Recording Industry Association of America (RIAA) for shipments of 500,000 copies.\n"
        );
        assert_eq!(clean("1\t@,@  234@.@5\n"), "1,234.5\n");
    }

    #[test]
    fn drops_heading_lines_without_inserting_blanks() {
        assert_eq!(
            clean(
                " = Robert Boulter = \n = = Career = = \n = = = 2000 – 2005 = = = \n== Career ==\nRobert Boulter .\n= Filmography =\nThe Bill .\n"
            ),
            "Robert Boulter.\nThe Bill.\n"
        );
        assert_eq!(clean(" = Title = \n"), "");
        assert_eq!(
            clean("An equation: x = y .\n= not a heading\n"),
            "An equation: x = y.\n= not a heading\n"
        );
    }

    #[test]
    fn collapses_blank_lines_even_across_removed_headings() {
        assert_eq!(
            clean(
                " \t\r\n\nRobert Boulter .\r\n \t\n= = Career = =\n\n\u{2003}\nThe Bill .\n\n \t\n"
            ),
            "\nRobert Boulter.\n\nThe Bill.\n\n"
        );
        assert_eq!(clean(""), "");
        assert_eq!(clean(" \t\n\r\n \n"), "\n");
    }

    #[test]
    fn restores_punctuation_spacing_without_corrupting_unicode() {
        assert_eq!(
            clean(
                " Du Fu ( Wade – Giles : Tu Fu ; Chinese : 杜甫 ; 712 – 770 ) was a prominent Chinese poet of the Tang dynasty .\n"
            ),
            "Du Fu (Wade – Giles : Tu Fu ; Chinese : 杜甫 ; 712 – 770) was a prominent Chinese poet of the Tang dynasty.\n"
        );
        assert_eq!(clean("( ( 杜甫 ) ) , poetry .\n"), "((杜甫)), poetry.\n");
    }

    #[test]
    fn removes_unknown_tokens_and_their_extra_spacing() {
        // Inject unknown-token artifacts into the Robert Boulter excerpt.
        assert_eq!(
            clean(
                " <unk> Robert Boulter is an English <unk> <unk> film , television and theatre actor <unk> . <unk> \n"
            ),
            "Robert Boulter is an English film, television and theatre actor.\n"
        );
        assert_eq!(
            clean("<unk>\n \n <unk> <unk> \nThe Bill ."),
            "\nThe Bill.\n"
        );
        assert_eq!(clean("( <unk> Du Fu <unk> )\n"), "(Du Fu)\n");
    }

    #[test]
    fn cleaning_is_idempotent_and_preserves_natural_spacing() {
        let raw = "= Robert Boulter =\n\nHe had a guest @-@ starring role <unk> .\n\n= = Career = =\n \nDu Fu ( 杜甫 ) .\n52 @.@ 9 million , 500 @,@ 000 copies .";
        let once = clean(raw);
        assert_eq!(clean(&once), once);
        let natural = "Du Fu (杜甫) was a poet, born in 712.\n52.9 million; 500,000 copies; guest-starring.\nx = y; a - b; mail@example.org\n";
        assert_eq!(clean(natural), natural);
    }

    #[test]
    fn handles_small_read_buffers_and_returns_utf8_and_write_errors() {
        let raw = "杜甫 ( Du Fu ) , guest @-@ starring .\r\n";
        let mut out = Vec::new();
        clean_wikitext(BufReader::with_capacity(1, raw.as_bytes()), &mut out).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), clean(raw));
        let err = clean_wikitext(&b"\xff\n"[..], io::sink()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let mut short = [0; 2];
        let err = clean_wikitext(raw.as_bytes(), &mut short[..]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WriteZero);
    }

    #[test]
    fn emits_completed_lines_before_reading_the_rest_of_the_corpus() {
        struct FailedRead;
        impl Read for FailedRead {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("read failed"))
            }
        }
        let input = io::Cursor::new(b"Robert Boulter .\n").chain(FailedRead);
        let mut out = Vec::new();
        let err = clean_wikitext(BufReader::new(input), &mut out).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert_eq!(out, b"Robert Boulter.\n");
    }
}
