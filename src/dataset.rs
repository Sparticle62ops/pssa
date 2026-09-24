use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

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

    pub fn from_corpus(corpus: &str, lower: bool) -> Self {
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
        Self::word_with_ordered_tokens(tokens, counts).expect("constructed vocabulary is valid")
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
    pub fn load_dataset(sources_arg: Option<&str>) -> String {
        Self::try_load_dataset(sources_arg)
            .unwrap_or_else(|_| Self::SCIENCE_REFERENCE_CORPUS.into())
    }
    pub fn download_url_dataset(url: &str) -> Result<String, String> {
        let body = ureq::get(url)
            .set("User-Agent", "oxide-ai/0.4.0")
            .timeout(std::time::Duration::from_secs(60))
            .call()
            .map_err(|e| format!("HTTP request failed: {e}"))?
            .into_string()
            .map_err(|e| format!("failed to read response body: {e}"))?;
        Ok(Self::extract_clean_text(&body))
    }
    pub fn download_huggingface_dataset(repo: &str) -> Result<String, String> {
        let endpoint = format!(
            "https://datasets-server.huggingface.co/rows?dataset={repo}&split=train&offset=0&limit=1000"
        );
        let text = Self::download_url_dataset(&endpoint)?;
        if text.trim().is_empty() {
            Err(format!("dataset '{repo}' is empty"))
        } else {
            Ok(text)
        }
    }
    pub fn extract_clean_text(raw: &str) -> String {
        raw.trim().to_string()
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
