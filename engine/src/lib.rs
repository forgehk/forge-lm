//! Inference for forge-lm models with nothing but the Rust standard library.
//!
//! The Python side trains the network in PyTorch and writes one `.flm` file. This
//! crate reads that file and runs the same arithmetic by hand: token embedding,
//! RMSNorm, rotary position embeddings, causal multi-head attention with a
//! key-value cache, a SwiGLU feed-forward layer, and a tied output projection.
//!
//! The integration tests feed a prompt through both implementations and require
//! the logits to agree at every position, so the two cannot drift apart.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 4] = b"FLM1";
const ROPE_BASE: f32 = 10000.0;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Format(String),
    ContextFull(usize),
    BadToken(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "cannot read the model: {err}"),
            Error::Format(why) => write!(f, "not a valid model file: {why}"),
            Error::ContextFull(size) => write!(f, "the context is full ({size} tokens)"),
            Error::BadToken(id) => write!(f, "token {id} is not in the vocabulary"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub hidden: usize,
    pub block_size: usize,
    pub eps: f32,
}

struct Layer {
    norm1: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    norm2: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

pub struct Model {
    pub config: Config,
    pub tokenizer: Tokenizer,
    tok: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    freqs: Vec<f32>,
}

/// Cursor over the bytes of a model file. Every read is bounds-checked.
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.data.len());
        match end {
            Some(end) => {
                let bytes = &self.data[self.at..end];
                self.at = end;
                Ok(bytes)
            }
            None => Err(Error::Format(format!("the file ends inside {what}"))),
        }
    }

    fn u32(&mut self, what: &str) -> Result<u32, Error> {
        let bytes = self.take(4, what)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn f32(&mut self, what: &str) -> Result<f32, Error> {
        Ok(f32::from_bits(self.u32(what)?))
    }

    fn tensor(&mut self, len: usize, what: &str) -> Result<Vec<f32>, Error> {
        let size = len
            .checked_mul(4)
            .ok_or_else(|| Error::Format(format!("{what} is impossibly large")))?;
        let bytes = self.take(size, what)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Model, Error> {
        Model::from_bytes(&fs::read(path)?)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Model, Error> {
        let mut r = Reader { data, at: 0 };
        if r.take(4, "the magic number")? != MAGIC {
            return Err(Error::Format("the magic number is not FLM1".into()));
        }
        let mut header = [0usize; 6];
        for field in header.iter_mut() {
            *field = r.u32("the header")? as usize;
        }
        let [vocab_size, dim, n_layers, n_heads, hidden, block_size] = header;
        let eps = r.f32("the header")?;
        if header.contains(&0) || vocab_size < 256 || dim % n_heads != 0 || (dim / n_heads) % 2 != 0
        {
            return Err(Error::Format(
                "the header describes an impossible network".into(),
            ));
        }
        if header.iter().any(|&v| v > 1 << 20) {
            return Err(Error::Format(
                "the header describes an implausibly large network".into(),
            ));
        }

        let n_merges = r.u32("the merge table")? as usize;
        if 256 + n_merges != vocab_size {
            return Err(Error::Format(
                "the merge table does not match the vocabulary size".into(),
            ));
        }
        let mut merges = Vec::with_capacity(n_merges);
        for index in 0..n_merges {
            let (left, right) = (r.u32("the merge table")?, r.u32("the merge table")?);
            // A merge may only combine tokens that already exist when it is learned.
            if left as usize >= 256 + index || right as usize >= 256 + index {
                return Err(Error::Format(format!(
                    "merge {index} refers to a token that does not exist yet"
                )));
            }
            merges.push((left, right));
        }

        let tok = r.tensor(vocab_size * dim, "the token embedding")?;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Layer {
                norm1: r.tensor(dim, "a layer")?,
                wq: r.tensor(dim * dim, "a layer")?,
                wk: r.tensor(dim * dim, "a layer")?,
                wv: r.tensor(dim * dim, "a layer")?,
                wo: r.tensor(dim * dim, "a layer")?,
                norm2: r.tensor(dim, "a layer")?,
                w1: r.tensor(hidden * dim, "a layer")?,
                w2: r.tensor(dim * hidden, "a layer")?,
                w3: r.tensor(hidden * dim, "a layer")?,
            });
        }
        let norm = r.tensor(dim, "the final norm")?;
        if r.at != data.len() {
            return Err(Error::Format(format!(
                "{} unexpected bytes after the last tensor",
                data.len() - r.at
            )));
        }

        let head_dim = dim / n_heads;
        let freqs = (0..head_dim / 2)
            .map(|i| ROPE_BASE.powf(-((2 * i) as f32) / head_dim as f32))
            .collect();
        Ok(Model {
            config: Config {
                vocab_size,
                dim,
                n_layers,
                n_heads,
                hidden,
                block_size,
                eps,
            },
            tokenizer: Tokenizer::new(merges),
            tok,
            layers,
            norm,
            freqs,
        })
    }

    pub fn n_params(&self) -> usize {
        let per_layer: usize = self.layers.first().map_or(0, |l| {
            l.norm1.len()
                + l.wq.len()
                + l.wk.len()
                + l.wv.len()
                + l.wo.len()
                + l.norm2.len()
                + l.w1.len()
                + l.w2.len()
                + l.w3.len()
        });
        self.tok.len() + per_layer * self.layers.len() + self.norm.len()
    }

    /// Start a fresh context with an empty key-value cache.
    pub fn session(&self) -> Session<'_> {
        let c = &self.config;
        Session {
            model: self,
            keys: vec![Vec::with_capacity(c.block_size * c.dim); c.n_layers],
            values: vec![Vec::with_capacity(c.block_size * c.dim); c.n_layers],
            pos: 0,
            logits: vec![0.0; c.vocab_size],
        }
    }
}

/// One running context. Feeding a token costs one pass over the cached keys and
/// values, not a recomputation of the whole prefix.
pub struct Session<'m> {
    model: &'m Model,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    pos: usize,
    logits: Vec<f32>,
}

impl Session<'_> {
    pub fn position(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.keys.iter_mut().for_each(Vec::clear);
        self.values.iter_mut().for_each(Vec::clear);
    }

    /// Append one token and return the logits for the token that follows it.
    pub fn feed(&mut self, token: u32) -> Result<&[f32], Error> {
        let m = self.model;
        let c = &m.config;
        if token as usize >= c.vocab_size {
            return Err(Error::BadToken(token));
        }
        if self.pos >= c.block_size {
            return Err(Error::ContextFull(c.block_size));
        }
        let (dim, heads, hidden) = (c.dim, c.n_heads, c.hidden);
        let head_dim = dim / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut x = m.tok[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut h = vec![0.0f32; dim];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; dim];
        let mut v = vec![0.0f32; dim];
        let mut attended = vec![0.0f32; dim];
        let mut delta = vec![0.0f32; dim];
        let mut gate = vec![0.0f32; hidden];
        let mut lin = vec![0.0f32; hidden];
        let mut scores = vec![0.0f32; self.pos + 1];

        for (index, layer) in m.layers.iter().enumerate() {
            rmsnorm(&mut h, &x, &layer.norm1, c.eps);
            matvec(&mut q, &layer.wq, &h);
            matvec(&mut k, &layer.wk, &h);
            matvec(&mut v, &layer.wv, &h);
            for head in 0..heads {
                let span = head * head_dim..(head + 1) * head_dim;
                rotate(&mut q[span.clone()], self.pos, &m.freqs);
                rotate(&mut k[span], self.pos, &m.freqs);
            }
            self.keys[index].extend_from_slice(&k);
            self.values[index].extend_from_slice(&v);
            let (keys, values) = (&self.keys[index], &self.values[index]);

            for head in 0..heads {
                let span = head * head_dim..(head + 1) * head_dim;
                let query = &q[span.clone()];
                for (t, score) in scores.iter_mut().enumerate() {
                    let key = &keys[t * dim + span.start..t * dim + span.end];
                    *score = dot(query, key) * scale;
                }
                softmax(&mut scores);
                let out = &mut attended[span.clone()];
                out.fill(0.0);
                for (t, &weight) in scores.iter().enumerate() {
                    let value = &values[t * dim + span.start..t * dim + span.end];
                    for (o, &val) in out.iter_mut().zip(value) {
                        *o += weight * val;
                    }
                }
            }
            matvec(&mut delta, &layer.wo, &attended);
            add(&mut x, &delta);

            rmsnorm(&mut h, &x, &layer.norm2, c.eps);
            matvec(&mut gate, &layer.w1, &h);
            matvec(&mut lin, &layer.w3, &h);
            for (g, &l) in gate.iter_mut().zip(&lin) {
                *g = silu(*g) * l;
            }
            matvec(&mut delta, &layer.w2, &gate);
            add(&mut x, &delta);
        }

        rmsnorm(&mut h, &x, &m.norm, c.eps);
        matvec(&mut self.logits, &m.tok, &h);
        self.pos += 1;
        Ok(&self.logits)
    }
}

/// `out = w * x`, with `w` stored row-major as `out.len()` rows of `x.len()` columns.
fn matvec(out: &mut [f32], w: &[f32], x: &[f32]) {
    debug_assert_eq!(w.len(), out.len() * x.len());
    for (o, row) in out.iter_mut().zip(w.chunks_exact(x.len())) {
        *o = dot(row, x);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn add(x: &mut [f32], delta: &[f32]) {
    for (a, b) in x.iter_mut().zip(delta) {
        *a += b;
    }
}

fn rmsnorm(out: &mut [f32], x: &[f32], gain: &[f32], eps: f32) {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_square + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(gain) {
        *o = v * inv * g;
    }
}

/// Rotary position embedding: turn each pair (2i, 2i+1) by `pos * freqs[i]`.
fn rotate(x: &mut [f32], pos: usize, freqs: &[f32]) {
    for (pair, &freq) in x.chunks_exact_mut(2).zip(freqs) {
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * cos - b * sin;
        pair[1] = a * sin + b * cos;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Byte-level BPE with the merge table stored in the model file.
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    rank: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn new(merges: Vec<(u32, u32)>) -> Tokenizer {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut rank = HashMap::with_capacity(merges.len());
        for (index, &(left, right)) in merges.iter().enumerate() {
            let mut bytes = vocab[left as usize].clone();
            bytes.extend_from_slice(&vocab[right as usize]);
            vocab.push(bytes);
            rank.insert((left, right), index as u32);
        }
        Tokenizer {
            merges,
            rank,
            vocab,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode bytes. Text is cut into chunks (whitespace run, then non-whitespace
    /// run) and merges are applied inside each chunk in the order they were learned.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let is_space = |b: u8| matches!(b, b' ' | b'\n' | b'\t' | b'\r');
        let mut ids = Vec::with_capacity(text.len());
        let mut at = 0;
        while at < text.len() {
            let start = at;
            while at < text.len() && is_space(text[at]) {
                at += 1;
            }
            while at < text.len() && !is_space(text[at]) {
                at += 1;
            }
            self.encode_chunk(&text[start..at], &mut ids);
        }
        ids
    }

    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<u32>) {
        let mut tokens: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        while tokens.len() > 1 {
            let best = tokens
                .windows(2)
                .filter_map(|pair| self.rank.get(&(pair[0], pair[1])))
                .min()
                .copied();
            let Some(rank) = best else { break };
            let (left, right) = self.merges[rank as usize];
            let mut merged = Vec::with_capacity(tokens.len());
            let mut read = 0;
            while read < tokens.len() {
                if read + 1 < tokens.len() && tokens[read] == left && tokens[read + 1] == right {
                    merged.push(256 + rank);
                    read += 2;
                } else {
                    merged.push(tokens[read]);
                    read += 1;
                }
            }
            tokens = merged;
        }
        out.extend_from_slice(&tokens);
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .flatten()
            .copied()
            .collect()
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.vocab.get(id as usize).map_or(&[], Vec::as_slice)
    }
}

/// xorshift64*: small, fast and reproducible from a seed. Not for cryptography.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }
}

/// Pick the next token. Temperature zero means greedy. `top_k` zero means no cut.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let argmax = || {
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    };
    if temperature <= 0.0 {
        return argmax();
    }
    let mut probs: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    if top_k > 0 && top_k < probs.len() {
        let mut sorted = probs.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = sorted[top_k - 1];
        for p in probs.iter_mut() {
            if *p < cutoff {
                *p = f32::NEG_INFINITY;
            }
        }
    }
    softmax(&mut probs);
    let mut target = rng.next_f32();
    for (i, &p) in probs.iter().enumerate() {
        target -= p;
        if target < 0.0 {
            return i as u32;
        }
    }
    argmax()
}

/// Generate `count` tokens after `prompt`, calling `emit` with each new token.
///
/// When the context fills up, the second half of it is kept and re-fed from
/// position zero, so generation can run past the block size.
pub fn generate(
    model: &Model,
    prompt: &[u32],
    count: usize,
    temperature: f32,
    top_k: usize,
    seed: u64,
    mut emit: impl FnMut(u32),
) -> Result<Vec<u32>, Error> {
    let block = model.config.block_size;
    let mut history: Vec<u32> = if prompt.is_empty() {
        vec![b'\n' as u32]
    } else {
        prompt.to_vec()
    };
    let mut session = model.session();
    let mut rng = Rng::new(seed);
    let mut made = Vec::with_capacity(count);

    let start = history.len().saturating_sub(block);
    let mut logits = Vec::new();
    for &token in &history[start..] {
        logits = session.feed(token)?.to_vec();
    }
    for _ in 0..count {
        let next = sample(&logits, temperature, top_k, &mut rng);
        emit(next);
        made.push(next);
        history.push(next);
        if session.position() == block {
            session.reset();
            let keep = history.len() - block / 2;
            for &token in &history[keep..history.len() - 1] {
                session.feed(token)?;
            }
        }
        logits = session.feed(next)?.to_vec();
    }
    Ok(made)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = [1000.0, 1001.0, 999.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[1] > x[0] && x[0] > x[2]);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rmsnorm_gives_unit_mean_square() {
        let x = [3.0, -4.0, 12.0, 0.5];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &[1.0; 4], 0.0);
        let mean_square = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((mean_square - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rotation_keeps_length_and_depends_only_on_relative_position() {
        let freqs = [1.0, 0.1];
        let (a, b) = ([0.3, -1.2, 0.7, 0.4], [1.1, 0.2, -0.5, 0.9]);
        let at = |x: [f32; 4], pos: usize| {
            let mut y = x;
            rotate(&mut y, pos, &freqs);
            y
        };
        let norm = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
        assert!((norm(&at(a, 17)) - norm(&a)).abs() < 1e-5);
        // The score between positions 5 and 3 equals the score between 9 and 7.
        let near = dot(&at(a, 5), &at(b, 3));
        let far = dot(&at(a, 9), &at(b, 7));
        assert!((near - far).abs() < 1e-4, "{near} vs {far}");
    }

    #[test]
    fn silu_matches_known_values() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((silu(-1.0) + 0.268_941_4).abs() < 1e-6);
    }

    #[test]
    fn tokenizer_applies_merges_in_learned_order() {
        // 256 = "ab", 257 = "abc" (256 + 'c'), 258 = "bc". "abc" must become [257], not ['a', 258].
        let t = Tokenizer::new(vec![(97, 98), (256, 99), (98, 99)]);
        assert_eq!(t.encode(b"abc"), vec![257]);
        assert_eq!(t.encode(b"bc"), vec![258]);
        assert_eq!(t.encode(b"abab"), vec![256, 256]);
        assert_eq!(t.decode(&[257, 258, 256]), b"abcbcab");
    }

    #[test]
    fn tokenizer_never_merges_across_a_chunk_boundary() {
        // "b a": the merge ('b', ' ') must not fire, because the space starts the next chunk.
        let t = Tokenizer::new(vec![(98, 32), (32, 97)]);
        assert_eq!(t.encode(b"b a"), vec![98, 257]);
        assert_eq!(t.encode(b""), Vec::<u32>::new());
        assert_eq!(t.encode(b"  "), vec![32, 32]);
    }

    #[test]
    fn greedy_sampling_is_the_argmax_and_top_k_never_leaves_the_top() {
        let logits = [0.1, 2.0, -1.0, 1.5, 0.0];
        let mut rng = Rng::new(1);
        assert_eq!(sample(&logits, 0.0, 0, &mut rng), 1);
        for _ in 0..500 {
            let pick = sample(&logits, 1.5, 2, &mut rng);
            assert!(pick == 1 || pick == 3, "sampled {pick} outside the top two");
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let logits = [0.0, (3.0f32).ln()]; // probabilities 0.25 and 0.75
        let mut rng = Rng::new(42);
        let ones = (0..20000)
            .filter(|_| sample(&logits, 1.0, 0, &mut rng) == 1)
            .count();
        assert!(
            (14500..15500).contains(&ones),
            "token 1 drawn {ones} times out of 20000"
        );
    }

    #[test]
    fn rng_is_reproducible_and_in_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            let v = a.next_f32();
            assert_eq!(v, b.next_f32());
            assert!((0.0..1.0).contains(&v));
        }
        assert_ne!(Rng::new(1).next_f32(), Rng::new(2).next_f32());
    }
}
