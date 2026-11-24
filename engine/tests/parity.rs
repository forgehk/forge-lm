//! The engine against the model PyTorch trained.
//!
//! `models/reference_logits.txt` holds the logits PyTorch computed for a prompt, one
//! row per position. The engine has to reproduce every row from the same model file.

use std::fs;
use std::path::PathBuf;

use forgelm::{generate, generate_with, sample, Error, Model, Rng, Window};

fn models() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn load() -> Model {
    Model::load(models().join("shakespeare.flm")).expect("the released model loads")
}

fn reference() -> (Vec<u32>, Vec<Vec<f32>>) {
    let text = fs::read_to_string(models().join("reference_logits.txt")).unwrap();
    let mut lines = text.lines();
    let ids = lines
        .next()
        .unwrap()
        .split(' ')
        .map(|t| t.parse().unwrap())
        .collect();
    let rows = lines
        .map(|line| line.split(' ').map(|v| v.parse().unwrap()).collect())
        .collect();
    (ids, rows)
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len())
        .max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap())
        .unwrap()
}

#[test]
fn logits_match_pytorch_at_every_position() {
    let model = load();
    let (ids, expected) = reference();
    assert_eq!(expected.len(), ids.len());
    let mut session = model.session();
    let mut worst = 0.0f32;
    for (&id, want) in ids.iter().zip(&expected) {
        let got = session.feed(id).unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((g - w).abs());
        }
        assert_eq!(
            argmax(got),
            argmax(want),
            "the most likely next token differs"
        );
    }
    println!(
        "largest difference from PyTorch over {} logits: {worst:e}",
        ids.len() * expected[0].len()
    );
    assert!(worst < 5e-4, "largest difference from PyTorch is {worst}");
}

#[test]
fn tokenizer_matches_python_on_every_case() {
    let model = load();
    let cases = fs::read_to_string(models().join("tokenizer_cases.txt")).unwrap();
    let mut checked = 0;
    for row in cases.lines() {
        let (text_hex, encoded) = row.split_once('\t').unwrap();
        let text: Vec<u8> = if text_hex == "-" {
            Vec::new()
        } else {
            (0..text_hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&text_hex[i..i + 2], 16).unwrap())
                .collect()
        };
        let expected: Vec<u32> = if encoded == "-" {
            Vec::new()
        } else {
            encoded.split(' ').map(|t| t.parse().unwrap()).collect()
        };
        assert_eq!(
            model.tokenizer.encode(&text),
            expected,
            "ids differ for {:?}",
            String::from_utf8_lossy(&text)
        );
        assert_eq!(model.tokenizer.decode(&expected), text);
        checked += 1;
    }
    assert!(checked >= 8);
}

#[test]
fn header_and_parameter_count_match_the_training_log() {
    let model = load();
    let log = fs::read_to_string(models().join("training_log.json")).unwrap();
    assert!(log.contains(&format!("\"parameters\": {}", model.n_params())));
    assert!(log.contains(&format!("\"vocab_size\": {}", model.config.vocab_size)));
    assert_eq!(model.tokenizer.vocab_size(), model.config.vocab_size);
}

#[test]
fn a_fresh_session_gives_the_same_logits_as_a_reset_one() {
    let model = load();
    let (ids, _) = reference();
    let mut session = model.session();
    let first: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| session.feed(id).unwrap().to_vec())
        .collect();
    session.reset();
    assert_eq!(session.position(), 0);
    let second: Vec<Vec<f32>> = ids
        .iter()
        .map(|&id| session.feed(id).unwrap().to_vec())
        .collect();
    assert_eq!(first, second);
}

#[test]
fn generation_is_reproducible_and_depends_on_the_seed() {
    let model = load();
    let prompt = model.tokenizer.encode(b"ROMEO:\n");
    let run = |seed| generate(&model, &prompt, 60, 0.8, 40, seed, |_| {}).unwrap();
    assert_eq!(run(1), run(1));
    assert_ne!(run(1), run(2));
    assert_eq!(run(1).len(), 60);
}

#[test]
fn greedy_generation_follows_the_argmax() {
    let model = load();
    let prompt = model.tokenizer.encode(b"First Citizen:\n");
    let made = generate(&model, &prompt, 12, 0.0, 0, 9, |_| {}).unwrap();
    let mut session = model.session();
    let mut logits = Vec::new();
    for &id in &prompt {
        logits = session.feed(id).unwrap().to_vec();
    }
    for &token in &made {
        assert_eq!(token as usize, argmax(&logits));
        logits = session.feed(token).unwrap().to_vec();
    }
}

#[test]
fn generation_runs_past_the_context_length() {
    let model = load();
    let block = model.config.block_size;
    let prompt = model.tokenizer.encode(b"KING:\n");
    let mut emitted = 0;
    let made = generate(&model, &prompt, block * 2 + 7, 0.8, 40, 3, |_| emitted += 1).unwrap();
    assert_eq!(made.len(), block * 2 + 7);
    assert_eq!(emitted, made.len());
    assert!(made.iter().all(|&t| (t as usize) < model.config.vocab_size));
}

#[test]
fn sliding_generation_predicts_from_exactly_the_last_block_of_tokens() {
    // PyTorch's generate feeds ids[-block_size:] for every new token. Rebuild
    // that by hand, one fresh session per token, and sample with the same
    // generator: the sliding engine must produce the same tokens.
    let model = load();
    let block = model.config.block_size;
    let prompt = model.tokenizer.encode(b"First Citizen:\n");
    let count = block + 40;
    let made = generate_with(&model, &prompt, count, 0.8, 40, 5, Window::Slide, |_| {}).unwrap();

    let mut history = prompt.clone();
    let mut rng = Rng::new(5);
    for _ in 0..count {
        let start = history.len().saturating_sub(block);
        let mut session = model.session();
        let mut logits = Vec::new();
        for &id in &history[start..] {
            logits = session.feed(id).unwrap().to_vec();
        }
        history.push(sample(&logits, 0.8, 40, &mut rng));
    }
    assert_eq!(made, history[prompt.len()..]);

    let halved = generate(&model, &prompt, count, 0.8, 40, 5, |_| {}).unwrap();
    assert_ne!(
        halved, made,
        "halving the context should not match the window"
    );
}

#[test]
fn the_window_policy_only_matters_once_the_context_is_full() {
    let model = load();
    let block = model.config.block_size;
    let prompt = model.tokenizer.encode(b"KING:\n");
    let count = block - prompt.len();
    let run = |window| generate_with(&model, &prompt, count, 0.8, 40, 11, window, |_| {}).unwrap();
    assert_eq!(run(Window::Halve), run(Window::Slide));
    assert_eq!(
        run(Window::Halve),
        generate(&model, &prompt, count, 0.8, 40, 11, |_| {}).unwrap()
    );
    let long =
        |window| generate_with(&model, &prompt, block + 40, 0.8, 40, 11, window, |_| {}).unwrap();
    assert_eq!(long(Window::Slide).len(), block + 40);
    assert_eq!(long(Window::Slide), long(Window::Slide));
}

#[test]
fn generated_text_is_valid_looking_english() {
    let model = load();
    let prompt = model.tokenizer.encode(b"ROMEO:\n");
    let made = generate(&model, &prompt, 120, 0.8, 40, 1, |_| {}).unwrap();
    let text = String::from_utf8_lossy(&model.tokenizer.decode(&made)).to_lowercase();
    let common = [
        "the", "and", "to", "of", "my", "you", "that", "not", "with", "thou", "thy", "is",
    ];
    let hits = text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| common.contains(w))
        .count();
    assert!(hits >= 5, "only {hits} common words in: {text}");
}

#[test]
fn the_context_limit_and_bad_tokens_are_errors_not_panics() {
    let model = load();
    let mut session = model.session();
    assert!(matches!(
        session.feed(model.config.vocab_size as u32),
        Err(Error::BadToken(_))
    ));
    for _ in 0..model.config.block_size {
        session.feed(10).unwrap();
    }
    assert!(matches!(session.feed(10), Err(Error::ContextFull(_))));
}

#[test]
fn damaged_files_are_rejected() {
    let data = fs::read(models().join("shakespeare.flm")).unwrap();
    let is_format = |bytes: &[u8]| matches!(Model::from_bytes(bytes), Err(Error::Format(_)));

    assert!(is_format(b""));
    assert!(is_format(b"FLM"));
    let mut wrong_magic = data.clone();
    wrong_magic[0] = b'X';
    assert!(is_format(&wrong_magic));
    assert!(is_format(&data[..data.len() - 1]));
    assert!(is_format(&data[..100]));
    let mut long = data.clone();
    long.extend_from_slice(&[0; 4]);
    assert!(is_format(&long));
    let mut zero_heads = data.clone();
    zero_heads[16..20].copy_from_slice(&0u32.to_le_bytes());
    assert!(is_format(&zero_heads));
    let mut huge = data.clone();
    huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(is_format(&huge));
    let mut bad_merge = data.clone();
    bad_merge[36..40].copy_from_slice(&9999u32.to_le_bytes());
    assert!(is_format(&bad_merge));
    assert!(matches!(
        Model::load(models().join("missing.flm")),
        Err(Error::Io(_))
    ));
}

#[test]
fn sampling_at_low_temperature_concentrates_on_the_likely_tokens() {
    let model = load();
    let (ids, _) = reference();
    let mut session = model.session();
    let mut logits = Vec::new();
    for &id in &ids {
        logits = session.feed(id).unwrap().to_vec();
    }
    let top = argmax(&logits) as u32;
    let mut rng = Rng::new(11);
    let picks = (0..300)
        .filter(|_| sample(&logits, 0.05, 0, &mut rng) == top)
        .count();
    assert!(
        picks > 270,
        "the top token was drawn only {picks} of 300 times at temperature 0.05"
    );
}
