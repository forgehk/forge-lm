//! Command line front end for the forge-lm inference engine.

use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

use forgelm::{generate_with, Model, Window};

const USAGE: &str = "usage:
  forge-lm info <model.flm>
  forge-lm encode <model.flm> <text>
  forge-lm generate <model.flm> [--prompt TEXT] [--tokens N] [--temperature T] [--top-k K] [--seed S]
                     [--window halve|slide]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("forge-lm: {message}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let (command, path) = match args {
        [command, path, ..] => (command.as_str(), path),
        _ => return Err(USAGE.into()),
    };
    let model = Model::load(path).map_err(|err| err.to_string())?;
    let rest = &args[2..];
    match command {
        "info" => {
            let c = &model.config;
            println!("parameters  {}", model.n_params());
            println!(
                "vocabulary  {} tokens ({} merges)",
                c.vocab_size,
                c.vocab_size - 256
            );
            println!("layers      {}", c.n_layers);
            println!("heads       {}", c.n_heads);
            println!("width       {} (feed-forward {})", c.dim, c.hidden);
            println!("context     {} tokens", c.block_size);
            Ok(())
        }
        "encode" => {
            let text = rest.first().ok_or(USAGE)?;
            let ids: Vec<String> = model
                .tokenizer
                .encode(text.as_bytes())
                .iter()
                .map(u32::to_string)
                .collect();
            println!("{}", ids.join(" "));
            Ok(())
        }
        "generate" => {
            let mut prompt = String::from("\n");
            let (mut tokens, mut temperature, mut top_k, mut seed) =
                (200usize, 0.8f32, 40usize, 1u64);
            let mut window = Window::Halve;
            let mut flags = rest.iter();
            while let Some(flag) = flags.next() {
                let value = flags
                    .next()
                    .ok_or_else(|| format!("{flag} needs a value"))?;
                let bad = || format!("{value} is not a valid value for {flag}");
                match flag.as_str() {
                    "--prompt" => prompt = value.clone(),
                    "--tokens" => tokens = value.parse().map_err(|_| bad())?,
                    "--temperature" => temperature = value.parse().map_err(|_| bad())?,
                    "--top-k" => top_k = value.parse().map_err(|_| bad())?,
                    "--seed" => seed = value.parse().map_err(|_| bad())?,
                    "--window" => {
                        window = match value.as_str() {
                            "halve" => Window::Halve,
                            "slide" => Window::Slide,
                            _ => return Err(bad()),
                        }
                    }
                    _ => return Err(format!("unknown option {flag}\n{USAGE}")),
                }
            }
            let ids = model.tokenizer.encode(prompt.as_bytes());
            let stdout = io::stdout();
            let mut out = stdout.lock();
            out.write_all(prompt.as_bytes())
                .map_err(|err| err.to_string())?;
            let started = Instant::now();
            let emit = |token| {
                let _ = out.write_all(model.tokenizer.token_bytes(token));
                let _ = out.flush();
            };
            generate_with(&model, &ids, tokens, temperature, top_k, seed, window, emit)
                .map_err(|err| err.to_string())?;
            let _ = out.write_all(b"\n");
            let seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "{tokens} tokens in {seconds:.2}s ({:.0} tokens/s)",
                tokens as f64 / seconds.max(1e-9)
            );
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
