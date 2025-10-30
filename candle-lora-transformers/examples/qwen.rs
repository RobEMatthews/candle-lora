#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use candle_lora::LoraConfig;
use clap::Parser;

use candle_core::{DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::{fs, io::Write, path::PathBuf};

use candle_lora_transformers::{qwen as model, varbuilder_utils::from_mmaped_safetensors};
use model::{Config, Qwen3};

const EOS_TOKEN: &str = "<|endoftext|>";
const DEFAULT_PROMPT: &str = "Hello, I am";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 50)]
    sample_len: usize,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than bf16 (tests fail otherwise -- to check)
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Path to base model directory containing safetensors files
    #[arg(long)]
    base_model: String,

    /// Path to adapter safetensors file
    #[arg(long)]
    adapter: String,

    /// LoRA rank (must match the rank used during adapter training)
    #[arg(long, default_value_t = 128)]
    rank: usize,

    /// LoRA alpha (must match the alpha used during adapter training)
    #[arg(long, default_value_t = 256.0)]
    alpha: f64,

    /// Merge adapter weights into base weights
    #[arg(long)]
    merge: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    println!("Base model: {}", args.base_model);
    println!("Adapter: {}", args.adapter);
    println!("LoRA config: rank={}, alpha={}", args.rank, args.alpha);
    println!("Merge weights: {}", args.merge);

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    println!("Device: {:?}, DType: {:?}\n", device, dtype);

    // Load config
    println!("Loading model config...");
    let config_path = PathBuf::from(&args.base_model).join("config.json");
    let config_json = fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config_json)?;
    println!("Config loaded:");
    println!("   - Hidden size: {}", config.hidden_size);
    println!("   - Num layers: {}", config.num_hidden_layers);
    println!("   - Vocab size: {}\n", config.vocab_size);

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_path = PathBuf::from(&args.base_model).join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    println!("Tokenizer loaded\n");

    // Find all model safetensors files
    println!("Loading base model weights...");
    let model_dir = PathBuf::from(&args.base_model);
    let mut model_files = Vec::new();

    for entry in fs::read_dir(&model_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("model-") && name.ends_with(".safetensors") {
                model_files.push(path);
            }
        }
    }
    model_files.sort();

    if model_files.is_empty() {
        bail!("No model safetensors files found in {}", args.base_model);
    }

    println!("Found {} model file(s):", model_files.len());
    for file in &model_files {
        println!("   - {}", file.display());
    }
    println!();

    // Combine base model files + adapter
    println!("Loading base model + adapter weights into combined VarBuilder...");
    let mut all_files = model_files.clone();
    all_files.push(PathBuf::from(&args.adapter));

    println!("   Loading {} files total:", all_files.len());
    for file in &all_files {
        println!("   - {}", file.display());
    }

    let vb = from_mmaped_safetensors(&all_files, dtype, &device, false)?;
    println!("Combined VarBuilder created with base + adapter weights\n");

    // Create LoRA config
    let lora_config = LoraConfig::new(args.rank, args.alpha, None);

    // Create model with LoRA
    println!("🔧 Creating Qwen3 model with LoRA adapter...");
    let model = Qwen3::new(&config, vb, args.merge, lora_config)?;

    if args.merge {
        println!("Model created with merged adapter weights (adapter baked into base)\n");
    } else {
        println!("Model created with separate adapter weights\n");
    }

    // Tokenize prompt
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    println!("Prompt: {:?}", prompt);

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    println!("Tokenized: {} tokens\n", tokens.len());

    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);

    // Generate
    println!("Generating {} tokens...", args.sample_len);
    print!("{}", prompt);
    std::io::stdout().flush()?;

    let mut logits_processor = LogitsProcessor::new(args.seed, Some(args.temperature), args.top_p);
    let start_gen = std::time::Instant::now();
    let mut token_generated = 0;

    for _index in 0..args.sample_len {
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        let logits = logits.squeeze(0)?.get(tokens.len() - 1)?;

        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ");
            print!("{}", text);
            std::io::stdout().flush()?;
        }

        if Some(next_token) == eos_token_id {
            break;
        }
    }

    let dt = start_gen.elapsed();
    println!(
        "\nGenerated {} tokens in {:.2}s ({:.2} token/s)\n",
        token_generated,
        dt.as_secs_f64(),
        token_generated as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
