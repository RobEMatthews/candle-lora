# Full Test Guide: Base Model + LoRA Adapter

## What This Does

The `qwen.rs` example **properly integrates your base model with your LoRA adapter**:

1. ✅ **Loads base model weights** (from any Qwen model safetensors files)
2. ✅ **Loads your adapter weights** (from adapter_model.safetensors)
3. ✅ **Combines them into a single VarBuilder** (both weight sets accessible)
4. ✅ **Creates model that uses BOTH**:
   - Base linear layers use base weights
   - LoraLinear wraps them and applies your adapter
5. ✅ **Runs actual inference** with the combined model

## How It Works

```rust
// Load ALL safetensors files (base + adapter) into one VarBuilder
let combined_vb = VarBuilder::from_mmaped_safetensors(
    &["model-00001.safetensors", "model-00002.safetensors", "adapter_model.safetensors"],
    dtype,
    &device
)?;

// Now VarBuilder contains:
// - "model.layers.0.self_attn.q_proj.weight" (base weight)
// - "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" (adapter)
// - "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" (adapter)

// When creating the model:
Qwen3::new(&config, combined_vb.pp("model"), merge=false, lora_config)?;
//          ↓
// Each layer does:
// 1. candle_nn::linear() loads base weight from "model.layers.*.*.weight"
// 2. LoraLinear wraps it and loads lora_A/B from adapter path
// 3. Forward pass: output = base_output + (lora_B @ lora_A @ input) * scale
```

## Setup

### 1. Download Base Model

Choose any Qwen model (Qwen2, Qwen2.5, Qwen3 - all sizes: 0.5B, 1.5B, 7B, 14B, 72B, etc.)

**Example: Qwen3-8B (~16GB)**

**Option A: Fast Download with aria2c (Recommended)**
```bash
# Install aria2c if needed
brew install aria2

# Set your model and create directory
MODEL="Qwen/Qwen3-8B"
DIR="models/qwen3-8b"
mkdir -p $DIR && cd $DIR

# Download with aria2c (much faster - 16 parallel connections)
BASE_URL="https://huggingface.co/${MODEL}/resolve/main"
aria2c -x 16 -s 16 -c "${BASE_URL}/config.json"
aria2c -x 16 -s 16 -c "${BASE_URL}/tokenizer.json"
aria2c -x 16 -s 16 -c "${BASE_URL}/model-00001-of-00005.safetensors"
aria2c -x 16 -s 16 -c "${BASE_URL}/model-00002-of-00005.safetensors"
aria2c -x 16 -s 16 -c "${BASE_URL}/model-00003-of-00005.safetensors"
aria2c -x 16 -s 16 -c "${BASE_URL}/model-00004-of-00005.safetensors"
aria2c -x 16 -s 16 -c "${BASE_URL}/model-00005-of-00005.safetensors"
```

**Note:** Number of files varies by model size. Check the HuggingFace repo to see how many files exist (model-00001-of-00XXX).

**Option B: Using HuggingFace CLI (Easiest)**
```bash
# Login first
hf auth login

# Download any Qwen model
hf download Qwen/Qwen3-8B \
    --include "*.safetensors" "config.json" "tokenizer.json" \
    --local-dir ./models/qwen3-8b

# Or other models:
# hf download Qwen/Qwen2-7B --local-dir ./models/qwen2-7b
# hf download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b
```

**Option C: Regular wget**
```bash
# Check HuggingFace repo for exact file names first!
BASE_URL="https://huggingface.co/Qwen/Qwen3-8B/resolve/main"
wget -c "${BASE_URL}/config.json"
wget -c "${BASE_URL}/tokenizer.json"
wget -c "${BASE_URL}/model-00001-of-00005.safetensors"
# ... (download all model-*.safetensors files)
```

### 2. Verify Your Files

Example structure (varies by model size):

```
./models/
└── qwen3-8b/              # Your model directory
    ├── model-*.safetensors    # One or more model weight files
    ├── config.json            # Model configuration
    └── tokenizer.json         # Tokenizer

./
└── adapter_model.safetensors  # Your LoRA adapter
```

**Examples by model size:**
- **Small models (0.5B-1.5B)**: Usually 1-2 files (~1-3 GB)
- **Medium models (7B-8B)**: Usually 2-5 files (~14-16 GB)  
- **Large models (14B-72B)**: Usually 5-20+ files (~28-145 GB)

## Running The Test

```bash
# Basic test
cargo run --example qwen -- \
    --base-model ./models/qwen3-8b \
    --adapter adapter_model.safetensors \
    --rank 128 \
    --alpha 256

# With custom prompt
cargo run --example qwen -- \
    --base-model ./models/qwen3-8b \
    --adapter adapter_model.safetensors \
    --rank 128 \
    --alpha 256 \
    --prompt "Explain quantum computing:" \
    --sample-len 100

# Use merged weights (faster)
cargo run --example qwen -- \
    --base-model ./models/qwen3-8b \
    --adapter adapter_model.safetensors \
    --rank 128 \
    --alpha 256 \
    --merge

# CPU only (slower but works without GPU)
cargo run --example qwen -- \
    --base-model ./models/qwen3-8b \
    --adapter adapter_model.safetensors \
    --rank 128 \
    --alpha 256 \
    --cpu
```

## Command-Line Options

- `--base-model <PATH>` - Path to base model directory (required)
- `--adapter <PATH>` - Path to adapter safetensors file (required)
- `--rank <N>` - LoRA rank (default: 128, matches your adapter)
- `--alpha <F>` - LoRA alpha (default: 256.0, common: rank × 2)
- `--prompt <TEXT>` - Input prompt (default: "Hello, I am")
- `--sample-len <N>` - Tokens to generate (default: 50)
- `--temperature <F>` - Sampling temperature (default: 0.8)
- `--merge` - Merge adapter into base weights (faster inference)
- `--cpu` - Run on CPU instead of GPU

## Merge vs Non-Merge

### Without `--merge` (default)
```
Forward pass: output = Linear(base_weight) + LoRA_delta
- Keeps weights separate
- Slightly slower (two matrix multiplications)
- Can easily swap adapters
```

### With `--merge`
```
At init: base_weight = base_weight + (lora_B @ lora_A) * scale
Forward pass: output = Linear(merged_weight)
- Single merged weight
- Faster inference (one matrix multiplication)
- Adapter "baked in"
```

## What To Expect

```
🚀 Qwen + LoRA Adapter Full Test
============================================================
Base model: ./models/qwen3-8b
Adapter: adapter_model.safetensors
LoRA config: rank=128, alpha=256
Merge weights: false
============================================================

📱 Device: Cuda(0), DType: BF16

📋 Loading model config...
✅ Config loaded:
   - Hidden size: 4096
   - Num layers: 28
   - Vocab size: 152064

🔤 Loading tokenizer...
✅ Tokenizer loaded

📦 Loading base model weights...
✅ Found 2 model file(s):
   - ./models/qwen3-8b/model-00001-of-00002.safetensors
   - ./models/qwen3-8b/model-00002-of-00002.safetensors

📦 Loading base model + adapter weights into combined VarBuilder...
   Loading 3 files total:
   - ./models/qwen3-8b/model-00001-of-00002.safetensors
   - ./models/qwen3-8b/model-00002-of-00002.safetensors
   - adapter_model.safetensors
✅ Combined VarBuilder created with base + adapter weights

🔍 Detecting adapter structure...
✅ Adapter prefix: base_model.model.model

🔧 Creating Qwen3 model with LoRA adapter...
✅ Model created with separate adapter weights

💬 Prompt: "Hello, I am"
✅ Tokenized: 4 tokens

🎯 Generating 50 tokens...
------------------------------------------------------------
Hello, I am a student at the University of...
```

## This IS Full Integration!

✅ Base model loaded  
✅ Adapter loaded  
✅ **Both combined in single VarBuilder**  
✅ **Model uses base weights + adapter deltas**  
✅ Actual inference with fine-tuned behavior  

This is the complete, proper way to test your adapter!
