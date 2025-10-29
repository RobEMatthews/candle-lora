use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_lora::LoraConfig;
use candle_lora_macro::{replace_layer_fields, AutoLoraConvert};
use candle_nn::{Activation, Embedding, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

use crate::with_tracing::{linear, linear_no_bias, TracedLoraLinear};

/// Qwen3 model configuration
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
    pub use_flash_attn: bool,
}

impl Config {
    /// Configuration for Qwen3-7B model
    pub fn qwen3_7b(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            max_position_embeddings: 32768,
            sliding_window: Some(32768),
            max_window_layers: 28,
            tie_word_embeddings: false,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Swiglu,
            use_flash_attn,
        }
    }

    /// Configuration for Qwen3-14B model
    pub fn qwen3_14b(use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 13696,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            sliding_window: Some(32768),
            max_window_layers: 40,
            tie_word_embeddings: false,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Swiglu,
            use_flash_attn,
        }
    }
}

/// RMS Normalization layer
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.to_dtype(internal_dtype)?;
        let mean_x2 = (x.sqr()?.sum_keepdim(D::Minus1)? / x.dims()[x.dims().len() - 1] as f64)?;
        let x_normed = x.broadcast_div(&(mean_x2 + self.eps)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        Ok(x)
    }
}

/// Rotary Position Embedding
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Multi-head attention with LoRA support
#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
struct Attention {
    q_proj: TracedLoraLinear,
    k_proj: TracedLoraLinear,
    v_proj: TracedLoraLinear,
    o_proj: TracedLoraLinear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    use_flash_attn: bool,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        let q_proj = linear(
            hidden_sz,
            num_heads * head_dim,
            vb.pp("q_proj"),
            merge,
            lora_config.clone(),
        )?;
        let k_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
            merge,
            lora_config.clone(),
        )?;
        let v_proj = linear(
            hidden_sz,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
            merge,
            lora_config.clone(),
        )?;
        let o_proj = linear_no_bias(
            num_heads * head_dim,
            hidden_sz,
            vb.pp("o_proj"),
            merge,
            lora_config,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states)?;

        let key_states = Self::repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = Self::repeat_kv(value_states, self.num_kv_groups)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, q_len, self.hidden_size))?;

        self.o_proj.forward(&attn_output)
    }

    fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
            xs.unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
        }
    }
}

/// Feed-forward network with LoRA support
#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
struct MLP {
    gate_proj: TracedLoraLinear,
    up_proj: TracedLoraLinear,
    down_proj: TracedLoraLinear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let gate_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("gate_proj"),
            merge,
            lora_config.clone(),
        )?;
        let up_proj = linear_no_bias(
            hidden_sz,
            intermediate_sz,
            vb.pp("up_proj"),
            merge,
            lora_config.clone(),
        )?;
        let down_proj = linear_no_bias(
            intermediate_sz,
            hidden_sz,
            vb.pp("down_proj"),
            merge,
            lora_config,
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

/// Transformer decoder layer with LoRA support
#[derive(Debug, AutoLoraConvert)]
#[replace_layer_fields]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            vb.pp("self_attn"),
            merge,
            lora_config.clone(),
        )?;
        let mlp = MLP::new(cfg, vb.pp("mlp"), merge, lora_config)?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Main Qwen3 model with LoRA support
#[derive(Debug)]
pub struct Qwen3 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: TracedLoraLinear,
    device: Device,
    dtype: DType,
}

impl Qwen3 {
    pub fn new(cfg: &Config, vb: VarBuilder, merge: bool, lora_config: LoraConfig) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                merge,
                lora_config.clone(),
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("lm_head"),
            merge,
            lora_config,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        // For now, using a simple causal mask
        let mask: Option<Tensor> = if seq_len > 1 {
            Some(self.make_causal_mask(seq_len)?)
        } else {
            None
        };

        for layer in self.layers.iter() {
            xs = layer.forward(&xs, mask.as_ref())?;
        }

        xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;
        Ok(logits)
    }

    pub fn get_last_hidden_states(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mask: Option<Tensor> = if seq_len > 1 {
            Some(self.make_causal_mask(seq_len)?)
        } else {
            None
        };

        for layer in self.layers.iter() {
            xs = layer.forward(&xs, mask.as_ref())?;
        }

        self.norm.forward(&xs)
    }

    fn make_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        // Create lower triangular mask (1s below diagonal, 0s above)
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY })
            })
            .collect();

        Tensor::from_vec(mask, (seq_len, seq_len), &self.device)?.to_dtype(self.dtype)
    }
}

/// Builder for creating Qwen3 models with custom LoRA configurations
pub struct Qwen3Builder {
    config: Config,
    merge: bool,
    lora_config: LoraConfig,
}

impl Qwen3Builder {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            merge: false,
            lora_config: LoraConfig::new(8, 16., None), // Default: rank=8, alpha=16
        }
    }

    pub fn with_lora_config(mut self, lora_config: LoraConfig) -> Self {
        self.lora_config = lora_config;
        self
    }

    pub fn with_merge(mut self, merge: bool) -> Self {
        self.merge = merge;
        self
    }

    pub fn build(self, vb: VarBuilder) -> Result<Qwen3> {
        Qwen3::new(&self.config, vb, self.merge, self.lora_config)
    }
}
