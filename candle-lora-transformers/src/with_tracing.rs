//! Tracing layers with proper LoRA support.

use candle_core::{Module, Result, Tensor};
use candle_lora::{
    EmbeddingLayerLike, LoraConfig, LoraEmbedding, LoraEmbeddingConfig, LoraLinear,
    LoraLinearConfig,
};
use candle_nn::{Conv2d, Linear, VarBuilder};
use std::sync::Arc;

/// Traced LoRA Embedding wrapper
#[derive(Debug)]
pub struct TracedLoraEmbedding {
    inner: Arc<LoraEmbedding>,
    span: tracing::Span,
}

impl TracedLoraEmbedding {
    pub fn new(
        d1: usize,
        d2: usize,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let embed_config = LoraEmbeddingConfig::new(d1, d2);
        let base_embed = candle_nn::embedding(d1, d2, vb.clone())?;
        let span = tracing::span!(tracing::Level::TRACE, "embedding");

        let mut lora_embed = LoraEmbedding::new(&base_embed, &embed_config, &lora_config, &vb, 0)?;

        if merge {
            use candle_lora::Merge;
            lora_embed.merge_weights().ok(); // Ignore error if no LoRA weights found
        }

        Ok(Self {
            inner: Arc::new(lora_embed),
            span,
        })
    }

    pub fn embeddings(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

impl Module for TracedLoraEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

/// Traced LoRA Linear wrapper - properly wraps LoraLinear instead of plain Linear
#[derive(Debug)]
pub struct TracedLoraLinear {
    inner: Arc<LoraLinear>,
    span: tracing::Span,
}

impl TracedLoraLinear {
    pub fn from_weights(
        weights: Tensor,
        bias: Option<Tensor>,
        vb: VarBuilder,
        merge: bool,
        lora_config: LoraConfig,
    ) -> Result<Self> {
        let (in_features, out_features) = weights.dims2()?;
        let linear_config = LoraLinearConfig::new(in_features, out_features);
        let base_linear = Linear::new(weights, bias);
        let span = tracing::span!(tracing::Level::TRACE, "linear");

        let mut lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &vb, 0)?;

        if merge {
            use candle_lora::Merge;
            lora_linear.merge_weights().ok(); // Ignore error if no LoRA weights found
        }

        Ok(Self {
            inner: Arc::new(lora_linear),
            span,
        })
    }
}

/// Create a traced LoRA linear layer
pub fn linear(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<TracedLoraLinear> {
    let linear_config = LoraLinearConfig::new(d1, d2);
    let base_linear = candle_nn::linear(d1, d2, vb.clone())?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");

    let mut lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &vb, 0)?;

    if merge {
        use candle_lora::Merge;
        lora_linear.merge_weights().ok(); // Ignore error if no LoRA weights found
    }

    Ok(TracedLoraLinear {
        inner: Arc::new(lora_linear),
        span,
    })
}

/// Create a traced LoRA linear layer without bias
pub fn linear_no_bias(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    merge: bool,
    lora_config: LoraConfig,
) -> Result<TracedLoraLinear> {
    let linear_config = LoraLinearConfig::new(d1, d2);
    let base_linear = candle_nn::linear_no_bias(d1, d2, vb.clone())?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");

    let mut lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &vb, 0)?;

    if merge {
        use candle_lora::Merge;
        lora_linear.merge_weights().ok(); // Ignore error if no LoRA weights found
    }

    Ok(TracedLoraLinear {
        inner: Arc::new(lora_linear),
        span,
    })
}

impl Module for TracedLoraLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// Wrap the conv2d op to provide some tracing.
#[derive(Debug, Clone)]
pub struct TracedLoraConv2d {
    inner: Conv2d,
    span: tracing::Span,
}

impl Module for TracedLoraConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv2dConfig,
    vs: candle_nn::VarBuilder,
) -> Result<TracedLoraConv2d> {
    let span = tracing::span!(tracing::Level::TRACE, "conv2d");
    let inner = candle_nn::conv2d(in_channels, out_channels, kernel_size, cfg, vs)?;
    Ok(TracedLoraConv2d { inner, span })
}

// QMatMul wrapper adding some tracing.
#[derive(Clone)]
pub struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub fn new(
        out_dim: usize,
        in_dim: usize,
        vb: candle_transformers::quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get((in_dim, out_dim), "weight")?;
        let inner = candle_core::quantized::QMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}
