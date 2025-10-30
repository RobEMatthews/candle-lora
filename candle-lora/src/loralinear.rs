use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Shape, Tensor, D};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;

use crate::{
    frozenlinear::FrozenLinear, LinearLayerLike, LoraConfig, Merge, MergeError, MergeErrorOrError,
    Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraLinear {
    old: Arc<FrozenLinear>,
    ff_a: Linear,
    ff_b: Linear,
    scale: Option<f64>,
    dropout: Option<Arc<Dropout>>,
    merged: bool,
    magnitude: Option<Tensor>, // DoRA magnitude vector (if present, this is a DoRA adapter)
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl LoraLinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LoraLinearConfig {
            in_features,
            out_features,
        }
    }
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        // Try HF format first (lora_A, lora_B), fallback to candle format (a{id}, b{id})
        let a = vb
            .pp("lora_A")
            .get_with_hints(
                (config.rank, linear_config.in_features),
                "weight",
                init::DEFAULT_KAIMING_NORMAL,
            )
            .or_else(|_| {
                // Fallback to candle-lora format
                vb.pp(format!("a{id}")).get_with_hints(
                    (config.rank, linear_config.in_features),
                    "weight",
                    init::DEFAULT_KAIMING_NORMAL,
                )
            })?;

        let b = vb
            .pp("lora_B")
            .get_with_hints(
                (linear_config.out_features, config.rank),
                "weight",
                init::ZERO,
            )
            .or_else(|_| {
                // Fallback to candle-lora format
                vb.pp(format!("b{id}")).get_with_hints(
                    (linear_config.out_features, config.rank),
                    "weight",
                    init::ZERO,
                )
            })?;

        // Try to load magnitude vector for DoRA
        // HF format: lora_magnitude_vector, candle format: magnitude{id}
        let magnitude = vb
            .pp("lora_magnitude_vector")
            .get((linear_config.out_features,), "weight")
            .or_else(|_| {
                vb.pp(format!("magnitude{id}"))
                    .get((linear_config.out_features,), "weight")
            })
            .ok(); // Use .ok() to make it optional - if not found, returns None

        Ok(LoraLinear {
            old: Arc::new(FrozenLinear::new_from_linear(old)?),
            ff_a: Linear::new(a, None),
            ff_b: Linear::new(b, None),
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            dropout: config.dropout.map(|x| Arc::new(Dropout::new(x))),
            merged: false,
            magnitude,
            prefix: vb.prefix(),
            id,
        })
    }
}

impl Merge for LoraLinear {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        // Compute BA
        let ba = self
            .ff_b
            .weight()
            .matmul(self.ff_a.weight())
            .map_err(Either::Right)?;

        // Apply scale
        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // If magnitude vector exists, this is DoRA - apply normalization
        if let Some(ref magnitude) = self.magnitude {
            // DoRA: W' = m ⊙ (W₀ + BA) / ||W₀ + BA||_c - W₀
            // First compute W₀ + BA
            let combined = (self.old.weight() + &scaled_ba).map_err(Either::Right)?;

            // Compute column-wise L2 norm
            let norm = combined
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(0)
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?;
            // Add epsilon to avoid division by zero
            let norm = (norm + 1e-8).map_err(Either::Right)?;

            // Normalize column-wise
            let normalized = combined.broadcast_div(&norm).map_err(Either::Right)?;

            // Apply magnitude scaling and subtract original weight to get delta
            // Reshape magnitude from [out_features] to [out_features, 1] for broadcasting
            let magnitude_reshaped = magnitude.unsqueeze(1).map_err(Either::Right)?;
            let scaled = normalized
                .broadcast_mul(&magnitude_reshaped)
                .map_err(Either::Right)?;
            (scaled - self.old.weight()).map_err(Either::Right)
        } else {
            // Standard LoRA
            Ok(scaled_ba)
        }
    }

    fn merge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if self.merged {
            Err(Either::Left(MergeError::AlreadyMerged))
        } else {
            self.old = Arc::new(
                FrozenLinear::new(
                    (self.old.weight() + self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = true;
            Ok(())
        }
    }

    fn unmerge_weights(&mut self) -> std::result::Result<(), MergeErrorOrError> {
        if !self.merged {
            Err(Either::Left(MergeError::NotMerged))
        } else {
            self.old = Arc::new(
                FrozenLinear::new(
                    (self.old.weight() - self.get_delta_weight()?).map_err(Either::Right)?,
                    self.old.bias().cloned(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            self.old.forward(input)
        } else if let Some(ref magnitude) = self.magnitude {
            // DoRA forward pass
            let input_processed = if let Some(ref dropout) = self.dropout {
                dropout.forward(input, true)?
            } else {
                input.clone()
            };

            // Compute base + LoRA output
            let base_output = self.old.forward(&input_processed)?;
            let lora_output = self.ff_b.forward(&self.ff_a.forward(&input_processed)?)?;
            let lora_output = if let Some(scale) = self.scale {
                lora_output.mul(scale)?
            } else {
                lora_output
            };

            // Combine outputs
            let combined = (base_output + lora_output)?;

            // For DoRA: normalize then apply magnitude
            // This is an approximation for efficiency during inference
            // Full DoRA would normalize weights, but we normalize outputs instead
            let norm = combined.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            let norm = (norm + 1e-8)?;
            let normalized = combined.broadcast_div(&norm)?;

            // Apply magnitude scaling
            normalized.broadcast_mul(magnitude)
        } else {
            // Standard LoRA forward pass
            let mut result = self.old.forward(input)?;
            if let Some(scale) = self.scale {
                let input_new = if self.dropout.is_some() {
                    self.dropout.as_ref().unwrap().forward(input, true)?
                } else {
                    input.clone()
                };

                let lora_delta = self
                    .ff_b
                    .forward(&self.ff_a.forward(&input_new)?)?
                    .mul(scale)?;
                result = (result + lora_delta)?;
            }
            Ok(result)
        }
    }
}

impl Saveable for LoraLinear {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.ff_a.weight().clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.ff_b.weight().clone(),
        );
        // Save magnitude vector if this is a DoRA adapter
        if let Some(ref magnitude) = self.magnitude {
            accum.insert(
                self.prefix.clone() + &format!(".magnitude{}.weight", self.id),
                magnitude.clone(),
            );
        }
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
}
