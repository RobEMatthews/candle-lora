use std::{collections::HashMap, ops::Mul, sync::Arc};

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{init, Embedding, Init, VarBuilder};
use either::Either;

use crate::{
    frozenembed::FrozenEmbedding, EmbeddingLayerLike, LoraConfig, Merge, MergeError,
    MergeErrorOrError, Saveable,
};

#[derive(Debug, Clone)]
pub struct LoraEmbedding {
    old: Arc<FrozenEmbedding>,
    embed_a: Embedding,
    a: Tensor,
    b: Tensor,
    scale: Option<f64>,
    merged: bool,
    magnitude: Option<Tensor>, // DoRA magnitude vector (if present, this is a DoRA adapter)
    prefix: String,
    id: usize,
}

#[derive(Clone, Debug)]
/// Configuration for LoraEmbedding, with `num_embeddings` vectors of `embedding_dim` size`.
pub struct LoraEmbeddingConfig {
    num_embeddings: usize,
    embedding_dim: usize,
}

impl LoraEmbeddingConfig {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        LoraEmbeddingConfig {
            num_embeddings,
            embedding_dim,
        }
    }
}

impl LoraEmbedding {
    pub fn new(
        old: &dyn EmbeddingLayerLike,
        embed_config: &LoraEmbeddingConfig,
        config: &LoraConfig,
        vb: &VarBuilder,
        id: usize,
    ) -> Result<Self> {
        // Try HF format first (lora_A, lora_B), fallback to candle format (a{id}, b{id})
        let a = vb
            .pp("lora_A")
            .get_with_hints(
                (config.rank, embed_config.num_embeddings),
                "weight",
                init::ZERO,
            )
            .or_else(|_| {
                // Fallback to candle-lora format
                vb.pp(format!("a{id}")).get_with_hints(
                    (config.rank, embed_config.num_embeddings),
                    "weight",
                    init::ZERO,
                )
            })?;

        let b: Tensor = vb
            .pp("lora_B")
            .get_with_hints(
                (embed_config.embedding_dim, config.rank),
                "weight",
                Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            )
            .or_else(|_| {
                // Fallback to candle-lora format
                vb.pp(format!("b{id}")).get_with_hints(
                    (embed_config.embedding_dim, config.rank),
                    "weight",
                    Init::Randn {
                        mean: 0.0,
                        stdev: 1.0,
                    },
                )
            })?;

        // Try to load magnitude vector for DoRA
        // HF format: lora_magnitude_vector, candle format: magnitude{id}
        // For embeddings, magnitude is per embedding (num_embeddings)
        let magnitude = vb
            .pp("lora_magnitude_vector")
            .get((embed_config.num_embeddings,), "weight")
            .or_else(|_| {
                vb.pp(format!("magnitude{id}"))
                    .get((embed_config.num_embeddings,), "weight")
            })
            .ok(); // Use .ok() to make it optional

        let mut a_t = a.t()?;
        a_t = a_t.reshape(a_t.shape())?;
        let embed_a = Embedding::new(a_t.clone(), a_t.dim(1)?);

        Ok(LoraEmbedding {
            old: Arc::new(FrozenEmbedding::new_from_embed(old)?),
            embed_a,
            a,
            b,
            scale: if config.rank > 0 {
                Some(config.alpha / config.rank as f64)
            } else {
                None
            },
            merged: false,
            magnitude,
            prefix: vb.prefix(),
            id,
        })
    }
}

impl Merge for LoraEmbedding {
    fn get_delta_weight(&self) -> std::result::Result<Tensor, MergeErrorOrError> {
        let ba = self.b.matmul(&self.a).map_err(Either::Right)?;
        let scaled_ba = match self.scale {
            Some(scale) => ba.mul(scale).map_err(Either::Right)?,
            None => ba,
        };

        // If magnitude vector exists, this is DoRA - apply normalization
        if let Some(ref magnitude) = self.magnitude {
            // For embeddings: transpose to match embedding format
            let scaled_ba_t = scaled_ba.transpose(0, 1).map_err(Either::Right)?;
            let combined = (self.embeddings() + &scaled_ba_t).map_err(Either::Right)?;

            // Compute row-wise L2 norm for embeddings
            let norm = combined
                .sqr()
                .map_err(Either::Right)?
                .sum_keepdim(1)
                .map_err(Either::Right)?
                .sqrt()
                .map_err(Either::Right)?;
            let norm = (norm + 1e-8).map_err(Either::Right)?;

            // Normalize row-wise
            let normalized = combined.broadcast_div(&norm).map_err(Either::Right)?;

            // Apply magnitude scaling
            let mag_reshaped = magnitude.unsqueeze(1).map_err(Either::Right)?;
            let scaled = normalized
                .broadcast_mul(&mag_reshaped)
                .map_err(Either::Right)?;

            // Return delta (need to transpose back)
            (scaled - self.embeddings())
                .map_err(Either::Right)?
                .transpose(0, 1)
                .map_err(Either::Right)
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
                FrozenEmbedding::new(
                    &(self.embeddings() + self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
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
                FrozenEmbedding::new(
                    &(self.embeddings() - self.get_delta_weight()?.transpose(0, 1))
                        .map_err(Either::Right)?,
                    self.hidden_size(),
                )
                .map_err(Either::Right)?,
            );
            self.merged = false;
            Ok(())
        }
    }
}

impl Module for LoraEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.merged {
            self.old.forward(input)
        } else if let Some(ref magnitude) = self.magnitude {
            // DoRA forward pass
            let base_embed = self.old.forward(input)?;

            if let Some(scale) = self.scale {
                let b = self.b.t()?;
                let b = b.reshape(b.shape())?;
                let after_a = self.embed_a.forward(input)?;
                let lora_embed = after_a.broadcast_matmul(&b)?.mul(scale)?;

                // Combine base and LoRA embeddings
                let combined = (base_embed + lora_embed)?;

                // Normalize row-wise
                let norm = combined.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
                let norm = (norm + 1e-8)?;
                let normalized = combined.broadcast_div(&norm)?;

                // Apply magnitude scaling
                // Flatten input to 1D for index selection, then reshape back
                let input_shape = input.dims();
                let input_flat = input.flatten_all()?;
                let gathered_mag = magnitude.index_select(&input_flat, 0)?;

                // Reshape gathered magnitude to match input shape, then add embedding dimension
                let gathered_mag = gathered_mag.reshape(input_shape)?;
                let mag_expanded = gathered_mag.unsqueeze(D::Minus1)?;
                normalized.broadcast_mul(&mag_expanded)
            } else {
                Ok(base_embed)
            }
        } else {
            // Standard LoRA forward pass
            let mut result = self.old.forward(input)?;
            if let Some(scale) = self.scale {
                let b = self.b.t()?;
                let b = b.reshape(b.shape())?;

                let after_a = self.embed_a.forward(input)?;
                result = (result + (after_a.broadcast_matmul(&b)?).mul(scale))?
            }
            Ok(result)
        }
    }
}

impl Saveable for LoraEmbedding {
    fn get_tensors(&self, accum: &mut HashMap<String, Tensor>) {
        accum.insert(
            self.prefix.clone() + &format!(".a{}.weight", self.id),
            self.a.clone(),
        );
        accum.insert(
            self.prefix.clone() + &format!(".b{}.weight", self.id),
            self.b.clone(),
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

impl EmbeddingLayerLike for LoraEmbedding {
    fn embeddings(&self) -> &Tensor {
        self.old.embeddings()
    }
    fn hidden_size(&self) -> usize {
        self.old.hidden_size()
    }
}
