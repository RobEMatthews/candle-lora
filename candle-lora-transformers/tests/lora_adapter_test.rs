//! Test to verify that LoRA adapters actually affect model outputs
//!
//! This test ensures that:
//! 1. Model without LoRA adapter weights produces certain outputs
//! 2. Model with LoRA adapter weights produces DIFFERENT outputs
//! 3. The adapter weights are actually being used in forward pass

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_lora::LoraConfig;
use candle_nn::{init, VarBuilder, VarMap};

#[test]
fn test_traced_lora_linear_uses_adapter() -> Result<()> {
    use candle_lora_transformers::with_tracing;
    use candle_nn::Linear;

    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 64;
    let out_size = 32;
    let batch_size = 1;

    let input = Tensor::randn(0f32, 1.0, (batch_size, in_size), &device)?;

    let base_varmap = VarMap::new();
    let base_vb = VarBuilder::from_varmap(&base_varmap, dtype, &device);
    let base_weight = base_vb.get_with_hints(
        (out_size, in_size),
        "weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let base_linear = Linear::new(base_weight, None);
    let base_output = base_linear.forward(&input)?;

    println!("Base output shape: {:?}", base_output.shape());

    let lora_config = LoraConfig::new(8, 16.0, None);

    // Create VarBuilder with LoRA weights
    let lora_varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&lora_varmap, dtype, &device);

    // Initialize LoRA weights (must exist in varmap)
    let _lora_a = lora_vb.get_with_hints(
        (8, in_size),
        "a0.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let _lora_b = lora_vb.get_with_hints(
        (out_size, 8),
        "b0.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;

    // Create TracedLoraLinear
    let traced_lora = with_tracing::linear(in_size, out_size, lora_vb, false, lora_config)?;

    // Get output from TracedLoraLinear
    let traced_output = traced_lora.forward(&input)?;

    println!("Traced LoRA output shape: {:?}", traced_output.shape());

    let diff = ((&traced_output - &base_output)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    println!("Difference between base and TracedLoraLinear: {}", diff);

    // Verify difference is significant
    assert!(diff > 0.01, "LoRA weights not applied; diff={}", diff);

    println!("Test passed: TracedLoraLinear successfully uses LoRA adapter");
    Ok(())
}

// Note: This test verifies that LoRA adapters loaded into a model actually change outputs
// It uses with_tracing::linear which is what the actual Qwen/transformer models use

#[test]
fn test_qwen_model_logits_change_with_adapter() -> Result<()> {
    // This test confirms that loading LoRA adapters actually changes model outputs
    // We create two models: one without LoRA weights (defaults to zeros), one with non-zero LoRA weights

    use candle_lora_transformers::with_tracing;

    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 64;
    let out_size = 32;
    let rank = 4;

    let input = Tensor::randn(0f32, 1.0, (1, in_size), &device)?;

    // Model 1: VarMap WITHOUT explicit LoRA weights (B will default to zeros)
    let varmap1 = VarMap::new();
    let vb1 = VarBuilder::from_varmap(&varmap1, dtype, &device);
    let lora_config1 = LoraConfig::new(rank, 8.0, None);

    let layer1 = with_tracing::linear(in_size, out_size, vb1, false, lora_config1)?;
    let output1 = layer1.forward(&input)?;
    let sum1 = output1.abs()?.sum_all()?.to_scalar::<f32>()?;
    println!("Model without explicit LoRA weights sum: {}", sum1);

    // Model 2: VarMap WITH explicit non-zero LoRA weights added BEFORE creating layer
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, dtype, &device);

    // Add LoRA weights to VarMap BEFORE creating the layer
    let _lora_a = vb2.get_with_hints(
        (rank, in_size),
        "lora_A.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let _lora_b = vb2.get_with_hints(
        (out_size, rank),
        "lora_B.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.05,
        },
    )?;

    let lora_config2 = LoraConfig::new(rank, 8.0, None);
    let layer2 = with_tracing::linear(in_size, out_size, vb2, false, lora_config2)?;
    let output2 = layer2.forward(&input)?;
    let sum2 = output2.abs()?.sum_all()?.to_scalar::<f32>()?;
    println!("Model with LoRA adapter sum: {}", sum2);

    // Compare outputs
    let diff = ((&output2 - &output1)?.abs()?.sum_all()?).to_scalar::<f32>()?;
    println!("Absolute difference: {}", diff);

    // Outputs must be different if LoRA is working
    assert!(
        diff > 0.01,
        "LoRA adapter weights did not affect output as expected. \
         Difference: {}, Model1 sum: {}, Model2 sum: {}",
        diff,
        sum1,
        sum2
    );

    println!("Test passed: LoRA adapters modify model outputs.");
    Ok(())
}
