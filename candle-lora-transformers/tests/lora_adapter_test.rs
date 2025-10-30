//! Test to verify that LoRA adapters actually affect model outputs
//!
//! This test ensures that:
//! 1. Base model (without adapter) produces certain logits
//! 2. Base model + adapter produces DIFFERENT logits
//! 3. The adapter weights are actually being used in forward pass

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_lora::LoraLinear;
use candle_lora::{LoraConfig, LoraLinearConfig};
use candle_nn::{init, Linear, VarBuilder, VarMap};

#[test]
fn test_lora_adapter_changes_output() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 128;
    let out_size = 64;
    let batch_size = 2;

    // Create base linear layer with non-zero initialization
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // Initialize base weights with randn (not zeros)
    let base_weight = vb.get_with_hints(
        (out_size, in_size),
        "base.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let base_linear = Linear::new(base_weight.clone(), None);

    // Verify base weights are not all zero
    let weight_sum = base_weight.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(weight_sum > 0.1, "Base weights are all zeros!");

    // Create random input
    let input = Tensor::randn(0f32, 1.0, (batch_size, in_size), &device)?;

    // Get output from base model only
    let base_output = base_linear.forward(&input)?;

    println!("Base model output shape: {:?}", base_output.shape());
    println!("Base model output sample: {:?}", base_output.i((0, ..5))?);

    // Now create LoRA adapter with rank 16
    let lora_config = LoraConfig::new(16, 32.0, None);
    let linear_config = LoraLinearConfig::new(in_size, out_size);

    // Create VarBuilder for LoRA weights
    let lora_varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&lora_varmap, dtype, &device);

    // Initialize LoRA A and B weights (they need to exist in varmap)
    let _lora_a = lora_vb.get_with_hints(
        (16, in_size),
        "a0.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let _lora_b = lora_vb.get_with_hints((out_size, 16), "b0.weight", init::ZERO)?;

    // Create LoRA linear layer wrapping the base layer
    let lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &lora_vb, 0)?;

    // Get output from base + LoRA adapter
    let lora_output = lora_linear.forward(&input)?;

    println!("LoRA model output shape: {:?}", lora_output.shape());
    println!("LoRA model output sample: {:?}", lora_output.i((0, ..5))?);

    // Compute difference between outputs
    let diff = ((&lora_output - &base_output)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    println!("Total absolute difference: {}", diff);

    // The outputs MUST be different if LoRA is working
    assert!(
        diff > 1e-6,
        "LoRA adapter did not change the output! Base and LoRA outputs are identical. \
         This indicates that LoRA weights are not being applied in the forward pass."
    );

    // Verify that difference is significant (not just numerical noise)
    assert!(
        diff > 0.01,
        "LoRA adapter difference is too small ({}). \
         This might indicate LoRA weights are not being properly applied.",
        diff
    );

    println!("✅ Test passed: LoRA adapter successfully changes model output");
    Ok(())
}

#[test]
fn test_lora_adapter_with_merge() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 64;
    let out_size = 32;
    let batch_size = 1;

    // Create base linear layer with non-zero initialization
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let base_weight = vb.get_with_hints(
        (out_size, in_size),
        "base.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let base_linear = Linear::new(base_weight, None);

    // Create input
    let input = Tensor::randn(0f32, 1.0, (batch_size, in_size), &device)?;

    // Get base output
    let base_output = base_linear.forward(&input)?;

    // Create LoRA adapter (not merged)
    let lora_config = LoraConfig::new(8, 16.0, None);
    let linear_config = LoraLinearConfig::new(in_size, out_size);

    let lora_varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&lora_varmap, dtype, &device);

    // Initialize LoRA weights
    let _lora_a = lora_vb.get_with_hints(
        (8, in_size),
        "a0.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let _lora_b = lora_vb.get_with_hints((out_size, 8), "b0.weight", init::ZERO)?;

    let mut lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &lora_vb, 0)?;

    // Get output before merge
    let before_merge = lora_linear.forward(&input)?;

    // Merge the weights
    use candle_lora::Merge;
    lora_linear
        .merge_weights()
        .expect("Failed to merge weights");

    // Get output after merge
    let after_merge = lora_linear.forward(&input)?;

    // Before and after merge should give same results
    let merge_diff = ((&before_merge - &after_merge)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    println!("Difference before/after merge: {}", merge_diff);

    assert!(
        merge_diff < 1e-4,
        "Merged LoRA output differs from unmerged output! This indicates merge is broken."
    );

    // Both should differ from base
    let base_diff = ((&before_merge - &base_output)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    assert!(
        base_diff > 0.01,
        "LoRA output does not differ from base output. LoRA weights not applied."
    );

    println!("✅ Test passed: LoRA merge preserves output");
    Ok(())
}

#[test]
fn test_lora_unmerge() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 64;
    let out_size = 32;

    // Create base linear layer with non-zero initialization
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    let base_weight = vb.get_with_hints(
        (out_size, in_size),
        "base.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let base_linear = Linear::new(base_weight, None);

    let input = Tensor::randn(0f32, 1.0, (1, in_size), &device)?;

    // Create LoRA adapter
    let lora_config = LoraConfig::new(8, 16.0, None);
    let linear_config = LoraLinearConfig::new(in_size, out_size);

    let lora_varmap = VarMap::new();
    let lora_vb = VarBuilder::from_varmap(&lora_varmap, dtype, &device);

    // Initialize LoRA weights
    let _lora_a = lora_vb.get_with_hints(
        (8, in_size),
        "a0.weight",
        init::Init::Randn {
            mean: 0.0,
            stdev: 0.1,
        },
    )?;
    let _lora_b = lora_vb.get_with_hints((out_size, 8), "b0.weight", init::ZERO)?;

    let mut lora_linear = LoraLinear::new(&base_linear, &linear_config, &lora_config, &lora_vb, 0)?;

    // Get output before merge
    let before_merge = lora_linear.forward(&input)?;

    // Merge
    use candle_lora::Merge;
    lora_linear.merge_weights().expect("Failed to merge");

    // Unmerge
    lora_linear.unmerge_weights().expect("Failed to unmerge");

    // Get output after unmerge
    let after_unmerge = lora_linear.forward(&input)?;

    // Should be identical to before merge
    let diff = ((&before_merge - &after_unmerge)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    println!("Difference before merge vs after unmerge: {}", diff);

    assert!(
        diff < 1e-4,
        "Unmerge did not restore original behavior! Diff: {}",
        diff
    );

    println!("✅ Test passed: LoRA unmerge restores original output");
    Ok(())
}

#[test]
fn test_traced_lora_linear_uses_adapter() -> Result<()> {
    use candle_lora_transformers::with_tracing;

    let device = Device::Cpu;
    let dtype = DType::F32;

    let in_size = 64;
    let out_size = 32;
    let batch_size = 1;

    // Create input
    let input = Tensor::randn(0f32, 1.0, (batch_size, in_size), &device)?;

    // Create base model using plain linear
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

    // Create LoRA config
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

    // Compute difference
    let diff = ((&traced_output - &base_output)?.abs()?.sum_all()?).to_scalar::<f32>()?;

    println!("Difference between base and TracedLoraLinear: {}", diff);

    // The outputs MUST be different if LoRA is working
    assert!(
        diff > 1e-6,
        "TracedLoraLinear did not change the output! \
         This indicates that LoRA weights are not being applied. \
         Difference: {}",
        diff
    );

    // Verify difference is significant
    assert!(
        diff > 0.01,
        "TracedLoraLinear difference is too small ({}). \
         LoRA weights may not be properly applied.",
        diff
    );

    println!("✅ Test passed: TracedLoraLinear successfully uses LoRA adapter");
    Ok(())
}
