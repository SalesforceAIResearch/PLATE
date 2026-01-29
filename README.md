# PLATE: Plasticity-Tunable Efficient Adapters

**PLATE** (Plasticity-Tunable Efficient Adapters) is a parameter-efficient fine-tuning method that reduces catastrophic forgetting in continual learning by combining neuron selection with orthogonal input basis computation.

## Installation

```bash
pip install -e .
```

## Requirements

See `requirements.txt` for full dependencies. Minimum requirements:
- Python >= 3.8
- PyTorch >= 2.0.0
- peft >= 0.15.0
- transformers >= 4.40.0

## Quick Start

```python
from transformers import AutoModelForCausalLM
from plate import PLATEConfig, get_plate_model

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")

# Configure PLATE (only one hyperparameter needed!)
config = PLATEConfig(
    r=64,                    # Number of trainable neurons
    col_tau=0.9,  # Input orthogonality threshold
    plate_alpha=1.0,  # Scaling factor
    max_rank =512, # Maximum input basis dimension
    plate_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply PLATE adapter
model = get_plate_model(model, config)

# Train as usual
model.print_trainable_parameters()
```

## Key Hyperparameter

- **`r`**: Number of trainable neurons per layer 
  - Higher `r` = more capacity but more forgetting

Other hyperparameters use sensible defaults:
- `col_tau=0.9`: Input orthogonality threshold
- `max_rank=512`: Maximum input basis dimension

## Example

See `examples/run_example.py` for a complete training example.

## Results

### Extended PLATE Parameter Sweep

**Qwen 2.5-3B - Extended PLATE sweep across (r,τ):** Columns fix the PLATE output rank r ∈ {32,64,128,256} and sweep τ ∈ {0.70,0.80,0.90,0.98} (green, solid) against LoRA baselines with varying ranks (blue, dashed). Top row reports WikiText-2 perplexity (forgetting) and bottom row reports Middle English perplexity (task learning), both over training steps.

![Extended PLATE Parameter Sweep](figures/forgetting_param_sweep_per_rank_new.png)

### Local-Geometry View of Forgetting

**Local-geometry view of forgetting - Extended PLATE sweep across (r,τ):** We sweep PLATE's two knobs on a two-moons continual-learning toy: increasing r expands the plasticity budget and improves task 2 performance but can increase task 1 drift/forgetting, while increasing τ tends to concentrate updates onto more redundant degrees of freedom and reduces drift/forgetting. Overall, PLATE provides an explicit mechanism to target a desired point on the retention-adaptation trade-off.

![Local-Geometry View of Forgetting](figures/plate_parameter_sweep.png)
