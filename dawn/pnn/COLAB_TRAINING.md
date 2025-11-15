# Google Colab Training Guide

## Setup (Run in first cell)

### Option 1: Safe Method (Recommended for Colab)

```python
# Clone repository
!git clone https://github.com/madst0614/plastic-neural-networks.git
%cd plastic-neural-networks

# Checkout experiment branch
!git checkout claude/refiner-experiments-ffn-011CUz2zZ6v1XqK5Fzwdy8eN

# Add to Python path (avoids package conflicts)
import sys
sys.path.insert(0, '/content/plastic-neural-networks')

# Install only missing packages
!pip install -q datasets accelerate

# Test import
from pnn.models.pnn import create_pnn_model
print("✅ Setup complete!")
```

### Option 2: Full Install (Requires runtime restart)

```bash
!git clone https://github.com/madst0614/plastic-neural-networks.git
%cd plastic-neural-networks
!git checkout claude/refiner-experiments-ffn-011CUz2zZ6v1XqK5Fzwdy8eN

# Install package
!pip install -e .

# ⚠️ IMPORTANT: Restart runtime after this
# Runtime → Restart runtime (from menu)
# Then run training commands in new cells
```

---

## Experiment 1: Dual Attention + Dual FFN (61M params)

**Structure**: Each refiner has 2 transformer blocks (Attention + FFN)

```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/exp1_dual_blocks \
  --use_amp \
  --use_tf32
```

**Expected size**: 55M → 61M (+5.55M)

---

## Experiment 2: Dual Refiners (68M params)

**Structure**: 2 independent refiners applied alternately

```bash
!python scripts/train.py \
  --model pnn_exp2 \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/exp2_dual_refiners \
  --use_amp \
  --use_tf32
```

**Expected size**: 55M → 68M (+13.4M)

---

## Baseline PNN (55M params)

**For comparison**

```bash
!python scripts/train.py \
  --model pnn \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/baseline_pnn \
  --use_amp \
  --use_tf32
```

---

## Monitor Training (Optional)

```python
# Load tensorboard (run in separate cell)
%load_ext tensorboard
%tensorboard --logdir checkpoints/
```

---

## Quick Test (Verify models work)

```python
import torch
from pnn.models.pnn import create_pnn_model

# Test all models
for model_type in ['pnn', 'pnn_exp1', 'pnn_exp2']:
    print(f"\n{'='*60}")
    model = create_pnn_model(model_type=model_type)

    # Quick forward pass
    input_ids = torch.randint(0, 30522, (2, 128))
    attention_mask = torch.ones(2, 128)
    output = model(input_ids, attention_mask)

    print(f"✓ {model_type.upper()} output shape: {output.shape}")
```

---

## Training Tips

### Batch Size Adjustment
If you get OOM (Out of Memory):
```bash
# Reduce batch size, increase gradient accumulation
--batch_size 128 \
--gradient_accumulation 8
```

### For Free Colab (T4 GPU)
```bash
# Conservative settings
--batch_size 128 \
--gradient_accumulation 8 \
--max_samples 100000
```

### For Colab Pro (A100 80GB) - RECOMMENDED

**Fast Training (~15-20 it/s)**
```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --batch_size 1024 \
  --gradient_accumulation 1 \
  --num_workers 12 \
  --max_samples 200000 \
  --epochs 15 \
  --lr 3e-4 \
  --use_amp \
  --use_tf32 \
  --checkpoint_dir checkpoints/exp1_fast

# Or use the pre-configured YAML
!python scripts/train.py --config configs/pnn_a100.yaml
```

**Expected Performance:**
- Speed: 15-20 iterations/second
- Training time: ~1-1.5 hours for 200K samples
- Memory usage: ~30-40GB / 80GB

---

## Expected Training Time

**On T4 GPU (Free Colab):**
- ~6-8 hours for 200K samples, 15 epochs

**On A100 GPU (Colab Pro):**
- ~2-3 hours for 200K samples, 15 epochs

---

## Resume Training

If training is interrupted:

```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --resume checkpoints/exp1_dual_blocks/checkpoint_epochX.pt \
  --epochs 15 \
  ...
```

---

## Compare Results

```python
import json

# Load metrics
with open('checkpoints/baseline_pnn/metrics.json') as f:
    baseline = json.load(f)

with open('checkpoints/exp1_dual_blocks/metrics.json') as f:
    exp1 = json.load(f)

with open('checkpoints/exp2_dual_refiners/metrics.json') as f:
    exp2 = json.load(f)

print(f"Baseline:  {baseline['best_accuracy']:.4f}")
print(f"Exp1:      {exp1['best_accuracy']:.4f}")
print(f"Exp2:      {exp2['best_accuracy']:.4f}")
```
