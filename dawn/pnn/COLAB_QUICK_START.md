# 🚀 Colab Quick Start Guide

## 1. Setup (First Cell)

```python
# Clone and setup
!git clone https://github.com/madst0614/plastic-neural-networks.git
%cd plastic-neural-networks
!git checkout claude/refiner-experiments-ffn-011CUz2zZ6v1XqK5Fzwdy8eN

# Add to Python path (avoids package conflicts)
import sys
sys.path.insert(0, '/content/plastic-neural-networks')

# Install dependencies
!pip install -q datasets accelerate

# Verify setup
from pnn.models.pnn import create_pnn_model
print("✅ Setup complete!")
```

## 2. Mount Google Drive (Optional - for saving checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 3. Train Models

### 🔥 Experiment 1: Dual Transformer Blocks (RECOMMENDED)

**Structure:** Attention1 → FFN1 → Attention2 → FFN2 (61M params)

**For A100 80GB (Fast - 15-20 it/s):**
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
  --checkpoint_dir /content/drive/MyDrive/pnn/exp1_dual_blocks

# Or use pre-configured YAML
!python scripts/train.py --config configs/pnn_a100.yaml \
  --checkpoint_dir /content/drive/MyDrive/pnn/exp1_fast
```

**For T4 GPU (Standard Colab):**
```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --batch_size 128 \
  --gradient_accumulation 8 \
  --num_workers 4 \
  --max_samples 100000 \
  --epochs 10 \
  --lr 3e-4 \
  --use_amp \
  --checkpoint_dir /content/drive/MyDrive/pnn/exp1_t4
```

---

### 🔥 Experiment 2: Dual Refiners (Alternating)

**Structure:** Refiner1 ⇄ Refiner2 alternating (68M params)

**For A100 80GB:**
```bash
!python scripts/train.py \
  --model pnn_exp2 \
  --batch_size 1024 \
  --gradient_accumulation 1 \
  --num_workers 12 \
  --max_samples 200000 \
  --epochs 15 \
  --lr 3e-4 \
  --use_amp \
  --use_tf32 \
  --checkpoint_dir /content/drive/MyDrive/pnn/exp2_dual_refiners
```

**For T4 GPU:**
```bash
!python scripts/train.py \
  --model pnn_exp2 \
  --batch_size 128 \
  --gradient_accumulation 8 \
  --num_workers 4 \
  --max_samples 100000 \
  --epochs 10 \
  --checkpoint_dir /content/drive/MyDrive/pnn/exp2_t4
```

---

### 📊 Baseline PNN (For Comparison)

**For A100 80GB:**
```bash
!python scripts/train.py \
  --model pnn \
  --batch_size 1024 \
  --gradient_accumulation 1 \
  --num_workers 12 \
  --max_samples 200000 \
  --epochs 15 \
  --checkpoint_dir /content/drive/MyDrive/pnn/baseline
```

---

## 4. What You'll See During Training

### Progress Bar (Real-time)
```
Epoch 1/15: 100%|██████| 195/195 [00:12<00:00, 15.2it/s, loss=3.2456, acc=0.4123, lr=2.8e-04]
```

### Epoch Summary
```
Epoch 1/15:
Train Loss: 3.2456
Step Losses: ['4.1234', '3.5678', '3.2345', '3.1234']
Step Accs:   ['0.2345', '0.3456', '0.3890', '0.4123']  ← Step-wise improvement!
Eval Loss:  3.1234
Eval Acc:   0.4567 (45.67%)
Time: 12.3s (0.2m)
```

### For Exp2 (Dual Refiners) - Shows Each Refiner's Performance
```
Refiner1 (steps 0,2): Loss=3.5678, 3.2345 | Acc=0.3456, 0.3890
Refiner2 (steps 1,3): Loss=3.3456, 3.1234 | Acc=0.3678, 0.4123
```

---

## 5. Expected Performance

### A100 80GB
- **Speed:** 15-20 iterations/second
- **Training time:** 1-1.5 hours (200K samples, 15 epochs)
- **Memory:** 30-40GB / 80GB

### T4 GPU (Free Colab)
- **Speed:** 3-5 iterations/second
- **Training time:** 3-5 hours (100K samples, 10 epochs)
- **Memory:** 10-12GB / 16GB

---

## 6. Monitor Training (Optional)

```python
# Load tensorboard
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/pnn/
```

---

## 7. Resume Training (If Interrupted)

```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --resume /content/drive/MyDrive/pnn/exp1_dual_blocks/checkpoint_epoch5.pt \
  --batch_size 1024 \
  --gradient_accumulation 1 \
  --num_workers 12 \
  --epochs 15
```

---

## 8. Compare Results

```python
import json

# Load metrics
experiments = {
    'Baseline': '/content/drive/MyDrive/pnn/baseline/metrics.json',
    'Exp1 (Dual Blocks)': '/content/drive/MyDrive/pnn/exp1_dual_blocks/metrics.json',
    'Exp2 (Dual Refiners)': '/content/drive/MyDrive/pnn/exp2_dual_refiners/metrics.json'
}

print("\n" + "="*60)
print("📊 Experiment Results")
print("="*60 + "\n")

for name, path in experiments.items():
    try:
        with open(path) as f:
            metrics = json.load(f)
        print(f"{name:25s} Acc: {metrics['best_accuracy']:.4f} ({metrics['best_accuracy']*100:.2f}%)")
    except FileNotFoundError:
        print(f"{name:25s} Not trained yet")

print("\n" + "="*60 + "\n")
```

---

## 🎯 Quick Tips

1. **For fastest training:** Use A100 with `batch_size=1024`, `num_workers=12`
2. **To save checkpoints:** Mount Google Drive first
3. **If OOM error:** Reduce batch_size and increase gradient_accumulation
4. **Step accuracy shows:** How each refinement step improves predictions
5. **For Exp2:** Check if Refiner1 and Refiner2 learn different patterns

---

## 🐛 Troubleshooting

**Problem:** `RuntimeError: CUDA out of memory`
```bash
# Solution: Reduce batch size
--batch_size 512  # or 256
--gradient_accumulation 2  # or 4
```

**Problem:** Slow training on A100
```bash
# Solution: Increase batch size and workers
--batch_size 1024
--num_workers 12
--gradient_accumulation 1
```

**Problem:** Training interrupted
```bash
# Solution: Resume from checkpoint
--resume /path/to/checkpoint_epochX.pt
```
