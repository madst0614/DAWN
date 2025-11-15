# SPROUT Training on Google Colab

Train SPROUT on Masked Language Modeling (MLM) with Google Colab.

## One-Line Quick Start 🚀

**Copy-paste this into a Colab cell:**

```python
# Clone and setup
!git clone https://github.com/madst0614/sprout.git 2>/dev/null || true
%cd /content/sprout
!git checkout claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4
!git pull origin claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4

# Install dependencies (takes ~30 seconds)
!pip install -q torch transformers datasets tqdm

# Quick test (debug mode - 1 minute)
!python scripts/train_sprout_mlm.py --debug_mode --num_epochs 1 --visualize_structure
```

## Full Training Setup

### 1. Mount Google Drive (for checkpoints)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone and Setup

```python
!git clone https://github.com/madst0614/sprout.git 2>/dev/null || true
%cd /content/sprout
!git checkout claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4
!git pull origin claude/implement-sprout-model-01L9icytevoJcfrrmKZo6Rm4
```

### 3. Install Dependencies

```python
!pip install -q torch transformers datasets tqdm
```

### 4. Run Training

```python
!python scripts/train_sprout_mlm.py \
  --checkpoint_dir /content/drive/MyDrive/sprout/checkpoints/ \
  --num_epochs 3 \
  --batch_size 32 \
  --max_samples 50000 \
  --max_nodes 5 \
  --visualize_structure
```

## Training Options

### Model Configuration

```bash
--hidden_dim 512              # Hidden dimension (default: 512)
--max_depth 2                 # Max tree depth (default: 2, ~5 nodes)
--max_nodes 5                 # Hard limit on nodes (default: 5)
--compatibility_threshold 0.8 # Branching threshold (default: 0.8)
--num_heads 4                 # Attention heads (default: 4)
```

### Training Configuration

```bash
--num_epochs 3          # Training epochs
--batch_size 32         # Batch size
--learning_rate 5e-5    # Learning rate
--warmup_steps 1000     # Warmup steps
--gradient_clip 1.0     # Gradient clipping
--mixed_precision       # Enable mixed precision (faster on GPU)
```

### Data Configuration

```bash
--max_samples 50000   # Maximum training samples
--max_length 128      # Maximum sequence length
--mask_prob 0.15      # MLM masking probability
```

### Debug Mode

For quick testing with tiny dataset:

```bash
--debug_mode
```

## Complete Colab Notebook Example

```python
# ============================================================================
# SPROUT MLM Training on Google Colab
# ============================================================================

# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo
!git clone https://github.com/madst0614/sprout.git
%cd /content/sprout

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Quick test with debug mode
print("=" * 70)
print("QUICK TEST (Debug Mode)")
print("=" * 70)

!python scripts/train_sprout_mlm.py \
  --debug_mode \
  --num_epochs 1 \
  --batch_size 8 \
  --visualize_structure

# 5. Full training
print("\n" + "=" * 70)
print("FULL TRAINING")
print("=" * 70)

!python scripts/train_sprout_mlm.py \
  --checkpoint_dir /content/drive/MyDrive/sprout/checkpoints/ \
  --num_epochs 3 \
  --batch_size 32 \
  --max_samples 50000 \
  --hidden_dim 512 \
  --max_depth 2 \
  --max_nodes 5 \
  --learning_rate 5e-5 \
  --mixed_precision \
  --visualize_structure

# 6. Check checkpoint
import torch
import os

checkpoint_dir = "/content/drive/MyDrive/sprout/checkpoints/"
checkpoint_path = os.path.join(checkpoint_dir, "sprout_mlm_best.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print(f"\n{'='*70}")
    print("CHECKPOINT INFO")
    print(f"{'='*70}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Loss: {checkpoint['loss']:.4f}")
    print(f"Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"Total nodes: {checkpoint['model_info']['total_nodes']}")
    print(f"Total params: {checkpoint['model_info']['total_params']:,}")
    print(f"{'='*70}\n")
else:
    print(f"Checkpoint not found: {checkpoint_path}")
```

## Advanced: Custom Training Loop

```python
import torch
from transformers import BertTokenizer
from sprout import SproutLanguageModel

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create model
model = SproutLanguageModel(
    vocab_size=len(tokenizer),
    hidden_dim=512,
    max_depth=2,
    max_nodes=5,
    compatibility_threshold=0.8
).to(device)

# Example forward pass
text = "The quick brown fox jumps over the lazy [MASK]."
encoding = tokenizer(text, return_tensors="pt")
input_ids = encoding["input_ids"].to(device)

# Forward
outputs = model(input_ids=input_ids)
logits = outputs["logits"]
path_log = outputs["path_log"]

print(f"Logits shape: {logits.shape}")
print(f"Routing decisions: {len(path_log)}")
print(f"Total nodes: {model.sprout.count_total_nodes()}")

# Visualize structure
model.visualize_structure()
```

## Monitoring Training

### Check GPU Usage

```python
!nvidia-smi
```

### Monitor Node Growth

The training script automatically tracks node growth. You'll see output like:

```
Epoch 1/3 Summary:
  Loss: 4.5234
  Accuracy: 23.45%
  Total nodes: 3/5
  Node limit reached: False

⚠️  Node limit reached: 5/5 nodes
```

### Structure Visualization

Use `--visualize_structure` flag to see the final tree structure:

```
=== SPROUT Structure ===
└── Node 0 (depth=0, children=2, usage=1500)
    ├── Node 0 (depth=1, children=2, usage=750)
    │   ├── Node 0 (depth=2, children=0, usage=400)
    │   └── Node 1 (depth=2, children=0, usage=350)
    └── Node 1 (depth=1, children=1, usage=750)
        └── Node 0 (depth=2, children=0, usage=750)
Total nodes: 5
```

## Tips

1. **Start with debug mode** to verify everything works
2. **Use mixed precision** (`--mixed_precision`) for faster training on GPU
3. **Save checkpoints to Drive** to avoid losing progress
4. **Monitor node limit** - training stops creating new nodes at max_nodes
5. **Adjust compatibility_threshold** (0.0-1.0) to control branching:
   - Higher (0.9-1.0): More aggressive branching
   - Lower (0.5-0.8): More conservative branching

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch_size 16
```

Or reduce model size:
```bash
--hidden_dim 256 --num_heads 2
```

### Training too slow

Enable mixed precision and reduce samples:
```bash
--mixed_precision --max_samples 10000
```

### Too many/few nodes

Adjust branching behavior:
```bash
--compatibility_threshold 0.9  # More branching
--compatibility_threshold 0.6  # Less branching
```

## Next Steps

After training:

1. **Analyze structure**: Use visualization to understand routing patterns
2. **Test convergence**: Check if structure stabilized
3. **Evaluate performance**: Compare with baseline models
4. **Experiment**: Try different configurations

For more details, see main [README.md](README.md)
