# Training Scripts

## DeepSets FFN Training

### Quick Start

```bash
# Baseline model
python scripts/train_deepsets.py --model baseline

# DeepSets-Basic
python scripts/train_deepsets.py --model deepsets-basic

# DeepSets-Context
python scripts/train_deepsets.py --model deepsets-context
```

### Custom Configuration

```bash
python scripts/train_deepsets.py \
  --model deepsets-basic \
  --d_model 512 \
  --d_ff 2048 \
  --n_layers 6 \
  --sparse_k 512 \
  --d_neuron 128 \
  --d_hidden 256 \
  --batch_size 32 \
  --num_epochs 30 \
  --lr 3e-4 \
  --use_amp  # Use mixed precision training
```

### Outputs

```
checkpoints/{model}/
├── config.json
├── best_checkpoint_epoch{N}_{timestamp}.pt
└── checkpoint_epoch{N}_{timestamp}.pt

logs/{model}/
└── training_log_{timestamp}.jsonl
```

### Model Parameters

| Model | Additional Params | Description |
|-------|-------------------|-------------|
| baseline | - | Standard neuron selection |
| deepsets-basic | `d_neuron`, `d_hidden` | Learnable neuron info vectors |
| deepsets-context | `d_neuron`, `d_hidden` | + Global context |

### Requirements

```bash
pip install torch transformers datasets tqdm
```

### Data

Uses WikiText-103 dataset (automatically downloaded via HuggingFace datasets)
