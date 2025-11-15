# SPROUT

**Self-organizing Progressive Routing with Organic Unified Trees**

A dynamic neural network architecture that grows tree-like knowledge structures based on input compatibility.

## Core Concepts

- **Dynamic Knowledge Structure**: Tree formation through adaptive branching
- **Compatibility-Based Routing**: Attention + gating mechanism for path selection
- **Recursive Node Architecture**: Attention → FFN → Router flow at each node
- **Single Root Entry Point**: Graph-like expansion from unified starting point
- **Convergence Tracking**: Monitor structure formation and stabilization

## Architecture

Each node in SPROUT follows a recursive structure:

```
Input → Attention (gather) → FFN (process) → Router (route)
                                               ↓
                                    [Child Nodes or New Branch]
```

### Key Components

1. **Router**: Computes compatibility between input and child nodes using attention + gating
2. **Node**: Processes input and routes to children based on compatibility
3. **SPROUT**: Main model with single root and convergence tracking

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Language Model Training (MLM)

Train SPROUT on Masked Language Modeling:

```bash
# Quick test
python scripts/train_sprout_mlm.py --debug_mode --num_epochs 1

# Full training
python scripts/train_sprout_mlm.py \
  --checkpoint_dir ./checkpoints \
  --num_epochs 3 \
  --batch_size 32 \
  --max_nodes 5 \
  --visualize_structure
```

For Google Colab training, see [COLAB_TRAINING.md](COLAB_TRAINING.md)

### Core SPROUT Model

```python
import torch
from sprout import SPROUT

# Initialize model
model = SPROUT(
    dim=512,                        # Hidden dimension
    max_depth=5,                    # Maximum tree depth
    compatibility_threshold=0.8,    # Branching threshold
    num_heads=4,                    # Attention heads
    ffn_mult=4                      # FFN expansion factor
)

# Forward pass
x = torch.randn(2, 10, 512)  # [batch, seq, dim]
output, path_log = model(x)

# Visualize structure
model.visualize_structure()

# Get statistics
stats = model.get_statistics()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Nodes by depth: {stats['nodes_by_depth']}")

# Check convergence
if model.is_converged():
    print("Structure has converged!")
```

## Features

### Dynamic Branching

SPROUT automatically creates new branches when input compatibility falls below threshold:

```python
# Low compatibility → new branch created
if best_compatibility < compatibility_threshold:
    new_child = create_child()
    route_to(new_child)
# High compatibility → route to existing child
else:
    route_to(best_matching_child)
```

### Context-Aware Routing

Support for peer context bias:

```python
output, path_log = model(x, context_bias=peer_context)
```

### Path Logging

Detailed routing decisions:

```python
for log in path_log:
    print(f"Node {log['node_id']}: {log['action']}")
    print(f"  Compatibility: {log.get('compatibility', 'N/A')}")
```

### Structure Visualization

```python
model.visualize_structure(max_depth=3)
```

Output:
```
=== SPROUT Structure ===
└── Node 0 (depth=0, children=2, usage=100)
    ├── Node 0 (depth=1, children=3, usage=45)
    │   ├── Node 0 (depth=2, children=1, usage=15)
    │   ├── Node 1 (depth=2, children=0, usage=12)
    │   └── Node 2 (depth=2, children=2, usage=18)
    └── Node 1 (depth=1, children=2, usage=55)
        ├── Node 0 (depth=2, children=1, usage=20)
        └── Node 1 (depth=2, children=1, usage=35)
Total nodes: 9
```

## Configuration

### Model Parameters

- `dim`: Hidden dimension (default: 512)
- `max_depth`: Maximum tree depth (default: 5)
- `compatibility_threshold`: Threshold for creating new branches (default: 0.8)
- `num_heads`: Number of attention heads (default: 4)
- `ffn_mult`: FFN expansion multiplier (default: 4)

### Convergence Parameters

```python
# Check if structure has stabilized
is_stable = model.is_converged(
    window=100,      # Check last 100 iterations
    threshold=0.001  # Max 0.1% branch rate
)
```

## Examples

See `examples/` directory for:

- `basic_usage.py`: Simple usage example
- `convergence_tracking.py`: Monitor structure convergence
- `context_routing.py`: Using context bias for routing

## Testing

```bash
python -m pytest tests/
```

## Architecture Details

### Compatibility Computation

```python
# Attention-based compatibility
attn_out = attention(input, node_key, node_key)
compatibility = sigmoid(gate(attn_out))
```

### Soft Gating

```python
# Blend based on compatibility
output = compatibility * child_output + (1 - compatibility) * input
```

### Node Creation

New nodes are created when:
1. A node has no children (first child creation)
2. Best compatibility < threshold (branching)

## License

MIT License - see LICENSE file for details

## Citation

If you use SPROUT in your research, please cite:

```bibtex
@software{sprout2025,
  title={SPROUT: Self-organizing Progressive Routing with Organic Unified Trees},
  author={SPROUT Team},
  year={2025},
  url={https://github.com/yourusername/sprout}
}
```

## Contributing

Contributions welcome! Please open an issue or PR.
