# DAWN: Dynamic Subspace Composition for Structural Interpretability

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://zenodo.org/records/18060421)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18060421.svg)](https://doi.org/10.5281/zenodo.18060421)

DAWN replaces static weight matrices with dynamic subspace composition over shared neuron pools, enabling structural interpretability through traceable routing paths.

## Key Results

| Model | Params | FLOPs | PPL ↓ | Acc ↑ |
|-------|--------|-------|-------|-------|
| Vanilla-40M | 40.35M | 23.23G | 41.2 | 34.0% |
| **DAWN-39M** | 39.25M | 26.54G | 42.2 | 34.0% |

*Trained on C4 (5B tokens). DAWN achieves comparable performance while providing interpretable internal structure.*

## Key Findings

- **Emergent Q/K Specialization**: Query and Key neurons develop distinct roles (correlation r=-0.75) despite sharing a pool
- **POS-Aligned Neurons**: Neurons specialize for linguistic categories without explicit supervision
- **Structural Interpretability**: Computation is traceable through routing paths

## Architecture

![DAWN Architecture](assets/fig1_architecture.png)

## Installation

```bash
git clone https://github.com/madst0614/DAWN
cd DAWN
pip install -r requirements.txt
```

## Quick Start

### Inference

```python
import torch
from models import DAWN

config = {
    'vocab_size': 30522,
    'd_model': 384,
    'n_layers': 12,
    'n_heads': 6,
    'rank': 64,
    'knowledge_rank': 64,
    'max_seq_len': 512,
    'state_dim': 64,
    'n_feature_qk': 64,
    'n_feature_v': 264,
    'n_restore_qk': 64,
    'n_restore_v': 264,
    'n_feature_know': 160,
    'n_restore_know': 160,
    'top_k_feature_qk': 12,
    'top_k_feature_v': 12,
    'top_k_restore_qk': 12,
    'top_k_restore_v': 12,
    'top_k_feature_know': 12,
    'top_k_restore_know': 12,
    'd_space': 256,
    'dropout': 0.1,
    'router_dropout': 0.1,
}

model = DAWN(**config)
weights = torch.load('dawn_39m_weights.pt', map_location='cpu')
model.load_state_dict(weights)
model.eval()
```

### Training

```bash
python train.py
```

## Checkpoints

Download pretrained weights from [Zenodo](https://zenodo.org/records/18060421):

| Model | Params | PPL | Download |
|-------|--------|-----|----------|
| DAWN-39M | 39.25M | 42.2 | [weights](https://zenodo.org/records/18060421) |

## Citation

```bibtex
@misc{choi2025dawn,
  title={DAWN: Dynamic Subspace Composition for Structural Interpretability},
  author={Choi, Seungho},
  year={2025},
  doi={10.5281/zenodo.18060421},
  url={https://zenodo.org/records/18060421}
}
```

## License

MIT