# DAWN: Dynamic Architecture With Neurons

[![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://zenodo.org/records/17986495)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17986495.svg)](https://doi.org/10.5281/zenodo.17986495)

DAWN replaces static weight matrices with dynamic routing over shared neuron pools, achieving 4× better perplexity than vanilla Transformers with 4.5× fewer parameters.

## Key Results

| Model | Params | PPL ↓ | Acc ↑ |
|-------|--------|-------|-------|
| Vanilla-22M | 22.6M | 53.1 | 31.5% |
| Vanilla-108M | 108.9M | 32.7 | 36.6% |
| **DAWN-24M** | 23.9M | **8.4** | **55.4%** |

*Trained on C4 (5B tokens)*

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
    'knowledge_rank': 128,
    'max_seq_len': 512,
    'state_dim': 64,
    'n_feature_qk': 120,
    'n_feature_v': 24,
    'n_restore_qk': 120,
    'n_restore_v': 24,
    'n_feature_know': 24,
    'n_restore_know': 24,
    'top_k_feature_qk': 20,
    'top_k_feature_v': 6,
    'top_k_restore_qk': 20,
    'top_k_restore_v': 6,
    'top_k_feature_know': 4,
    'top_k_restore_know': 4,
    'd_space': 64,
    'dropout': 0.1,
    'router_dropout': 0.1,
}

model = DAWN(**config)
weights = torch.load('dawn_24m_weights.pt', map_location='cpu')
model.load_state_dict(weights)
model.eval()
```

### Training

```bash
python train.py
```

## Checkpoints

Download pretrained weights from [Zenodo](https://zenodo.org/records/17986495):

| Model | Params | PPL | Download |
|-------|--------|-----|----------|
| DAWN-24M | 23.9M | 8.4 | [weights](https://zenodo.org/records/17986495) |

## Citation

```bibtex
@misc{choi2025dawn,
  title={DAWN: Dynamic Architecture With Neurons},
  author={Choi, Seungho},
  year={2025},
  doi={10.5281/zenodo.17986495},
  url={https://zenodo.org/records/17986495}
}
```

## License

MIT
