# DeepSets Experiments

DeepSets 기반 FFN 실험

## Main Training Script

Use the main training script in `scripts/train_deepsets.py`:

```bash
# Navigate to project root
cd /home/user/sprout

# Baseline
python scripts/train_deepsets.py --model baseline

# DeepSets-Basic  
python scripts/train_deepsets.py --model deepsets-basic

# DeepSets-Context
python scripts/train_deepsets.py --model deepsets-context --use_amp
```

See `scripts/README.md` for full documentation.
