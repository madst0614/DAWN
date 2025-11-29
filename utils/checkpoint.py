"""
DAWN Checkpoint Utilities

Version-aware checkpoint loading with automatic migration support.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


# =============================================================================
# Version Parameter Registry
# =============================================================================
# Defines what parameters each version adds/removes
# Key: version string, Value: dict with 'added', 'removed', 'renamed' lists

VERSION_PARAM_CHANGES = {
    "9.0": {
        "description": "CompressNeurons + ExpandNeurons + ReflectionNeurons",
        "added": [
            "shared_neurons.compress_neurons",
            "shared_neurons.expand_neurons",
            "shared_neurons.reflect_d",
            "shared_neurons.reflect_r",
            "router_compress", "router_expand",
            "router_d", "router_r",
        ],
        "removed": [
            # v8.x parameters not in v9.0
            "shared_neurons.input_neuron_",
            "shared_neurons.output_neuron",
            "shared_neurons.process_",
            "base_input", "base_output",
        ],
        "renamed": {
            # old_name -> new_name (if any direct mappings exist)
        },
    },
    "8.0": {
        "description": "SharedNeurons + NeuronMemory",
        "added": [
            "shared_neurons.input_neuron_",
            "shared_neurons.output_neuron",
            "shared_neurons.process_",
            "shared_neurons.knowledge_K",
            "shared_neurons.knowledge_V",
        ],
        "removed": [
            "circuit.input_",
            "circuit.process_",
            "circuit.output_",
        ],
        "renamed": {},
    },
}


def strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove _orig_mod. prefix from torch.compile() wrapped state dicts.

    Args:
        state_dict: Model state dictionary

    Returns:
        State dict with prefixes stripped
    """
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    return state_dict


def categorize_keys(
    missing_keys: List[str],
    unexpected_keys: List[str],
    source_version: str,
    target_version: str
) -> Dict[str, Dict[str, List[str]]]:
    """
    Categorize missing and unexpected keys by version changes.

    Args:
        missing_keys: Keys in model but not in checkpoint
        unexpected_keys: Keys in checkpoint but not in model
        source_version: Version of checkpoint
        target_version: Version of model

    Returns:
        Categorized keys dict
    """
    result = {
        'missing': {
            'new_in_target': [],  # Expected - new parameters in target version
            'other': [],
        },
        'unexpected': {
            'removed_from_source': [],  # Expected - old parameters from source
            'other': [],
        }
    }

    # Get target version's new parameters
    target_info = VERSION_PARAM_CHANGES.get(target_version, {})
    target_added = target_info.get('added', [])

    # Get source version's parameters that were removed
    source_info = VERSION_PARAM_CHANGES.get(source_version, {})
    source_removed_in_next = []

    # Find parameters that source has but target doesn't
    for version, info in VERSION_PARAM_CHANGES.items():
        if info.get('removed'):
            source_removed_in_next.extend(info['removed'])

    # Categorize missing keys
    for key in missing_keys:
        is_new = any(pattern in key for pattern in target_added)
        if is_new:
            result['missing']['new_in_target'].append(key)
        else:
            result['missing']['other'].append(key)

    # Categorize unexpected keys
    for key in unexpected_keys:
        is_old = any(pattern in key for pattern in source_removed_in_next)
        if is_old:
            result['unexpected']['removed_from_source'].append(key)
        else:
            result['unexpected']['other'].append(key)

    return result


def load_checkpoint_smart(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = None,
    verbose: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Smart checkpoint loading with version awareness.

    Args:
        model: Target model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint to
        strict: If None, auto-detect based on version match
        verbose: Print loading information

    Returns:
        Tuple of (checkpoint_dict, load_info)
        load_info contains: version_match, missing_keys, unexpected_keys, categorized
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get versions
    checkpoint_version = checkpoint.get('model_version', 'unknown')
    model_version = getattr(model, '__version__', 'unknown')

    version_match = (checkpoint_version == model_version)

    # Auto-determine strict mode
    if strict is None:
        strict = version_match

    # Get and clean state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = strip_compile_prefix(state_dict)

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    # Categorize keys if there are mismatches
    categorized = None
    if missing_keys or unexpected_keys:
        categorized = categorize_keys(
            missing_keys, unexpected_keys,
            checkpoint_version, model_version
        )

    # Build load info
    load_info = {
        'version_match': version_match,
        'checkpoint_version': checkpoint_version,
        'model_version': model_version,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'categorized': categorized,
        'strict': strict,
    }

    # Print info if verbose
    if verbose:
        print_load_info(load_info)

    return checkpoint, load_info


def print_load_info(load_info: Dict[str, Any]) -> None:
    """Print formatted loading information."""
    version_match = load_info['version_match']
    ckpt_v = load_info['checkpoint_version']
    model_v = load_info['model_version']

    if version_match:
        print(f"✅ Version match: v{model_v}")
    else:
        print(f"🔄 Cross-version loading: v{ckpt_v} → v{model_v}")

    categorized = load_info.get('categorized')
    if not categorized:
        print("✅ All parameters loaded successfully!")
        return

    # New parameters in target (expected when upgrading)
    new_params = categorized['missing']['new_in_target']
    if new_params:
        print(f"\n✨ New parameters (v{model_v}, randomly initialized): {len(new_params)}")
        if len(new_params) <= 5:
            for k in new_params:
                print(f"   + {k}")
        else:
            print(f"   (showing first 5 of {len(new_params)})")
            for k in new_params[:5]:
                print(f"   + {k}")

    # Deprecated parameters from source (expected when upgrading)
    old_params = categorized['unexpected']['removed_from_source']
    if old_params:
        print(f"\n♻️  Deprecated parameters (v{ckpt_v}, ignored): {len(old_params)}")
        if len(old_params) <= 5:
            for k in old_params:
                print(f"   - {k}")
        else:
            print(f"   (showing first 5 of {len(old_params)})")
            for k in old_params[:5]:
                print(f"   - {k}")

    # Other missing (might be a problem)
    other_missing = categorized['missing']['other']
    if other_missing:
        print(f"\n⚠️  Other missing keys: {len(other_missing)}")
        for k in other_missing[:5]:
            print(f"   ? {k}")

    # Other unexpected (might be a problem)
    other_unexpected = categorized['unexpected']['other']
    if other_unexpected:
        print(f"\n⚠️  Other unexpected keys: {len(other_unexpected)}")
        for k in other_unexpected[:5]:
            print(f"   ? {k}")


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: Dict[str, Any],
    scheduler=None,
    scaler=None,
    verbose: bool = True
) -> Tuple[int, float]:
    """
    Load optimizer, scheduler, and scaler states from checkpoint.

    Args:
        optimizer: Optimizer to load state into
        checkpoint: Checkpoint dictionary
        scheduler: Optional scheduler
        scaler: Optional AMP scaler
        verbose: Print loading information

    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

        if verbose:
            print(f"✅ Optimizer/scheduler loaded successfully")
            print(f"✅ Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

        return start_epoch, best_val_loss

    except Exception as e:
        if verbose:
            print(f"\n⚠️  Could not load optimizer state: {str(e)[:100]}")
            print(f"   Starting with fresh optimizer (model weights preserved)")

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

        if verbose:
            print(f"   Epoch count: {start_epoch}, Best val loss: {best_val_loss:.4f}")

        return start_epoch, best_val_loss
