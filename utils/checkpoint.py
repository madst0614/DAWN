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
    "9.1": {
        "description": "v9.0 + hard selection + gated reflection + separate reflect pools",
        "added": [
            # v9.1: separate n_reflect_d / n_reflect_r (instead of shared n_reflect)
            "shared_neurons.reflect_d",  # [n_reflect_d, d_model]
            "shared_neurons.reflect_r",  # [n_reflect_r, rank]
        ],
        "removed": [],
        "renamed": {},
        "notes": "v9.1 separates reflect_d/reflect_r pool sizes (n_reflect_d, n_reflect_r)",
    },
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

    # Handle excitability_weight migration between versions
    scalar_key = 'global_routers.neuron_router.excitability_weight'
    v16_vector_keys = [
        'global_routers.neuron_router.excitability_weight_fr',
        'global_routers.neuron_router.excitability_weight_fv',
        'global_routers.neuron_router.excitability_weight_r',
        'global_routers.neuron_router.excitability_weight_v',
        'global_routers.neuron_router.excitability_weight_k',
    ]

    # Check checkpoint format
    ckpt_has_scalar = scalar_key in state_dict
    ckpt_has_vectors = any(k in state_dict for k in v16_vector_keys)

    # Check model format (v16.3 uses scalar, v16.0/16.1/16.2 use vectors)
    model_state_keys = set(model.state_dict().keys())
    model_expects_scalar = scalar_key in model_state_keys
    model_expects_vectors = any(k in model_state_keys for k in v16_vector_keys)

    if ckpt_has_vectors and model_expects_scalar and not model_expects_vectors:
        # v16.0/16.1/16.2 checkpoint → v16.3 model: vectors → scalar
        all_weights = []
        for k in v16_vector_keys:
            if k in state_dict:
                w = state_dict[k]
                if isinstance(w, torch.Tensor):
                    all_weights.append(w.mean().item())
                del state_dict[k]
        avg_w = sum(all_weights) / len(all_weights) if all_weights else 0.5
        state_dict[scalar_key] = torch.tensor(avg_w)
        print(f"  Note: Migrating per-neuron vectors → scalar excitability_weight={avg_w:.4f}")

    elif ckpt_has_scalar and model_expects_vectors and not model_expects_scalar:
        # Old scalar checkpoint → v16.0/16.1/16.2 model: scalar → vectors
        old_w = state_dict[scalar_key]
        if isinstance(old_w, torch.Tensor):
            old_w = old_w.item()
        old_w = max(0.1, min(0.5, old_w))

        config = checkpoint.get('config', {})
        n_fr = config.get('n_feature_r', 72)
        n_fv = config.get('n_feature_v', 72)
        n_r = config.get('n_relational', 196)
        n_v = config.get('n_value', 48)
        n_k = config.get('n_knowledge', 300)

        state_dict['global_routers.neuron_router.excitability_weight_fr'] = torch.full((n_fr,), old_w)
        state_dict['global_routers.neuron_router.excitability_weight_fv'] = torch.full((n_fv,), old_w)
        state_dict['global_routers.neuron_router.excitability_weight_r'] = torch.full((n_r,), old_w)
        state_dict['global_routers.neuron_router.excitability_weight_v'] = torch.full((n_v,), old_w)
        state_dict['global_routers.neuron_router.excitability_weight_k'] = torch.full((n_k,), old_w)
        del state_dict[scalar_key]
        print(f"  Note: Migrating scalar excitability_weight={old_w:.4f} → per-neuron vectors")

    # Remove deprecated excitability keys (v17.1+ no longer uses excitability)
    excitability_keys_to_remove = [k for k in state_dict.keys() if 'excitability' in k]
    if excitability_keys_to_remove:
        for k in excitability_keys_to_remove:
            del state_dict[k]
        print(f"  Note: Removed {len(excitability_keys_to_remove)} deprecated excitability keys")

    # v16.3 neuron parameter migration: separate → contiguous
    # Migrate fq_neurons, fk_neurons, fv_neurons → f_neurons
    # Migrate rq_neurons, rk_neurons, rv_neurons → r_neurons
    def migrate_neurons_to_contiguous(state_dict, layer_prefix):
        """Migrate separate neuron params to contiguous f_neurons/r_neurons"""
        fq_key = f'{layer_prefix}.shared_neurons.fq_neurons'
        fk_key = f'{layer_prefix}.shared_neurons.fk_neurons'
        fv_key = f'{layer_prefix}.shared_neurons.fv_neurons'
        rq_key = f'{layer_prefix}.shared_neurons.rq_neurons'
        rk_key = f'{layer_prefix}.shared_neurons.rk_neurons'
        rv_key = f'{layer_prefix}.shared_neurons.rv_neurons'
        f_key = f'{layer_prefix}.shared_neurons.f_neurons'
        r_key = f'{layer_prefix}.shared_neurons.r_neurons'

        migrated = False
        # Migrate F group
        if fq_key in state_dict and fk_key in state_dict and fv_key in state_dict:
            if f_key not in state_dict:
                f_neurons = torch.cat([state_dict[fq_key], state_dict[fk_key], state_dict[fv_key]], dim=0)
                state_dict[f_key] = f_neurons
                del state_dict[fq_key], state_dict[fk_key], state_dict[fv_key]
                migrated = True
        # Migrate R group
        if rq_key in state_dict and rk_key in state_dict and rv_key in state_dict:
            if r_key not in state_dict:
                r_neurons = torch.cat([state_dict[rq_key], state_dict[rk_key], state_dict[rv_key]], dim=0)
                state_dict[r_key] = r_neurons
                del state_dict[rq_key], state_dict[rk_key], state_dict[rv_key]
                migrated = True
        return migrated

    # Check if model uses contiguous neurons (v16.3+)
    model_uses_contiguous = any('shared_neurons.f_neurons' in k for k in model_state_keys)
    ckpt_has_separate = any('shared_neurons.fq_neurons' in k for k in state_dict.keys())

    if model_uses_contiguous and ckpt_has_separate:
        migrated = False
        for i in range(20):  # up to 20 layers
            if migrate_neurons_to_contiguous(state_dict, f'layers.{i}'):
                migrated = True
        if migrated:
            print("  Note: Migrated separate neuron params → contiguous f_neurons/r_neurons")

    # v16.2 proj migration: 5개 개별 proj → proj_all 통합
    if needs_v16_2_migration(state_dict):
        print("  Migrating v16.2 checkpoint: 5 proj → proj_all...")
        state_dict = migrate_v16_2_proj_to_unified(state_dict)

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
) -> Tuple[int, float, int]:
    """
    Load optimizer, scheduler, and scaler states from checkpoint.

    Args:
        optimizer: Optimizer to load state into
        checkpoint: Checkpoint dictionary
        scheduler: Optional scheduler
        scaler: Optional AMP scaler
        verbose: Print loading information

    Returns:
        Tuple of (start_epoch, best_val_loss, start_step)
        - start_epoch: epoch to resume from
        - best_val_loss: best validation loss so far
        - start_step: step within the epoch to resume from (0 if starting new epoch)
    """
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Check if this is an intermediate checkpoint (has 'step' but epoch not completed)
        saved_epoch = checkpoint.get('epoch', 0)
        saved_step = checkpoint.get('step', 0)
        epoch_completed = checkpoint.get('epoch_completed', True)  # Default True for backward compat

        if epoch_completed:
            # Epoch was completed, start next epoch from step 0
            start_epoch = saved_epoch + 1
            start_step = 0
        else:
            # Epoch was not completed, resume same epoch from next step
            start_epoch = saved_epoch
            start_step = saved_step + 1

        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

        if verbose:
            print(f"✅ Optimizer/scheduler loaded successfully")
            if start_step > 0:
                print(f"✅ Resuming from epoch {start_epoch}, step {start_step} (best val loss: {best_val_loss:.4f})")
            else:
                print(f"✅ Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

        return start_epoch, best_val_loss, start_step

    except Exception as e:
        if verbose:
            print(f"\n⚠️  Could not load optimizer state: {str(e)[:100]}")
            print(f"   Starting with fresh optimizer (model weights preserved)")

        saved_epoch = checkpoint.get('epoch', 0)
        saved_step = checkpoint.get('step', 0)
        epoch_completed = checkpoint.get('epoch_completed', True)

        if epoch_completed:
            start_epoch = saved_epoch + 1
            start_step = 0
        else:
            start_epoch = saved_epoch
            start_step = saved_step + 1

        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

        if verbose:
            print(f"   Epoch: {start_epoch}, Step: {start_step}, Best val loss: {best_val_loss:.4f}")

        return start_epoch, best_val_loss, start_step


# =============================================================================
# v16.2 Projection Migration
# =============================================================================

def migrate_v16_2_proj_to_unified(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    기존 v16.2 체크포인트의 개별 projection을 통합 proj_all로 변환.

    Before: proj, proj_FR_Q, proj_FR_K, proj_rel_Q, proj_rel_K (5개)
    After: proj_all (1개, d_space * 5)
    """
    new_state = {}
    migrated_routers = set()

    for key, value in state_dict.items():
        # neuron_router의 개별 proj 찾기
        if 'neuron_router.proj.weight' in key and 'proj_all' not in key:
            base = key.rsplit('neuron_router.proj.weight', 1)[0] + 'neuron_router.'

            # 이미 처리한 router는 스킵
            if base in migrated_routers:
                continue
            migrated_routers.add(base)

            # 5개 weight 가져오기
            proj_w = state_dict.get(base + 'proj.weight')
            proj_FR_Q_w = state_dict.get(base + 'proj_FR_Q.weight')
            proj_FR_K_w = state_dict.get(base + 'proj_FR_K.weight')
            proj_rel_Q_w = state_dict.get(base + 'proj_rel_Q.weight')
            proj_rel_K_w = state_dict.get(base + 'proj_rel_K.weight')

            if all(w is not None for w in [proj_w, proj_FR_Q_w, proj_FR_K_w, proj_rel_Q_w, proj_rel_K_w]):
                # [d_space, d_model] * 5 → [d_space*5, d_model]
                combined_w = torch.cat([proj_w, proj_FR_Q_w, proj_FR_K_w, proj_rel_Q_w, proj_rel_K_w], dim=0)
                new_state[base + 'proj_all.weight'] = combined_w

                # bias도 처리
                proj_b = state_dict.get(base + 'proj.bias')
                proj_FR_Q_b = state_dict.get(base + 'proj_FR_Q.bias')
                proj_FR_K_b = state_dict.get(base + 'proj_FR_K.bias')
                proj_rel_Q_b = state_dict.get(base + 'proj_rel_Q.bias')
                proj_rel_K_b = state_dict.get(base + 'proj_rel_K.bias')

                if all(b is not None for b in [proj_b, proj_FR_Q_b, proj_FR_K_b, proj_rel_Q_b, proj_rel_K_b]):
                    combined_b = torch.cat([proj_b, proj_FR_Q_b, proj_FR_K_b, proj_rel_Q_b, proj_rel_K_b], dim=0)
                    new_state[base + 'proj_all.bias'] = combined_b
            continue

        # 개별 proj 키들은 스킵 (이미 합침)
        if any(x in key for x in ['neuron_router.proj_FR_Q', 'neuron_router.proj_FR_K',
                                   'neuron_router.proj_rel_Q', 'neuron_router.proj_rel_K']):
            continue
        if 'neuron_router.proj.bias' in key and 'proj_all' not in key:
            continue

        # 나머지는 그대로
        new_state[key] = value

    return new_state


def needs_v16_2_migration(state_dict: Dict[str, torch.Tensor]) -> bool:
    """v16.2 proj migration이 필요한지 확인"""
    for key in state_dict.keys():
        if 'neuron_router.proj.weight' in key and 'proj_all' not in key:
            return True
    return False
