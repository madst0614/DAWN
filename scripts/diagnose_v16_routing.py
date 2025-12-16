#!/usr/bin/env python3
"""
DAWN v16 Routing Diagnostics
=============================
체크포인트로 routing 상태 진단:
1. Usage EMA 분포
2. 실제 뉴런 선택 빈도
3. Importance 집중도 (entropy)

Usage:
    python diagnose_v16_routing.py --checkpoint path/to/checkpoint.pt --val_data path/to/val.pt
"""

import argparse
import sys
import os
import torch
import numpy as np
from collections import Counter, defaultdict

# POS tagging (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_checkpoint_path(path):
    """Resolve path to actual checkpoint file"""
    import glob
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        patterns = ["*.pt", "*.pth"]
        all_ckpts = []
        for pattern in patterns:
            all_ckpts.extend(glob.glob(os.path.join(path, pattern)))
            all_ckpts.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        if all_ckpts:
            for ckpt in all_ckpts:
                name = os.path.basename(ckpt).lower()
                if 'best' in name or 'final' in name:
                    return ckpt
            all_ckpts.sort(key=os.path.getmtime, reverse=True)
            return all_ckpts[0]
    raise FileNotFoundError(f"No checkpoint found: {path}")


def get_router(model):
    """Get neuron router (version-agnostic)"""
    if hasattr(model, 'router') and hasattr(model.router, 'neuron_router'):
        return model.router.neuron_router
    if hasattr(model, 'global_routers'):
        return model.global_routers.neuron_router
    return None


def detect_version(router):
    """Detect model version from router attributes"""
    if router is None:
        return 'unknown'
    if hasattr(router, 'usage_ema_feature_qk'):
        if hasattr(router, 'usage_ema_feature_know'):
            return '17.1'
        return '16.4'
    elif hasattr(router, 'usage_ema_feature_r'):
        return '16.0'
    return 'unknown'


def diagnose_usage_ema(model):
    """1. Usage EMA 분포 확인"""
    print(f"\n{'='*60}")
    print("1. Usage EMA Distribution")
    print(f"{'='*60}")

    results = {}

    router = get_router(model)
    if router is None:
        print("  No router found")
        return results

    version = detect_version(router)
    print(f"  Detected version: {version}")
    results['version'] = version

    # Version-specific EMA attributes
    if version in ['16.4', '17.1']:
        ema_list = [
            ('FQK', getattr(router, 'usage_ema_feature_qk', None)),
            ('FV', getattr(router, 'usage_ema_feature_v', None)),
            ('RQK', getattr(router, 'usage_ema_restore_qk', None)),
            ('RV', getattr(router, 'usage_ema_restore_v', None)),
        ]
        if version == '17.1':
            ema_list.extend([
                ('FK', getattr(router, 'usage_ema_feature_know', None)),
                ('RK', getattr(router, 'usage_ema_restore_know', None)),
            ])
    else:
        # v16.0/16.1
        ema_list = [
            ('FR', getattr(router, 'usage_ema_feature_r', None)),
            ('FV', getattr(router, 'usage_ema_feature_v', None)),
            ('R', getattr(router, 'usage_ema_relational', None)),
            ('V', getattr(router, 'usage_ema_value', None)),
        ]

    for name, ema in ema_list:
        if ema is None:
            print(f"  {name}: Not found")
            continue

        usage = ema.cpu().numpy()
        dead_count = int((usage < 0.01).sum())

        print(f"\n  {name} ({len(usage)} neurons):")
        print(f"    max={usage.max():.4f} (idx {usage.argmax()})")
        print(f"    min={usage.min():.4f} (idx {usage.argmin()})")
        print(f"    mean={usage.mean():.4f}, std={usage.std():.4f}")
        print(f"    dead (<0.01): {dead_count}/{len(usage)} ({100*dead_count/len(usage):.1f}%)")

        # Top 5 most used
        top5_idx = np.argsort(usage)[-5:][::-1]
        print(f"    Top 5: {[(int(i), f'{usage[i]:.4f}') for i in top5_idx]}")

        results[name] = {
            'n_neurons': len(usage),
            'max': float(usage.max()),
            'max_idx': int(usage.argmax()),
            'min': float(usage.min()),
            'mean': float(usage.mean()),
            'std': float(usage.std()),
            'dead_count': dead_count,
            'dead_pct': 100 * dead_count / len(usage),
            'top5': [(int(i), float(usage[i])) for i in top5_idx]
        }

    return results


def diagnose_neuron_selection(model, dataloader, device, config, max_batches=20):
    """2. 실제 뉴런 선택 빈도 확인"""
    print(f"\n{'='*60}")
    print("2. Neuron Selection Frequency (from routing)")
    print(f"{'='*60}")

    # Detect version
    router = get_router(model)
    version = detect_version(router)
    print(f"  Detected version: {version}")

    # Build topk_map and pref_key_map based on version
    if version in ['16.4', '17.1']:
        topk_map = {
            'FQK_Q': config.get('top_k_feature_qk', 8),
            'FQK_K': config.get('top_k_feature_qk', 8),
            'FV': config.get('top_k_feature_v', 6),
            'RQK_Q': config.get('top_k_restore_qk', 8),
            'RQK_K': config.get('top_k_restore_qk', 8),
            'RV': config.get('top_k_restore_v', 4),
        }
        pref_key_map = {
            'FQK_Q': 'fqk_q_pref',
            'FQK_K': 'fqk_k_pref',
            'FV': 'fv_pref',
            'RQK_Q': 'rqk_q_pref',
            'RQK_K': 'rqk_k_pref',
            'RV': 'rv_pref',
        }
    else:
        # v16.0/16.1
        topk_map = {
            'FR': config.get('top_k_feature_r', 8),
            'FV': config.get('top_k_feature_v', 6),
            'R': config.get('top_k_relational', 20),
            'V': config.get('top_k_value', 4),
        }
        pref_key_map = {
            'FR': 'feature_r_pref',
            'FV': 'feature_v_pref',
            'R': 'relational_q_pref',
            'V': 'value_pref',
        }

    print(f"  Using top-k from config: {topk_map}")

    neuron_counts = {nt: Counter() for nt in topk_map.keys()}
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape
            total_tokens += B * L

            # Forward with routing info
            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                print("  Model doesn't return routing_info")
                return

            routing_infos = output[-1]

            # Aggregate neuron selections across layers
            for layer_info in routing_infos:
                if 'attention' not in layer_info:
                    continue

                attn = layer_info['attention']

                # Get preferences and find top-k selections (version-aware)
                for nt, pref_key in pref_key_map.items():
                    pref = attn.get(pref_key)
                    if pref is None:
                        continue
                    k = min(topk_map.get(nt, 4), pref.shape[-1])
                    _, topk_idx = torch.topk(pref, k, dim=-1)

                    # Count selections (vectorized)
                    flat_idx = topk_idx.view(-1).cpu().numpy()
                    unique, counts = np.unique(flat_idx, return_counts=True)
                    for idx, cnt in zip(unique, counts):
                        neuron_counts[nt][int(idx)] += int(cnt)

    print(f"\n  Analyzed {max_batches} batches, {total_tokens} tokens")

    results = {'total_tokens': total_tokens, 'version': version}

    for nt, counts in neuron_counts.items():
        if not counts:
            print(f"\n  {nt}: No data")
            continue

        top10 = counts.most_common(10)
        total_selections = sum(counts.values())

        print(f"\n  {nt} - Top 10 neurons (total selections: {total_selections}):")
        for idx, count in top10:
            pct = 100 * count / total_selections
            print(f"    neuron {idx}: {count} ({pct:.1f}%)")

        # Coverage: how many neurons cover 50% of selections?
        sorted_counts = sorted(counts.values(), reverse=True)
        cumsum = np.cumsum(sorted_counts)
        n_for_50 = int(np.searchsorted(cumsum, total_selections * 0.5) + 1)
        n_for_90 = int(np.searchsorted(cumsum, total_selections * 0.9) + 1)
        print(f"    Coverage: {n_for_50} neurons for 50%, {n_for_90} neurons for 90%")

        results[nt] = {
            'total_selections': total_selections,
            'top10': [(int(idx), int(cnt)) for idx, cnt in top10],
            'n_for_50_coverage': n_for_50,
            'n_for_90_coverage': n_for_90,
            'n_active_neurons': len(counts)
        }

    return results


def diagnose_importance_entropy(model, dataloader, tokenizer, device, max_batches=10):
    """3. Importance 집중도 (entropy) 확인"""
    print(f"\n{'='*60}")
    print("3. Importance Entropy Analysis")
    print(f"{'='*60}")

    all_entropies = []
    sample_outputs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape

            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                continue

            routing_infos = output[-1]

            for layer_idx, layer_info in enumerate(routing_infos):
                if 'attention' not in layer_info:
                    continue

                attn = layer_info['attention']

                # Check for importance tensor
                importance = attn.get('importance')
                if importance is None:
                    # Try to compute from preferences
                    pref = attn.get('relational_q_pref')
                    if pref is not None:
                        # Normalize to get importance-like distribution
                        importance = torch.softmax(pref, dim=-1)

                if importance is not None:
                    # Entropy: -sum(p * log(p))
                    # importance shape: [B, L, N] or similar
                    if importance.dim() >= 2:
                        p = importance.float()
                        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
                        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
                        all_entropies.append(entropy.mean().item())

                        # Sample top tokens for first batch
                        if batch_idx == 0 and layer_idx == 0 and len(sample_outputs) < 5:
                            # Find tokens with highest importance concentration (lowest entropy)
                            batch_entropy = entropy[0]  # First sequence
                            top5_pos = torch.argsort(batch_entropy)[:5]  # Lowest entropy

                            tokens = input_ids[0].cpu().tolist()
                            for pos in top5_pos:
                                pos = pos.item()
                                if pos < len(tokens):
                                    tok_str = tokenizer.decode([tokens[pos]])
                                    sample_outputs.append({
                                        'pos': pos,
                                        'token': tok_str,
                                        'entropy': batch_entropy[pos].item()
                                    })

    results = {}

    if all_entropies:
        mean_entropy = np.mean(all_entropies)
        std_entropy = np.std(all_entropies)

        print(f"\n  Mean entropy: {mean_entropy:.4f} (std: {std_entropy:.4f})")
        print(f"  (Lower = more concentrated routing)")

        # Max entropy reference
        # For uniform distribution over N neurons: log(N)
        print(f"\n  Reference: uniform over 100 neurons → entropy ≈ {np.log(100):.2f}")
        print(f"  Reference: uniform over 10 neurons → entropy ≈ {np.log(10):.2f}")

        results['mean_entropy'] = float(mean_entropy)
        results['std_entropy'] = float(std_entropy)
        results['ref_uniform_100'] = float(np.log(100))
        results['ref_uniform_10'] = float(np.log(10))

    if sample_outputs:
        print(f"\n  Sample tokens with lowest entropy (most concentrated):")
        for s in sample_outputs[:5]:
            print(f"    pos {s['pos']}: '{s['token'].strip()}' (entropy={s['entropy']:.4f})")

        results['sample_low_entropy_tokens'] = [
            {'pos': s['pos'], 'token': s['token'].strip(), 'entropy': s['entropy']}
            for s in sample_outputs[:5]
        ]

    return results


def diagnose_neuron_roles(model, dataloader, tokenizer, device, target_neurons=None, max_batches=20):
    """4. 뉴런별 역할 분석 - 어떤 토큰에서 활성화되는가?

    Args:
        target_neurons: dict of {neuron_type: [neuron_indices]} to analyze
                       e.g., {'R': [106, 126, 62], 'FR': [0, 5]}
                       If None, uses top 3 most selected neurons per type
    """
    print(f"\n{'='*60}")
    print("4. Neuron Role Analysis (Token Preference)")
    print(f"{'='*60}")

    # Detect version
    router = get_router(model)
    version = detect_version(router)
    print(f"  Detected version: {version}")

    # Version-specific pref key map
    if version in ['16.4', '17.1']:
        pref_key_map = {
            'FQK_Q': 'fqk_q_pref',
            'FQK_K': 'fqk_k_pref',
            'FV': 'fv_pref',
            'RQK_Q': 'rqk_q_pref',
            'RQK_K': 'rqk_k_pref',
            'RV': 'rv_pref',
        }
    else:
        pref_key_map = {
            'FR': 'feature_r_pref',
            'FV': 'feature_v_pref',
            'R': 'relational_q_pref',
            'V': 'value_pref',
        }

    # Collect token preferences for each neuron
    # Structure: {neuron_type: {neuron_idx: [(token, score, context, pos)]}}
    neuron_token_data = defaultdict(lambda: defaultdict(list))

    # Track which neurons to analyze (will be filled if target_neurons is None)
    neuron_selection_counts = defaultdict(Counter)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape

            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                continue

            routing_infos = output[-1]

            # Use layer 0 for analysis
            if len(routing_infos) == 0 or 'attention' not in routing_infos[0]:
                continue

            attn = routing_infos[0]['attention']

            # Build pref_map dynamically
            pref_map = {nt: attn.get(pref_key) for nt, pref_key in pref_key_map.items()}

            # First pass: count selections to find top neurons
            for nt, pref in pref_map.items():
                if pref is None:
                    continue
                # pref shape: [B, L, N] or [B, N]
                if pref.dim() == 3:
                    # Token-level preference
                    _, topk_idx = torch.topk(pref, min(4, pref.shape[-1]), dim=-1)
                    for idx in topk_idx.view(-1).cpu().numpy():
                        neuron_selection_counts[nt][int(idx)] += 1
                elif pref.dim() == 2:
                    # Batch-level preference
                    _, topk_idx = torch.topk(pref, min(4, pref.shape[-1]), dim=-1)
                    for idx in topk_idx.view(-1).cpu().numpy():
                        neuron_selection_counts[nt][int(idx)] += 1

    # Determine target neurons if not specified
    if target_neurons is None:
        target_neurons = {}
        for nt, counts in neuron_selection_counts.items():
            if counts:
                top3 = [idx for idx, _ in counts.most_common(3)]
                target_neurons[nt] = top3

    print(f"\n  Analyzing neurons: {target_neurons}")

    # Second pass: collect token-level data for target neurons
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B, L = input_ids.shape
            tokens_batch = input_ids.cpu().tolist()

            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                continue

            routing_infos = output[-1]

            if len(routing_infos) == 0 or 'attention' not in routing_infos[0]:
                continue

            attn = routing_infos[0]['attention']

            # Build pref_map dynamically (same as first loop)
            pref_map = {nt: attn.get(pref_key) for nt, pref_key in pref_key_map.items()}

            for nt, neuron_indices in target_neurons.items():
                pref = pref_map.get(nt)
                if pref is None:
                    continue

                for neuron_idx in neuron_indices:
                    if neuron_idx >= pref.shape[-1]:
                        continue

                    if pref.dim() == 3:
                        # Token-level: pref[:, :, neuron_idx] -> [B, L]
                        neuron_pref = pref[:, :, neuron_idx]  # [B, L]

                        for b in range(B):
                            tokens = tokens_batch[b]
                            prefs_seq = neuron_pref[b].cpu().numpy()  # [L]

                            # Get top 10 positions with highest preference for this neuron
                            top_pos = np.argsort(prefs_seq)[-10:][::-1]

                            for pos in top_pos:
                                if pos >= len(tokens):
                                    continue

                                tok = tokenizer.decode([tokens[pos]]).strip()
                                score = float(prefs_seq[pos])

                                # Context: ±3 tokens
                                ctx_start = max(0, pos - 3)
                                ctx_end = min(len(tokens), pos + 4)
                                context_tokens = tokens[ctx_start:ctx_end]
                                context_str = tokenizer.decode(context_tokens)

                                # Mark target position in context
                                rel_pos = pos - ctx_start

                                neuron_token_data[nt][neuron_idx].append({
                                    'token': tok,
                                    'score': score,
                                    'context': context_str,
                                    'rel_pos': rel_pos,
                                    'batch': batch_idx,
                                    'pos': pos
                                })

                    elif pref.dim() == 2:
                        # Batch-level: preference is same for all tokens in batch
                        neuron_pref = pref[:, neuron_idx]  # [B]

                        for b in range(B):
                            score = neuron_pref[b].item()
                            tokens = tokens_batch[b]

                            # For batch-level, record sample tokens from high-preference batches
                            if score > 0.1:  # Threshold
                                sample_tokens = tokenizer.decode(tokens[:20])
                                neuron_token_data[nt][neuron_idx].append({
                                    'token': '[batch]',
                                    'score': score,
                                    'context': sample_tokens,
                                    'rel_pos': -1,
                                    'batch': batch_idx,
                                    'pos': -1
                                })

    # Analyze and report
    results = {}

    for nt, neurons_data in neuron_token_data.items():
        results[nt] = {}

        for neuron_idx, token_list in neurons_data.items():
            if not token_list:
                continue

            print(f"\n  {nt}_{neuron_idx}:")

            # Sort by score
            token_list.sort(key=lambda x: x['score'], reverse=True)

            # Token frequency analysis
            token_counts = Counter(item['token'] for item in token_list)
            top20_tokens = token_counts.most_common(20)

            print(f"    Top 20 tokens preferring this neuron:")
            for tok, cnt in top20_tokens:
                print(f"      '{tok}': {cnt}")

            # POS analysis (if spacy available)
            pos_counts = Counter()
            if HAS_SPACY:
                for item in token_list[:500]:  # Limit for speed
                    tok = item['token']
                    if tok and tok not in ['[batch]', '[CLS]', '[SEP]', '[PAD]']:
                        try:
                            doc = nlp(tok)
                            if doc:
                                pos_counts[doc[0].pos_] += 1
                        except:
                            pass

                if pos_counts:
                    print(f"\n    POS distribution:")
                    total_pos = sum(pos_counts.values())
                    for pos, cnt in pos_counts.most_common(10):
                        print(f"      {pos}: {cnt} ({100*cnt/total_pos:.1f}%)")

            # Context samples
            print(f"\n    Context samples (top 10 by score):")
            for item in token_list[:10]:
                ctx = item['context'].replace('\n', ' ')[:80]
                print(f"      [{item['score']:.4f}] ...{ctx}...")

            results[nt][neuron_idx] = {
                'top20_tokens': [(tok, cnt) for tok, cnt in top20_tokens],
                'pos_distribution': dict(pos_counts.most_common(10)) if pos_counts else {},
                'sample_contexts': [
                    {'score': item['score'], 'context': item['context'][:100]}
                    for item in token_list[:10]
                ],
                'total_samples': len(token_list)
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="DAWN Routing Diagnostics (v16.x/v17.x)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./routing_diagnosis")
    parser.add_argument("--target_neurons", type=str, default=None,
                       help="Target neurons to analyze, e.g., 'R:106,126,62;FR:0,5' (default: top 3 per type)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model (version-agnostic)
    from models import create_model_by_version
    from transformers import BertTokenizer

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Detect version
    version = config.get('model_version', None)
    if version is None:
        path_str = str(ckpt_path).lower()
        if 'v17.1' in path_str or 'v17_1' in path_str:
            version = '17.1'
        elif 'v17' in path_str:
            version = '17.0'
        elif 'v16.4' in path_str or 'v16_4' in path_str:
            version = '16.4'
        elif 'v16' in path_str:
            version = '16.0'
        else:
            version = '16.0'

    print(f"Model version: {version}")
    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    model = create_model_by_version(version, config)

    # Load with strict=False to handle missing keys in old checkpoints
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    load_result = model.load_state_dict(cleaned, strict=False)
    if load_result.missing_keys:
        print(f"  Note: Missing keys (using defaults): {load_result.missing_keys}")

    model.to(args.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    print(f"Loading data: {args.val_data}")
    val_data = torch.load(args.val_data)
    if isinstance(val_data, dict):
        input_ids = val_data.get('input_ids', val_data.get('tokens'))
    else:
        input_ids = val_data

    if input_ids.dim() == 1:
        seq_len = config.get('max_seq_len', 512)
        n_seqs = input_ids.shape[0] // seq_len
        input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)

    dataset = torch.utils.data.TensorDataset(input_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'='*60}")
    print("DAWN v16 Routing Diagnostics")
    print(f"{'='*60}")

    results = {}

    # 1. Usage EMA
    results['usage_ema'] = diagnose_usage_ema(model)

    # 2. Neuron Selection
    results['neuron_selection'] = diagnose_neuron_selection(model, dataloader, args.device, config, args.max_batches)

    # 3. Importance Entropy
    results['importance_entropy'] = diagnose_importance_entropy(model, dataloader, tokenizer, args.device, args.max_batches)

    # 4. Neuron Role Analysis
    # Parse target_neurons: "R:106,126,62;FR:0,5" -> {'R': [106, 126, 62], 'FR': [0, 5]}
    target_neurons = None
    if args.target_neurons:
        target_neurons = {}
        for part in args.target_neurons.split(';'):
            if ':' in part:
                nt, indices = part.split(':')
                target_neurons[nt.strip()] = [int(i.strip()) for i in indices.split(',')]
        if not target_neurons:
            target_neurons = None

    results['neuron_roles'] = diagnose_neuron_roles(
        model, dataloader, tokenizer, args.device,
        target_neurons=target_neurons, max_batches=args.max_batches
    )

    # Save results
    import json
    output_path = os.path.join(args.output_dir, "routing_diagnosis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*60}")
    print("Diagnostics Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
