#!/usr/bin/env python3
"""
DAWN v16 FR-R Co-selection Analysis
=====================================
FR(Feature-R) 뉴런과 R(Relational) 뉴런의 공동 선택 패턴 분석

Usage:
    python analyze_fr_r_coselection.py --checkpoint path/to/checkpoint.pt --val_data path/to/val.pt
"""

import argparse
import sys
import os
import torch
import numpy as np
from collections import Counter

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


def analyze_coselection(model, dataloader, device, config, max_batches=50):
    """Analyze FR-R co-selection patterns."""

    n_feature_r = config.get('n_feature_r', 96)
    n_relational = config.get('n_relational', 96)

    print(f"\n{'='*60}")
    print("FR-R Co-selection Analysis")
    print(f"{'='*60}")
    print(f"  n_feature_r: {n_feature_r}")
    print(f"  n_relational: {n_relational}")

    # Co-selection matrix
    co_selection = torch.zeros(n_feature_r, n_relational, device=device)

    # Individual selection counts
    fr_counts = torch.zeros(n_feature_r, device=device)
    r_counts = torch.zeros(n_relational, device=device)

    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            B = input_ids.shape[0]
            total_samples += B

            # Forward with routing info
            output = model(input_ids, return_routing_info=True)
            if not isinstance(output, tuple) or len(output) < 2:
                print("  Warning: Model doesn't return routing_info")
                continue

            routing_infos = output[-1]

            # Use layer 0 (can extend to all layers)
            if len(routing_infos) == 0 or 'attention' not in routing_infos[0]:
                continue

            attn = routing_infos[0]['attention']

            # Get weights (soft selection weights after softmax)
            fr_weights = attn.get('feature_r_weights')  # [B, n_feature_r] or [B, S, n_feature_r]
            r_weights = attn.get('relational_weights_Q')  # [B, n_relational] or [B, S, n_relational]

            if fr_weights is None or r_weights is None:
                # Try preference tensors instead
                fr_weights = attn.get('feature_r_pref')
                r_weights = attn.get('relational_q_pref')

                if fr_weights is None or r_weights is None:
                    print(f"  Warning: No FR/R weights found in routing_info")
                    print(f"  Available keys: {list(attn.keys())}")
                    continue

            # Handle different tensor shapes
            if fr_weights.dim() == 3:
                # [B, S, N] -> average over sequence
                fr_weights = fr_weights.mean(dim=1)  # [B, N]
            if r_weights.dim() == 3:
                r_weights = r_weights.mean(dim=1)  # [B, N]

            # Binary selection (threshold at mean or use top-k)
            # Using threshold: selected if weight > uniform expectation
            fr_threshold = 1.0 / n_feature_r
            r_threshold = 1.0 / n_relational

            fr_selected = (fr_weights > fr_threshold).float()  # [B, n_feature_r]
            r_selected = (r_weights > r_threshold).float()     # [B, n_relational]

            # Count individual selections
            fr_counts += fr_selected.sum(dim=0)
            r_counts += r_selected.sum(dim=0)

            # Co-occurrence: outer product per batch, then sum
            # co_selection[i, j] = count of times FR_i and R_j both selected
            for b in range(B):
                co_selection += torch.outer(fr_selected[b], r_selected[b])

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{max_batches} batches...")

    print(f"\n  Total samples analyzed: {total_samples}")

    results = analyze_coselection_matrix(co_selection, fr_counts, r_counts, n_feature_r, n_relational)
    results['total_samples'] = total_samples

    return results, co_selection.cpu().numpy()


def analyze_coselection_matrix(co_selection, fr_counts, r_counts, n_feature_r, n_relational):
    """Analyze the co-selection matrix."""

    results = {}

    # Normalize to get joint probability
    total = co_selection.sum()
    if total > 0:
        co_prob = co_selection / total
    else:
        co_prob = co_selection

    # Marginal probabilities
    fr_prob = fr_counts / fr_counts.sum() if fr_counts.sum() > 0 else fr_counts
    r_prob = r_counts / r_counts.sum() if r_counts.sum() > 0 else r_counts

    # Top 20 pairs
    flat_co = co_selection.view(-1)
    top_k = min(20, flat_co.numel())
    top_values, top_indices = torch.topk(flat_co, top_k)

    print(f"\n{'='*60}")
    print("Top 20 FR-R Pairs (by co-selection count)")
    print(f"{'='*60}")

    top_pairs = []
    for i in range(top_k):
        idx = top_indices[i].item()
        fr_idx = idx // n_relational
        r_idx = idx % n_relational
        count = top_values[i].item()
        pct = count / total.item() * 100 if total > 0 else 0

        print(f"  FR_{fr_idx} + R_{r_idx}: {int(count)} ({pct:.2f}%)")
        top_pairs.append({
            'fr_idx': fr_idx,
            'r_idx': r_idx,
            'count': int(count),
            'pct': pct
        })

    results['top_pairs'] = top_pairs

    # Concentration analysis
    print(f"\n{'='*60}")
    print("Pair Concentration Analysis")
    print(f"{'='*60}")

    # What % of co-selections come from top-k pairs?
    cumsum = torch.cumsum(torch.sort(flat_co, descending=True)[0], dim=0)

    top10_pct = (cumsum[9] / total * 100).item() if total > 0 and len(cumsum) > 9 else 0
    top50_pct = (cumsum[49] / total * 100).item() if total > 0 and len(cumsum) > 49 else 0
    top100_pct = (cumsum[99] / total * 100).item() if total > 0 and len(cumsum) > 99 else 0

    print(f"  Top 10 pairs: {top10_pct:.1f}% of all co-selections")
    print(f"  Top 50 pairs: {top50_pct:.1f}% of all co-selections")
    print(f"  Top 100 pairs: {top100_pct:.1f}% of all co-selections")

    # Entropy of co-selection distribution
    co_prob_flat = co_prob.view(-1)
    co_prob_flat = co_prob_flat[co_prob_flat > 0]  # Remove zeros for log
    entropy = -(co_prob_flat * co_prob_flat.log()).sum().item()
    max_entropy = np.log(n_feature_r * n_relational)
    normalized_entropy = entropy / max_entropy

    print(f"\n  Co-selection entropy: {entropy:.2f} (max: {max_entropy:.2f})")
    print(f"  Normalized entropy: {normalized_entropy:.2%}")
    print(f"  Interpretation:")
    if normalized_entropy < 0.5:
        print(f"    → CONCENTRATED: Strong FR-R pairing learned")
    elif normalized_entropy < 0.8:
        print(f"    → MODERATE: Some pairing structure")
    else:
        print(f"    → UNIFORM: Shared space convergence (FR/R independent)")

    results['concentration'] = {
        'top10_pct': top10_pct,
        'top50_pct': top50_pct,
        'top100_pct': top100_pct,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': normalized_entropy
    }

    # FR neuron analysis: which FR neurons have strongest pairing?
    print(f"\n{'='*60}")
    print("FR Neuron Specialization")
    print(f"{'='*60}")

    # For each FR, find its most common R partner
    fr_specialization = []
    for fr_idx in range(n_feature_r):
        row = co_selection[fr_idx]
        if row.sum() > 0:
            top_r = row.argmax().item()
            top_r_pct = (row[top_r] / row.sum() * 100).item()
            fr_specialization.append({
                'fr_idx': fr_idx,
                'top_r': top_r,
                'top_r_pct': top_r_pct,
                'total_count': int(row.sum().item())
            })

    # Sort by specialization (how concentrated on one R)
    fr_specialization.sort(key=lambda x: x['top_r_pct'], reverse=True)

    print(f"  Most specialized FR neurons (strongest R preference):")
    for item in fr_specialization[:10]:
        print(f"    FR_{item['fr_idx']}: {item['top_r_pct']:.1f}% with R_{item['top_r']} (n={item['total_count']})")

    results['fr_specialization'] = fr_specialization[:20]

    # R neuron analysis: which R neurons have strongest pairing?
    print(f"\n{'='*60}")
    print("R Neuron Specialization")
    print(f"{'='*60}")

    r_specialization = []
    for r_idx in range(n_relational):
        col = co_selection[:, r_idx]
        if col.sum() > 0:
            top_fr = col.argmax().item()
            top_fr_pct = (col[top_fr] / col.sum() * 100).item()
            r_specialization.append({
                'r_idx': r_idx,
                'top_fr': top_fr,
                'top_fr_pct': top_fr_pct,
                'total_count': int(col.sum().item())
            })

    r_specialization.sort(key=lambda x: x['top_fr_pct'], reverse=True)

    print(f"  Most specialized R neurons (strongest FR preference):")
    for item in r_specialization[:10]:
        print(f"    R_{item['r_idx']}: {item['top_fr_pct']:.1f}% with FR_{item['top_fr']} (n={item['total_count']})")

    results['r_specialization'] = r_specialization[:20]

    return results


def analyze_neuron_subspace_diversity(model, dataloader, device, config, max_batches=20):
    """FR/R 뉴런 간 subspace 다양성 분석 - 뉴런들이 서로 다른 subspace를 사용하는가?"""

    import torch.nn.functional as F

    print(f"\n{'='*60}")
    print("Neuron Subspace Diversity Analysis")
    print(f"{'='*60}")

    results = {}

    # Get neuron embeddings from router
    if not hasattr(model, 'global_routers'):
        print("  Model doesn't have global_routers")
        return results

    router = model.global_routers.neuron_router

    # Analyze FR neurons (compression matrices)
    if hasattr(router, 'neuron_emb_feature_r'):
        print(f"\n  FR Neurons Subspace Analysis:")

        # Get FR neuron parameters from layers
        # FR neurons: W_down [d_model, rank] per neuron
        fr_embeddings = []

        # Check if we can access layer neurons
        for layer in model.layers:
            if hasattr(layer, 'attn') and hasattr(layer.attn, 'feature_r_neurons'):
                fr_neurons = layer.attn.feature_r_neurons  # Should be [n_feature_r, d_model, rank]
                if fr_neurons is not None:
                    fr_embeddings.append(fr_neurons)
                    break

        if fr_embeddings:
            fr_neurons = fr_embeddings[0]  # [n_feature_r, d_model, rank]
            n_fr = fr_neurons.shape[0]

            # Flatten each neuron's matrix to a vector for comparison
            fr_flat = fr_neurons.view(n_fr, -1)  # [n_feature_r, d_model * rank]

            # Compute pairwise cosine similarity
            fr_norm = F.normalize(fr_flat, dim=-1)
            fr_sim = torch.mm(fr_norm, fr_norm.t())  # [n_fr, n_fr]

            # Remove diagonal (self-similarity)
            mask = ~torch.eye(n_fr, dtype=torch.bool, device=device)
            fr_sim_off_diag = fr_sim[mask]

            avg_sim = fr_sim_off_diag.mean().item()
            max_sim = fr_sim_off_diag.max().item()
            min_sim = fr_sim_off_diag.min().item()
            std_sim = fr_sim_off_diag.std().item()

            print(f"    Pairwise cosine similarity:")
            print(f"      Mean: {avg_sim:.4f}")
            print(f"      Std:  {std_sim:.4f}")
            print(f"      Min:  {min_sim:.4f}")
            print(f"      Max:  {max_sim:.4f}")

            # Interpretation
            if avg_sim < 0.3:
                print(f"    → DIVERSE: FR neurons use distinct subspaces (good!)")
            elif avg_sim < 0.6:
                print(f"    → MODERATE: Some overlap in FR neuron subspaces")
            else:
                print(f"    → COLLAPSED: FR neurons converging to similar subspaces (bad!)")

            # Find most similar pairs
            top_k_pairs = 10
            sim_flat = fr_sim.view(-1)
            # Exclude diagonal
            for i in range(n_fr):
                sim_flat[i * n_fr + i] = -1

            top_vals, top_idx = torch.topk(sim_flat, top_k_pairs)
            print(f"\n    Most similar FR neuron pairs:")
            for i in range(top_k_pairs):
                idx = top_idx[i].item()
                fr_i = idx // n_fr
                fr_j = idx % n_fr
                sim_val = top_vals[i].item()
                print(f"      FR_{fr_i} - FR_{fr_j}: {sim_val:.4f}")

            results['fr_subspace'] = {
                'mean_similarity': avg_sim,
                'std_similarity': std_sim,
                'min_similarity': min_sim,
                'max_similarity': max_sim,
                'interpretation': 'diverse' if avg_sim < 0.3 else ('moderate' if avg_sim < 0.6 else 'collapsed'),
                'top_similar_pairs': [(int(top_idx[i].item() // n_fr), int(top_idx[i].item() % n_fr), float(top_vals[i].item())) for i in range(top_k_pairs)]
            }
        else:
            print("    Could not access FR neuron weights")

    # Analyze R neurons (expansion matrices)
    print(f"\n  R Neurons Subspace Analysis:")

    r_embeddings = []
    for layer in model.layers:
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'relational_neurons_Q'):
            r_neurons = layer.attn.relational_neurons_Q  # Should be [n_relational, rank, d_model]
            if r_neurons is not None:
                r_embeddings.append(r_neurons)
                break

    if r_embeddings:
        r_neurons = r_embeddings[0]  # [n_relational, rank, d_model]
        n_r = r_neurons.shape[0]

        # Flatten each neuron's matrix
        r_flat = r_neurons.view(n_r, -1)  # [n_relational, rank * d_model]

        # Compute pairwise cosine similarity
        r_norm = F.normalize(r_flat, dim=-1)
        r_sim = torch.mm(r_norm, r_norm.t())  # [n_r, n_r]

        # Remove diagonal
        mask = ~torch.eye(n_r, dtype=torch.bool, device=device)
        r_sim_off_diag = r_sim[mask]

        avg_sim = r_sim_off_diag.mean().item()
        max_sim = r_sim_off_diag.max().item()
        min_sim = r_sim_off_diag.min().item()
        std_sim = r_sim_off_diag.std().item()

        print(f"    Pairwise cosine similarity:")
        print(f"      Mean: {avg_sim:.4f}")
        print(f"      Std:  {std_sim:.4f}")
        print(f"      Min:  {min_sim:.4f}")
        print(f"      Max:  {max_sim:.4f}")

        if avg_sim < 0.3:
            print(f"    → DIVERSE: R neurons use distinct subspaces (good!)")
        elif avg_sim < 0.6:
            print(f"    → MODERATE: Some overlap in R neuron subspaces")
        else:
            print(f"    → COLLAPSED: R neurons converging to similar subspaces (bad!)")

        # Find most similar pairs
        top_k_pairs = 10
        sim_flat = r_sim.view(-1)
        for i in range(n_r):
            sim_flat[i * n_r + i] = -1

        top_vals, top_idx = torch.topk(sim_flat, top_k_pairs)
        print(f"\n    Most similar R neuron pairs:")
        for i in range(top_k_pairs):
            idx = top_idx[i].item()
            r_i = idx // n_r
            r_j = idx % n_r
            sim_val = top_vals[i].item()
            print(f"      R_{r_i} - R_{r_j}: {sim_val:.4f}")

        results['r_subspace'] = {
            'mean_similarity': avg_sim,
            'std_similarity': std_sim,
            'min_similarity': min_sim,
            'max_similarity': max_sim,
            'interpretation': 'diverse' if avg_sim < 0.3 else ('moderate' if avg_sim < 0.6 else 'collapsed'),
            'top_similar_pairs': [(int(top_idx[i].item() // n_r), int(top_idx[i].item() % n_r), float(top_vals[i].item())) for i in range(top_k_pairs)]
        }
    else:
        print("    Could not access R neuron weights")

    # Analyze router embeddings (d_space dimension)
    print(f"\n  Router Embedding Analysis:")

    router_embs = {
        'FR': getattr(router, 'neuron_emb_feature_r', None),
        'FV': getattr(router, 'neuron_emb_feature_v', None),
        'R': getattr(router, 'neuron_emb_relational', None),
        'V': getattr(router, 'neuron_emb_value', None),
    }

    for name, emb in router_embs.items():
        if emb is None:
            continue

        n_neurons = emb.shape[0]
        emb_norm = F.normalize(emb, dim=-1)
        sim = torch.mm(emb_norm, emb_norm.t())

        mask = ~torch.eye(n_neurons, dtype=torch.bool, device=device)
        sim_off_diag = sim[mask]

        avg_sim = sim_off_diag.mean().item()
        print(f"    {name} router embeddings: mean sim = {avg_sim:.4f}", end="")

        if avg_sim < 0.3:
            print(" (diverse)")
        elif avg_sim < 0.6:
            print(" (moderate)")
        else:
            print(" (collapsed!)")

        results[f'{name.lower()}_router_emb_similarity'] = avg_sim

    return results


def analyze_fr_r_subspace_similarity(model, dataloader, device, config, max_batches=20):
    """
    FR/R Subspace Similarity Analysis (Data-Driven)

    실제 데이터를 통해 FR 뉴런 출력 벡터와 R 뉴런 입력 선호도를 분석:
    1. FR output vectors: 실제 x @ FR_neuron 출력의 평균
    2. R input preference: SVD로 각 R 뉴런의 선호 입력 방향 추출
    3. FR-R alignment: FR 출력이 어떤 R 입력과 정렬되는지
    """
    import torch.nn.functional as F

    print(f"\n{'='*60}")
    print("FR/R Subspace Similarity Analysis (Data-Driven)")
    print(f"{'='*60}")

    n_feature_r = config.get('n_feature_r', 96)
    n_relational = config.get('n_relational', 96)
    rank = config.get('rank', 64)

    print(f"  n_feature_r: {n_feature_r}")
    print(f"  n_relational: {n_relational}")
    print(f"  rank (subspace dim): {rank}")

    results = {}

    # Get neuron weights from first layer
    layer = model.layers[0]

    if not hasattr(layer.attn, 'feature_r_neurons') or not hasattr(layer.attn, 'relational_neurons_Q'):
        print("  Error: Cannot access FR/R neuron weights")
        return results, None

    fr_neurons = layer.attn.feature_r_neurons  # [n_fr, d_model, rank]
    r_neurons = layer.attn.relational_neurons_Q  # [n_r, rank, d_model]

    if fr_neurons is None or r_neurons is None:
        print("  Error: FR/R neurons are None")
        return results, None

    print(f"  FR neurons shape: {fr_neurons.shape}")
    print(f"  R neurons shape: {r_neurons.shape}")

    # =========================================================================
    # Step 1: Compute FR output vectors using actual data
    # =========================================================================
    print(f"\n  Computing FR output vectors from {max_batches} batches...")

    fr_outputs = [[] for _ in range(n_feature_r)]

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
            else:
                input_ids = batch.to(device)

            # Get embeddings (token + position)
            x = model.token_emb(input_ids)

            if hasattr(model, 'pos_emb'):
                seq_len = input_ids.shape[1]
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                x = x + model.pos_emb(positions)

            # Compute FR outputs: x @ FR_neuron[i] → [B, S, rank]
            for i in range(n_feature_r):
                # fr_neurons[i]: [d_model, rank]
                out = torch.matmul(x, fr_neurons[i])  # [B, S, rank]
                # Average over batch and sequence to get representative vector
                fr_outputs[i].append(out.mean(dim=[0, 1]))  # [rank]

            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {batch_idx + 1}/{max_batches} batches...")

    # Average FR outputs across batches
    fr_vecs = torch.stack([torch.stack(outputs).mean(0) for outputs in fr_outputs])  # [n_fr, rank]
    fr_vecs_norm = F.normalize(fr_vecs, dim=-1)

    # =========================================================================
    # Step 2: Compute R "preferred input" vectors via SVD
    # =========================================================================
    print(f"\n  Computing R preferred input directions (SVD)...")

    r_vecs = []
    r_singular_values = []

    for j in range(n_relational):
        W = r_neurons[j]  # [rank, d_model]
        # SVD to find dominant input direction
        # W = U @ S @ V^T, U[:,0] is the input direction that produces max output
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r_vecs.append(U[:, 0])  # First left singular vector (rank-dimensional)
        r_singular_values.append(S[0].item())

    r_vecs = torch.stack(r_vecs)  # [n_r, rank]
    r_vecs_norm = F.normalize(r_vecs, dim=-1)

    # =========================================================================
    # Step 3: FR-FR Similarity (in rank space)
    # =========================================================================
    print(f"\n{'='*60}")
    print("FR Output Similarity (64-dim rank space)")
    print(f"{'='*60}")

    fr_fr_sim = torch.mm(fr_vecs_norm, fr_vecs_norm.t())  # [n_fr, n_fr]

    # Remove diagonal for statistics
    mask = ~torch.eye(n_feature_r, dtype=torch.bool, device=device)
    fr_fr_off_diag = fr_fr_sim[mask]

    avg_sim = fr_fr_off_diag.mean().item()
    std_sim = fr_fr_off_diag.std().item()
    min_sim = fr_fr_off_diag.min().item()
    max_sim = fr_fr_off_diag.max().item()

    print(f"  Pairwise cosine similarity:")
    print(f"    Mean: {avg_sim:.4f}")
    print(f"    Std:  {std_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")

    if avg_sim < 0.3:
        interp = "DIVERSE: FR neurons output in distinct directions (good!)"
    elif avg_sim < 0.6:
        interp = "MODERATE: Some overlap in FR output directions"
    else:
        interp = "COLLAPSED: FR outputs converging (bad!)"
    print(f"  → {interp}")

    # Top similar FR pairs
    top_k = 10
    sim_flat = fr_fr_sim.clone().view(-1)
    for i in range(n_feature_r):
        sim_flat[i * n_feature_r + i] = -2  # Exclude diagonal

    top_vals, top_idx = torch.topk(sim_flat, top_k)
    print(f"\n  Most similar FR output pairs:")
    fr_similar_pairs = []
    for i in range(top_k):
        idx = top_idx[i].item()
        fr_i = idx // n_feature_r
        fr_j = idx % n_feature_r
        sim_val = top_vals[i].item()
        print(f"    FR_{fr_i} - FR_{fr_j}: {sim_val:.4f}")
        fr_similar_pairs.append((fr_i, fr_j, sim_val))

    results['fr_output_similarity'] = {
        'mean': avg_sim,
        'std': std_sim,
        'min': min_sim,
        'max': max_sim,
        'interpretation': interp,
        'top_similar_pairs': fr_similar_pairs
    }

    # =========================================================================
    # Step 4: R-R Similarity (preferred input directions)
    # =========================================================================
    print(f"\n{'='*60}")
    print("R Preferred Input Similarity (64-dim rank space)")
    print(f"{'='*60}")

    r_r_sim = torch.mm(r_vecs_norm, r_vecs_norm.t())  # [n_r, n_r]

    mask = ~torch.eye(n_relational, dtype=torch.bool, device=device)
    r_r_off_diag = r_r_sim[mask]

    avg_sim = r_r_off_diag.mean().item()
    std_sim = r_r_off_diag.std().item()
    min_sim = r_r_off_diag.min().item()
    max_sim = r_r_off_diag.max().item()

    print(f"  Pairwise cosine similarity:")
    print(f"    Mean: {avg_sim:.4f}")
    print(f"    Std:  {std_sim:.4f}")
    print(f"    Min:  {min_sim:.4f}")
    print(f"    Max:  {max_sim:.4f}")

    if avg_sim < 0.3:
        interp = "DIVERSE: R neurons prefer distinct input directions (good!)"
    elif avg_sim < 0.6:
        interp = "MODERATE: Some overlap in R input preferences"
    else:
        interp = "COLLAPSED: R input preferences converging (bad!)"
    print(f"  → {interp}")

    # Top similar R pairs
    sim_flat = r_r_sim.clone().view(-1)
    for i in range(n_relational):
        sim_flat[i * n_relational + i] = -2

    top_vals, top_idx = torch.topk(sim_flat, top_k)
    print(f"\n  Most similar R preferred input pairs:")
    r_similar_pairs = []
    for i in range(top_k):
        idx = top_idx[i].item()
        r_i = idx // n_relational
        r_j = idx % n_relational
        sim_val = top_vals[i].item()
        print(f"    R_{r_i} - R_{r_j}: {sim_val:.4f}")
        r_similar_pairs.append((r_i, r_j, sim_val))

    results['r_input_similarity'] = {
        'mean': avg_sim,
        'std': std_sim,
        'min': min_sim,
        'max': max_sim,
        'interpretation': interp,
        'top_similar_pairs': r_similar_pairs
    }

    # =========================================================================
    # Step 5: FR-R Alignment (which FR outputs align with which R inputs)
    # =========================================================================
    print(f"\n{'='*60}")
    print("FR-R Alignment (FR output → R input)")
    print(f"{'='*60}")

    fr_r_alignment = torch.mm(fr_vecs_norm, r_vecs_norm.t())  # [n_fr, n_r]

    print(f"  Alignment matrix shape: [{n_feature_r}, {n_relational}]")

    # Statistics
    avg_align = fr_r_alignment.abs().mean().item()
    max_align = fr_r_alignment.abs().max().item()

    print(f"  Mean |alignment|: {avg_align:.4f}")
    print(f"  Max |alignment|:  {max_align:.4f}")

    # Top aligned FR-R pairs
    align_flat = fr_r_alignment.abs().view(-1)
    top_vals, top_idx = torch.topk(align_flat, 20)

    print(f"\n  Top 20 FR-R Aligned Pairs:")
    fr_r_pairs = []
    for i in range(20):
        idx = top_idx[i].item()
        fr_i = idx // n_relational
        r_j = idx % n_relational
        align_val = fr_r_alignment[fr_i, r_j].item()
        print(f"    FR_{fr_i} → R_{r_j}: {align_val:+.4f}")
        fr_r_pairs.append((fr_i, r_j, align_val))

    results['fr_r_alignment'] = {
        'mean_abs': avg_align,
        'max_abs': max_align,
        'top_pairs': fr_r_pairs
    }

    # =========================================================================
    # Step 6: Per-FR specialization (how many R neurons does each FR align with?)
    # =========================================================================
    print(f"\n{'='*60}")
    print("FR Specialization (alignment concentration)")
    print(f"{'='*60}")

    # For each FR, compute entropy of its R alignments
    fr_align_probs = F.softmax(fr_r_alignment.abs() * 5, dim=-1)  # Temperature scaling
    fr_entropy = -(fr_align_probs * (fr_align_probs + 1e-10).log()).sum(dim=-1)  # [n_fr]

    max_entropy = np.log(n_relational)
    normalized_entropy = fr_entropy / max_entropy

    # Most specialized (low entropy) and least specialized (high entropy)
    sorted_idx = torch.argsort(normalized_entropy)

    print(f"  Most specialized FR neurons (low alignment entropy):")
    for i in range(min(10, n_feature_r)):
        fr_i = sorted_idx[i].item()
        ent = normalized_entropy[fr_i].item()
        top_r = fr_r_alignment[fr_i].abs().argmax().item()
        top_val = fr_r_alignment[fr_i, top_r].item()
        print(f"    FR_{fr_i}: entropy={ent:.3f}, strongest→R_{top_r} ({top_val:+.4f})")

    print(f"\n  Least specialized FR neurons (high alignment entropy):")
    for i in range(max(0, n_feature_r - 5), n_feature_r):
        fr_i = sorted_idx[i].item()
        ent = normalized_entropy[fr_i].item()
        print(f"    FR_{fr_i}: entropy={ent:.3f}")

    results['fr_specialization'] = {
        'mean_entropy': normalized_entropy.mean().item(),
        'min_entropy': normalized_entropy.min().item(),
        'max_entropy': normalized_entropy.max().item(),
    }

    # =========================================================================
    # Step 7: Per-R receptiveness (how many FR neurons feed into each R?)
    # =========================================================================
    print(f"\n{'='*60}")
    print("R Receptiveness (how many FR neurons feed each R)")
    print(f"{'='*60}")

    r_align_probs = F.softmax(fr_r_alignment.abs().t() * 5, dim=-1)  # [n_r, n_fr]
    r_entropy = -(r_align_probs * (r_align_probs + 1e-10).log()).sum(dim=-1)

    max_entropy_r = np.log(n_feature_r)
    normalized_entropy_r = r_entropy / max_entropy_r

    sorted_idx_r = torch.argsort(normalized_entropy_r)

    print(f"  Most selective R neurons (few FR sources):")
    for i in range(min(10, n_relational)):
        r_j = sorted_idx_r[i].item()
        ent = normalized_entropy_r[r_j].item()
        top_fr = fr_r_alignment[:, r_j].abs().argmax().item()
        top_val = fr_r_alignment[top_fr, r_j].item()
        print(f"    R_{r_j}: entropy={ent:.3f}, strongest←FR_{top_fr} ({top_val:+.4f})")

    results['r_receptiveness'] = {
        'mean_entropy': normalized_entropy_r.mean().item(),
        'min_entropy': normalized_entropy_r.min().item(),
        'max_entropy': normalized_entropy_r.max().item(),
    }

    return results, fr_r_alignment.cpu().numpy()


def save_heatmap(co_selection, output_path):
    """Save co-selection heatmap as image."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))
        plt.imshow(co_selection, aspect='auto', cmap='hot')
        plt.colorbar(label='Co-selection count')
        plt.xlabel('R neuron index')
        plt.ylabel('FR neuron index')
        plt.title('FR-R Co-selection Heatmap')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n  Heatmap saved to: {output_path}")
    except ImportError:
        print("\n  Warning: matplotlib not available, skipping heatmap")


def save_alignment_heatmap(alignment_matrix, output_path):
    """Save FR-R alignment heatmap as image."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 10))

        # Use diverging colormap for alignment (positive/negative)
        vmax = max(abs(alignment_matrix.min()), abs(alignment_matrix.max()))
        plt.imshow(alignment_matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(label='Alignment (cosine similarity)')
        plt.xlabel('R neuron index (preferred input direction)')
        plt.ylabel('FR neuron index (output direction)')
        plt.title('FR-R Subspace Alignment\n(+: aligned, -: anti-aligned)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n  Alignment heatmap saved to: {output_path}")
    except ImportError:
        print("\n  Warning: matplotlib not available, skipping alignment heatmap")


def main():
    parser = argparse.ArgumentParser(description="DAWN v16 FR-R Co-selection Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./coselection_analysis")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    try:
        from models.model_v16 import DAWN
    except ImportError:
        from model_v16 import DAWN

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    config = checkpoint.get('config', {})

    # Check model version for v16.1
    model_version = config.get('model_version', '16.0')
    if model_version == '16.1':
        try:
            from models.model_v16_1 import DAWN
        except ImportError:
            from model_v16_1 import DAWN

    print(f"Model version: {model_version}")
    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    model = DAWN(**config)

    # Load with strict=False to handle missing excitability_weight in old checkpoints
    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if load_result.missing_keys:
        print(f"  Note: Missing keys (using defaults): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  Note: Unexpected keys (ignored): {load_result.unexpected_keys}")

    model.to(args.device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

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

    # Run co-selection analysis
    results, co_matrix = analyze_coselection(model, dataloader, args.device, config, args.max_batches)

    # Run subspace diversity analysis (weight-based)
    subspace_results = analyze_neuron_subspace_diversity(model, dataloader, args.device, config, args.max_batches)
    results['subspace_diversity'] = subspace_results

    # Run FR/R subspace similarity analysis (data-driven)
    fr_r_sim_results, fr_r_alignment = analyze_fr_r_subspace_similarity(
        model, dataloader, args.device, config, args.max_batches
    )
    results['fr_r_subspace_similarity'] = fr_r_sim_results

    # Save heatmaps
    heatmap_path = os.path.join(args.output_dir, "fr_r_coselection_heatmap.png")
    save_heatmap(co_matrix, heatmap_path)

    # Save FR-R alignment heatmap
    if fr_r_alignment is not None and len(fr_r_alignment) > 0:
        alignment_heatmap_path = os.path.join(args.output_dir, "fr_r_alignment_heatmap.png")
        save_alignment_heatmap(fr_r_alignment, alignment_heatmap_path)
        np.save(os.path.join(args.output_dir, "fr_r_alignment_matrix.npy"), fr_r_alignment)
        print(f"FR-R alignment matrix saved to: {args.output_dir}/fr_r_alignment_matrix.npy")

    # Save results
    import json
    output_path = os.path.join(args.output_dir, "fr_r_coselection.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")

    # Save raw matrix
    np.save(os.path.join(args.output_dir, "co_selection_matrix.npy"), co_matrix)
    print(f"Raw matrix saved to: {args.output_dir}/co_selection_matrix.npy")

    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
