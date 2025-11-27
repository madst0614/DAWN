"""
DAWN (Dynamic Architecture With Neurons) Training Script

Usage:
    # 기본 학습 (자동으로 최신 체크포인트 이어서 학습)
    python scripts/train.py

    # 처음부터 새로 시작
    python scripts/train.py --from-scratch

    # 특정 체크포인트 폴더에서 이어서 학습
    python scripts/train.py --resume checkpoints/run_20240101_120000_1234

    # 커스텀 config 파일 사용
    python scripts/train.py --config configs/my_config.yaml

Checkpoint Options:
    (기본)           - 자동으로 최신 best_model.pt 탐색 후 이어서 학습
    --from-scratch   - 자동 탐색 비활성화, 처음부터 시작
    --resume <폴더>  - 지정한 폴더의 best_model.pt에서 이어서 학습
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress noisy torch inductor warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')
warnings.filterwarnings('ignore', message='.*online softmax.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import time
import numpy as np
import math

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

from models import DAWN, DAWNLanguageModel, create_model_by_version  # v7.1 default, version-aware loading
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.data import MLM_CONFIG, apply_mlm_masking, TextDataset, collate_fn_dynamic_padding, load_data, compute_mlm_accuracy


# ============================================================
# DEBUG LOGGING FUNCTIONS
# ============================================================

class DebugLogger:
    """Debug logger for basis_up analysis, gradients, and orthogonality loss"""

    # Epochs to log detailed stats
    LOG_EPOCHS = {0, 1, 5, 10, 20, 50, 100}

    def __init__(self, log_file_path):
        self.log_file = log_file_path
        self.epoch_logged = set()

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DAWN Debug Log - Basis_up Analysis\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def should_log_epoch(self, epoch):
        """Check if this epoch should be logged"""
        return epoch in self.LOG_EPOCHS or epoch % 10 == 0

    def log(self, message):
        """Write message to debug log file"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def log_section(self, title):
        """Log a section header"""
        self.log(f"\n{'='*60}")
        self.log(f"{title}")
        self.log(f"{'='*60}")

    def log_basis_stats(self, model, epoch, step=None):
        """
        Log basis statistics at specific epochs

        Supports:
        - v7.8: NeuronBank W_Q/K/V/O (no basis)
        - v7.7: basis_qk / basis_vo
        - v7.6: basis_down / basis_up

        Tracks:
        - Singular value distribution per basis/neuron
        - Overall condition number
        - Potential collapse detection
        """
        base_model = get_underlying_model(model)

        # Only for models with shared_basis (aliased as neuron_bank for v7.8)
        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Basis/Neuron Stats - Epoch {epoch}{step_str}")

        sb = base_model.shared_basis

        # Detect model version
        is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')  # NeuronBank
        is_v77 = hasattr(sb, 'basis_qk')

        with torch.no_grad():
            if is_v78:
                # v7.8: NeuronBank with independent W_Q/K/V/O per neuron
                self.log(f"\n[v7.8 NeuronBank - Independent Neuron W Matrices]")
                n_neurons = sb.n_neurons

                for W_name, W in [('W_Q', sb.W_Q), ('W_K', sb.W_K), ('W_V', sb.W_V), ('W_O', sb.W_O)]:
                    # Per-neuron condition numbers
                    cond_nums = []
                    for i in range(n_neurons):
                        _, S, _ = torch.linalg.svd(W[i])
                        cond = (S[0] / (S[-1] + 1e-10)).item()
                        cond_nums.append(cond)

                    avg_cond = sum(cond_nums) / len(cond_nums)
                    max_cond = max(cond_nums)
                    min_cond = min(cond_nums)

                    self.log(f"\n  {W_name} condition numbers (across {n_neurons} neurons):")
                    self.log(f"    mean = {avg_cond:.2f}, max = {max_cond:.2f}, min = {min_cond:.2f}")

                    if max_cond > 100:
                        self.log(f"    ⚠️  WARNING: High condition in some neurons!")
                    elif max_cond > 10:
                        self.log(f"    ⚠️  CAUTION: Condition numbers drifting from init")
                    else:
                        self.log(f"    ✓ Orthogonality well maintained")

            elif is_v77:
                # v7.7: basis_vo is the "output" basis (O uses its transpose)
                basis_o = sb.basis_vo.detach()  # [n_basis, D, rank]
                basis_o_name = "Basis_VO"
                basis_qk = sb.basis_qk.detach()  # [n_basis, D, rank]
                basis_qk_name = "Basis_QK"

                n_basis = basis_o.shape[0]

                # Per-basis singular values for O projection basis
                self.log(f"\n[{basis_o_name} Per-Basis Singular Values (top 5)]")
                all_singular_values = []
                for i in range(n_basis):
                    _, S, _ = torch.linalg.svd(basis_o[i])
                    all_singular_values.extend(S.cpu().tolist())
                    self.log(f"  {basis_o_name}[{i}]: {S[:5].cpu().numpy()}")

                # Overall condition number for O basis
                B_o_flat = basis_o.view(n_basis, -1)
                _, S_all, _ = torch.linalg.svd(B_o_flat)
                sigma_max = S_all[0].item()
                sigma_min = S_all[-1].item()
                condition_number = sigma_max / (sigma_min + 1e-10)

                self.log(f"\n[{basis_o_name} Overall Condition Number]")
                self.log(f"  σ_max = {sigma_max:.6f}")
                self.log(f"  σ_min = {sigma_min:.10f}")
                self.log(f"  Condition number = {condition_number:.2e}")

                # Collapse detection
                if condition_number > 1e6:
                    self.log(f"  ⚠️  WARNING: High condition number indicates potential collapse!")
                elif condition_number > 1e4:
                    self.log(f"  ⚠️  CAUTION: Condition number getting high")
                else:
                    self.log(f"  ✓ Condition number is healthy")

                # Also log QK basis for comparison
                B_qk_flat = basis_qk.view(n_basis, -1)
                _, S_qk, _ = torch.linalg.svd(B_qk_flat)
                cond_qk = S_qk[0].item() / (S_qk[-1].item() + 1e-10)

                self.log(f"\n[{basis_qk_name} Condition Number (for comparison)]")
                self.log(f"  σ_max = {S_qk[0].item():.6f}")
                self.log(f"  σ_min = {S_qk[-1].item():.10f}")
                self.log(f"  Condition number = {cond_qk:.2e}")

            elif hasattr(sb, 'basis_up') and hasattr(sb, 'basis_down'):
                # v7.6: basis_up is the "output" basis
                basis_o = sb.basis_up.detach()  # [n_basis, rank, D]
                basis_o_name = "Basis_up"
                basis_qk = sb.basis_down.detach()  # [n_basis, D, rank]
                basis_qk_name = "Basis_down"

                n_basis = basis_o.shape[0]

                # Per-basis singular values for O projection basis
                self.log(f"\n[{basis_o_name} Per-Basis Singular Values (top 5)]")
                all_singular_values = []
                for i in range(n_basis):
                    _, S, _ = torch.linalg.svd(basis_o[i])
                    all_singular_values.extend(S.cpu().tolist())
                    self.log(f"  {basis_o_name}[{i}]: {S[:5].cpu().numpy()}")

                # Overall condition number for O basis
                B_o_flat = basis_o.view(n_basis, -1)
                _, S_all, _ = torch.linalg.svd(B_o_flat)
                sigma_max = S_all[0].item()
                sigma_min = S_all[-1].item()
                condition_number = sigma_max / (sigma_min + 1e-10)

                self.log(f"\n[{basis_o_name} Overall Condition Number]")
                self.log(f"  σ_max = {sigma_max:.6f}")
                self.log(f"  σ_min = {sigma_min:.10f}")
                self.log(f"  Condition number = {condition_number:.2e}")

                # Collapse detection
                if condition_number > 1e6:
                    self.log(f"  ⚠️  WARNING: High condition number indicates potential collapse!")
                elif condition_number > 1e4:
                    self.log(f"  ⚠️  CAUTION: Condition number getting high")
                else:
                    self.log(f"  ✓ Condition number is healthy")

                # Also log QK basis for comparison
                B_qk_flat = basis_qk.view(n_basis, -1)
                _, S_qk, _ = torch.linalg.svd(B_qk_flat)
                cond_qk = S_qk[0].item() / (S_qk[-1].item() + 1e-10)

                self.log(f"\n[{basis_qk_name} Condition Number (for comparison)]")
                self.log(f"  σ_max = {S_qk[0].item():.6f}")
                self.log(f"  σ_min = {S_qk[-1].item():.10f}")
                self.log(f"  Condition number = {cond_qk:.2e}")
            else:
                # Other versions (v7.5, v7.1, etc.) - skip detailed basis logging
                self.log(f"\n[Note: Detailed basis stats not available for this model version]")

    def log_gradient_flow(self, model, epoch, step=None):
        """
        Log gradient flow for basis/neuron parameters

        Supports:
        - v7.8: neuron_bank.W_Q/K/V/O
        - v7.6/v7.7: basis_down/up, basis_qk/vo

        Tracks:
        - Gradient norms for basis/neuron W parameters
        - Parameter norms
        - Gradient/parameter ratio
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Gradient Flow - Epoch {epoch}{step_str}")

        self.log("\n[Basis/Neuron Gradient Analysis]")
        for name, param in base_model.named_parameters():
            # Match basis or neuron_bank parameters
            if ('basis' in name or 'neuron_bank' in name) and param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.norm().item()
                ratio = grad_norm / (param_norm + 1e-10)

                self.log(f"  {name}:")
                self.log(f"    grad_norm = {grad_norm:.8f}")
                self.log(f"    param_norm = {param_norm:.6f}")
                self.log(f"    ratio = {ratio:.8f}")

                # Gradient health check
                if grad_norm < 1e-8:
                    self.log(f"    ⚠️  WARNING: Vanishing gradient!")
                elif grad_norm > 100:
                    self.log(f"    ⚠️  WARNING: Exploding gradient!")

    def log_orthogonality_breakdown(self, model, epoch, step=None):
        """
        Log orthogonality loss breakdown

        Supports:
        - v7.8: neuron_bank W_Q/K/V/O (per-neuron orthogonality)
        - v7.7: basis_qk / basis_vo
        - v7.6: basis_down / basis_up

        Tracks:
        - ortho loss for each basis/neuron
        - Which direction dominates
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Orthogonality Loss Breakdown - Epoch {epoch}{step_str}")

        with torch.no_grad():
            sb = base_model.shared_basis

            # Detect model version
            is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')
            is_v77 = hasattr(sb, 'basis_qk')

            if is_v78:
                # v7.8: Per-neuron W orthogonality
                self.log(f"\n[Orthogonality Loss Components (v7.8 - Per Neuron)]")
                n_neurons = sb.n_neurons

                for W_name, W in [('W_Q', sb.W_Q), ('W_K', sb.W_K), ('W_V', sb.W_V), ('W_O', sb.W_O)]:
                    # Compute per-neuron orthogonality error
                    ortho_errors = []
                    for i in range(n_neurons):
                        W_i = W[i]  # [D, rank] or [rank, D]
                        if W_i.shape[0] > W_i.shape[1]:
                            gram = W_i.T @ W_i  # [rank, rank]
                        else:
                            gram = W_i @ W_i.T  # [rank, rank]
                        I = torch.eye(gram.shape[0], device=gram.device)
                        error = ((gram - I) ** 2).mean().item()
                        ortho_errors.append(error)

                    avg_error = sum(ortho_errors) / len(ortho_errors)
                    max_error = max(ortho_errors)
                    self.log(f"  {W_name}: mean_error={avg_error:.8f}, max_error={max_error:.8f}")

                self.log(f"\n  (Lower is better, 0 = perfect orthogonality)")

            elif is_v77:
                # v7.7: basis_qk and basis_vo (both QR initialized)
                n_basis = sb.n_basis
                I = torch.eye(n_basis, device=sb.basis_qk.device)
                B_qk = sb.basis_qk.view(n_basis, -1)
                B_vo = sb.basis_vo.view(n_basis, -1)
                gram_qk = B_qk @ B_qk.T
                gram_vo = B_vo @ B_vo.T
                ortho_qk = ((gram_qk - I) ** 2).mean().item()
                ortho_vo = ((gram_vo - I) ** 2).mean().item()
                off_diagonal_mask = ~I.bool()

                self.log(f"\n[Orthogonality Loss Components (v7.7)]")
                self.log(f"  ortho_qk = {ortho_qk:.8f}")
                self.log(f"  ortho_vo = {ortho_vo:.8f}")
                self.log(f"  total (avg) = {(ortho_qk + ortho_vo) / 2:.8f}")

                self.log(f"\n[Gram Matrix Diagnostics]")
                self.log(f"  gram_qk diagonal mean: {gram_qk.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_qk off-diag mean: {gram_qk[off_diagonal_mask].mean().item():.6f} (target: 0.0)")
                self.log(f"  gram_vo diagonal mean: {gram_vo.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_vo off-diag mean: {gram_vo[off_diagonal_mask].mean().item():.6f} (target: 0.0)")

                if ortho_vo > ortho_qk * 10:
                    self.log(f"\n  ⚠️  ortho_vo >> ortho_qk: V/O projection learning may be unstable")
                elif ortho_qk > ortho_vo * 10:
                    self.log(f"\n  ⚠️  ortho_qk >> ortho_vo: Q/K projection learning may be unstable")
            elif hasattr(sb, 'basis_up') and hasattr(sb, 'basis_down'):
                # v7.6: basis_down and basis_up
                n_basis = sb.n_basis
                I = torch.eye(n_basis, device=sb.basis_down.device)
                B_down = sb.basis_down.view(n_basis, -1)
                gram_down = B_down @ B_down.T
                ortho_down = ((gram_down - I) ** 2).mean().item()

                # basis_up orthogonality (with normalization for v7.6)
                B_up = sb.basis_up.view(n_basis, -1)
                B_up_norm = F.normalize(B_up, dim=-1)
                gram_up = B_up_norm @ B_up_norm.T
                off_diagonal_mask = ~I.bool()
                ortho_up = (gram_up[off_diagonal_mask] ** 2).mean().item()

                self.log(f"\n[Orthogonality Loss Components (v7.6)]")
                self.log(f"  ortho_down = {ortho_down:.8f}")
                self.log(f"  ortho_up   = {ortho_up:.8f}")
                self.log(f"  total (avg) = {(ortho_down + ortho_up) / 2:.8f}")

                self.log(f"\n[Gram Matrix Diagnostics]")
                self.log(f"  gram_down diagonal mean: {gram_down.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_down off-diag mean: {gram_down[off_diagonal_mask].mean().item():.6f} (target: 0.0)")
                self.log(f"  gram_up diagonal mean: {gram_up.diag().mean().item():.6f} (target: 1.0)")
                self.log(f"  gram_up off-diag mean: {gram_up[off_diagonal_mask].mean().item():.6f} (target: 0.0)")

                if ortho_up > ortho_down * 10:
                    self.log(f"\n  ⚠️  ortho_up >> ortho_down: O projection learning may be unstable")
                elif ortho_down > ortho_up * 10:
                    self.log(f"\n  ⚠️  ortho_down >> ortho_up: Q/K/V projection learning may be unstable")
            else:
                # Other versions - skip
                self.log(f"\n[Note: Orthogonality breakdown not available for this model version]")

    def log_recipe_analysis(self, model, sample_input, epoch, step=None):
        """
        Log Recipe → W_O analysis

        Supports:
        - v7.8: NeuronBank W_O (no recipe, direct neuron mixing)
        - v7.7: get_basis_o() (basis_vo.T)
        - v7.6: get_basis_up()

        Tracks:
        - Recipe/neuron weight entropy (diversity)
        - Max weight concentration
        - W_O singular values
        """
        base_model = get_underlying_model(model)

        if not hasattr(base_model, 'shared_basis'):
            return

        step_str = f" Step {step}" if step else ""
        self.log_section(f"Recipe/Neuron → W_O Analysis - Epoch {epoch}{step_str}")

        sb = base_model.shared_basis
        is_v78 = hasattr(sb, 'W_Q') and hasattr(sb, 'W_K')
        is_v77 = hasattr(sb, 'basis_qk')

        base_model.eval()
        with torch.no_grad():
            B, S = sample_input.shape
            device = sample_input.device

            # Get first layer's qkv_dynamic
            layer = base_model.layers[0]
            qkv = layer.qkv_dynamic

            # Routing
            pos = torch.arange(S, device=device).unsqueeze(0)
            x = base_model.token_emb(sample_input) + base_model.pos_emb(pos)

            scores = qkv.W_router(x)
            topk_scores, topk_idx = torch.topk(scores, qkv.k, dim=-1)
            weights = F.softmax(topk_scores, dim=-1)

            if is_v78:
                # v7.8: No recipe, analyze neuron weight distribution
                self.log(f"\n[v7.8 Neuron Weight Statistics (no recipe)]")

                # Neuron weight entropy
                entropy = (-weights * torch.log(weights + 1e-10)).sum(-1).mean()
                max_entropy = math.log(qkv.k)  # Maximum entropy for k neurons

                # Max weight
                max_weight = weights.max(-1)[0].mean()

                self.log(f"  Neuron weight entropy: {entropy.item():.4f} (max possible: {max_entropy:.4f})")
                self.log(f"  Normalized entropy: {entropy.item() / max_entropy:.4f}")
                self.log(f"  Max weight (mean): {max_weight.item():.4f}")

                if max_weight.item() > 0.8:
                    self.log(f"  ⚠️  WARNING: Neuron weights too concentrated on single neuron!")

                # W_O construction via neuron mixing
                self.log(f"\n[W_O Construction (v7.8): weighted_avg(neuron_W_O)]")
                nb = qkv.neuron_bank
                W_O_neurons = nb.get_W_O(topk_idx)  # [B, S, k, rank, D]
                weights_exp = weights.unsqueeze(-1).unsqueeze(-1)
                W_O = (W_O_neurons * weights_exp).sum(dim=2)  # [B, S, rank, D]

                # Analyze sample W_O
                W_O_sample = W_O[0, 0]  # [rank, D]
                _, S_wo, _ = torch.linalg.svd(W_O_sample)

                self.log(f"\n[Mixed W_O Singular Values (sample token)]")
                self.log(f"  Top 5: {S_wo[:5].cpu().numpy()}")
                self.log(f"  σ_max/σ_min: {(S_wo[0] / (S_wo[-1] + 1e-10)).item():.2e}")

                if S_wo[-1].item() < 1e-6:
                    self.log(f"  ⚠️  WARNING: W_O has near-zero singular values!")

            elif hasattr(qkv, 'neuron_recipe_O'):
                # v7.5/v7.6/v7.7: Recipe-based analysis
                # Recipe O
                recipe_O = qkv.neuron_recipe_O[topk_idx]
                token_recipe_O = (recipe_O * weights.unsqueeze(-1)).sum(dim=2)
                token_recipe_O = F.softmax(token_recipe_O, dim=-1)  # [B, S, n_basis]

                # Recipe entropy
                entropy = (-token_recipe_O * torch.log(token_recipe_O + 1e-10)).sum(-1).mean()
                max_entropy = math.log(qkv.n_basis)  # Maximum possible entropy

                # Max weight
                max_weight = token_recipe_O.max(-1)[0].mean()

                self.log(f"\n[Recipe_O Statistics]")
                self.log(f"  Entropy: {entropy.item():.4f} (max possible: {max_entropy:.4f})")
                self.log(f"  Normalized entropy: {entropy.item() / max_entropy:.4f}")
                self.log(f"  Max weight (mean): {max_weight.item():.4f}")

                if max_weight.item() > 0.8:
                    self.log(f"  ⚠️  WARNING: Recipe too concentrated on single basis!")

                # W_O singular values - handle v7.6 vs v7.7
                if is_v77:
                    basis_o = sb.get_basis_o()  # [n_basis, rank, D] = basis_vo.T
                    self.log(f"\n[W_O Construction (v7.7): recipe_O @ basis_vo.T]")
                else:
                    basis_o = sb.get_basis_up()  # [n_basis, rank, D]
                    self.log(f"\n[W_O Construction (v7.6): recipe_O @ basis_up]")

                W_O = torch.einsum('bsn,nrd->bsrd', token_recipe_O, basis_o)  # [B, S, rank, D]

                # Analyze first token's W_O
                W_O_sample = W_O[0, 0]  # [rank, D]
                _, S_wo, _ = torch.linalg.svd(W_O_sample)

                self.log(f"\n[W_O Singular Values (sample token)]")
                self.log(f"  Top 5: {S_wo[:5].cpu().numpy()}")
                self.log(f"  σ_max/σ_min: {(S_wo[0] / (S_wo[-1] + 1e-10)).item():.2e}")

                # Check if W_O is degenerating
                if S_wo[-1].item() < 1e-6:
                    self.log(f"  ⚠️  WARNING: W_O has near-zero singular values!")

            else:
                # Other versions - skip recipe analysis
                self.log(f"\n[Note: Recipe analysis not available for this model version]")

        base_model.train()

    def log_epoch_summary(self, model, sample_input, epoch, step=None):
        """Log all debug info for an epoch"""
        self.log_basis_stats(model, epoch, step)
        self.log_orthogonality_breakdown(model, epoch, step)
        self.log_recipe_analysis(model, sample_input, epoch, step)

    def log_post_backward(self, model, epoch, step=None):
        """Log gradient info after backward pass"""
        self.log_gradient_flow(model, epoch, step)


def get_underlying_model(model):
    """Get the underlying model from a potentially torch.compile() wrapped model"""
    # torch.compile() wraps models in OptimizedModule with _orig_mod attribute
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model


def is_v75_or_v76_model(model):
    """Robust detection of v7.5/v7.6/v7.7 models, handling torch.compile() wrapped models"""
    base_model = get_underlying_model(model)

    # Check model version attribute
    if hasattr(base_model, '__version__') and base_model.__version__ in ["7.5", "7.6", "7.7", "7.8", "7.9"]:
        return True

    # Check for qkv_dynamic attribute on layers (v7.5/v7.6/v7.7 specific structure)
    if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
        if hasattr(base_model.layers[0], 'qkv_dynamic'):
            return True

    return False


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None, log_file=None,
                orthogonality_weight=0.0, diversity_weight=0.0, load_balance_weight=0.0, debug_logger=None):
    """Train for one epoch"""
    model.train()

    # Debug: Log at start of key epochs
    debug_log_steps = {0, 100, 500}  # Steps to log gradient info

    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0
    num_batches = 0

    # Window accumulators for logging every 100 steps
    log_interval = 100
    window_loss = 0.0
    window_acc_correct = 0
    window_acc_valid = 0
    window_count = 0

    # Last neuron metrics (for epoch summary)
    last_neuron_metrics = None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)

        # Apply MLM masking
        if tokenizer is not None:
            input_ids, labels = apply_mlm_masking(input_ids, tokenizer, MLM_CONFIG)
        else:
            labels = input_ids.clone()

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # Get underlying model for attribute checks (handles torch.compile)
                base_model = get_underlying_model(model)

                # v7.0: Use model's get_loss method (handles diversity & load balance)
                if hasattr(base_model, 'get_loss') and orthogonality_weight == 0:
                    loss, loss_dict, logits = model.get_loss(
                        input_ids, labels,
                        diversity_weight=diversity_weight,
                        load_balance_weight=load_balance_weight
                    )
                else:
                    # v7.5+: Dynamic Q/K/V with routing
                    # v6.0: compatibility
                    if is_v75_or_v76_model(model):
                        # v7.5/v7.6: Get routing info for load balance loss
                        if load_balance_weight > 0:
                            ce_loss, logits, routing_infos = model(input_ids, labels, return_routing_info=True)
                        else:
                            ce_loss, logits = model(input_ids, labels)
                            routing_infos = None

                        # Basis orthogonality loss (v7.5/v7.6)
                        if orthogonality_weight > 0:
                            orth_loss = base_model.orthogonality_loss()
                        else:
                            orth_loss = 0.0

                        # Recipe diversity loss (v7.6 only - built-in method)
                        if diversity_weight > 0 and hasattr(base_model, 'recipe_diversity_loss'):
                            diversity_loss = base_model.recipe_diversity_loss()
                        else:
                            diversity_loss = 0.0
                    elif orthogonality_weight > 0:
                        logits, losses = model(input_ids, return_losses=True)
                        orth_loss = losses['orth_total']
                        ce_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                        diversity_loss = 0.0
                        routing_infos = None
                    else:
                        logits = model(input_ids)
                        orth_loss = 0.0
                        diversity_loss = 0.0
                        routing_infos = None

                        # Cross-entropy loss
                        B, S, V = logits.shape
                        ce_loss = F.cross_entropy(
                            logits.view(B * S, V),
                            labels.view(B * S),
                            ignore_index=-100
                        )

                    # Recipe diversity loss (v7.0-v7.4)
                    if diversity_weight > 0 and hasattr(base_model.layers[0], 'ffn') and hasattr(base_model.layers[0].ffn, 'neuron_recipe'):
                        for layer in base_model.layers:
                            recipe = layer.ffn.neuron_recipe
                            recipe_norm = F.softmax(recipe, dim=-1)
                            recipe_normalized = F.normalize(recipe_norm, dim=-1)
                            similarity = torch.mm(recipe_normalized, recipe_normalized.T)
                            mask = 1 - torch.eye(base_model.n_neurons, device=similarity.device)
                            avg_similarity = (similarity * mask).sum() / mask.sum()
                            diversity_loss += avg_similarity
                        diversity_loss = diversity_loss / len(base_model.layers)

                    # Load balance loss (v7.5+)
                    lb_loss = 0.0
                    if load_balance_weight > 0 and routing_infos is not None:
                        for routing_info in routing_infos:
                            neuron_indices = routing_info['neuron_indices']  # [B, S, k]
                            # Count neuron usage
                            counts = torch.bincount(neuron_indices.reshape(-1), minlength=base_model.n_neurons)
                            freq = counts.float() / (counts.sum() + 1e-8)
                            # L2 distance from uniform distribution
                            uniform = 1.0 / base_model.n_neurons
                            lb_loss += ((freq - uniform) ** 2).sum() * base_model.n_neurons
                        lb_loss = lb_loss / len(routing_infos)

                    # Total loss
                    loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + load_balance_weight * lb_loss

            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Debug: Log gradient flow at specific steps
            if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                debug_logger.log_post_backward(model, epoch, step)

            scaler.step(optimizer)
            scaler.update()
        else:
            # Get underlying model for attribute checks (handles torch.compile)
            base_model = get_underlying_model(model)

            # v7.0: Use model's get_loss method (handles diversity & load balance)
            if hasattr(base_model, 'get_loss') and orthogonality_weight == 0:
                loss, loss_dict, logits = model.get_loss(
                    input_ids, labels,
                    diversity_weight=diversity_weight,
                    load_balance_weight=load_balance_weight
                )
            else:
                # v7.5+: Dynamic Q/K/V with routing
                # v6.0: compatibility
                if is_v75_or_v76_model(model):
                    # v7.5/v7.6: Get routing info for load balance loss
                    if load_balance_weight > 0:
                        ce_loss, logits, routing_infos = model(input_ids, labels, return_routing_info=True)
                    else:
                        ce_loss, logits = model(input_ids, labels)
                        routing_infos = None

                    # Basis orthogonality loss (v7.5/v7.6)
                    if orthogonality_weight > 0:
                        orth_loss = base_model.orthogonality_loss()
                    else:
                        orth_loss = 0.0

                    # Recipe diversity loss (v7.6 only - built-in method)
                    if diversity_weight > 0 and hasattr(base_model, 'recipe_diversity_loss'):
                        diversity_loss = base_model.recipe_diversity_loss()
                    else:
                        diversity_loss = 0.0
                elif orthogonality_weight > 0:
                    logits, losses = model(input_ids, return_losses=True)
                    orth_loss = losses['orth_total']
                    ce_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    diversity_loss = 0.0
                    routing_infos = None
                else:
                    logits = model(input_ids)
                    orth_loss = 0.0
                    diversity_loss = 0.0
                    routing_infos = None

                    # Cross-entropy loss
                    B, S, V = logits.shape
                    ce_loss = F.cross_entropy(
                        logits.view(B * S, V),
                        labels.view(B * S),
                        ignore_index=-100
                    )

                # Recipe diversity loss (v7.0-v7.4)
                if diversity_weight > 0 and hasattr(base_model.layers[0], 'ffn') and hasattr(base_model.layers[0].ffn, 'neuron_recipe'):
                    for layer in base_model.layers:
                        recipe = layer.ffn.neuron_recipe
                        recipe_norm = F.softmax(recipe, dim=-1)
                        recipe_normalized = F.normalize(recipe_norm, dim=-1)
                        similarity = torch.mm(recipe_normalized, recipe_normalized.T)
                        mask = 1 - torch.eye(base_model.n_neurons, device=similarity.device)
                        avg_similarity = (similarity * mask).sum() / mask.sum()
                        diversity_loss += avg_similarity
                    diversity_loss = diversity_loss / len(base_model.layers)

                # Load balance loss (v7.5+)
                lb_loss = 0.0
                if load_balance_weight > 0 and routing_infos is not None:
                    for routing_info in routing_infos:
                        neuron_indices = routing_info['neuron_indices']  # [B, S, k]
                        # Count neuron usage
                        counts = torch.bincount(neuron_indices.reshape(-1), minlength=base_model.n_neurons)
                        freq = counts.float() / (counts.sum() + 1e-8)
                        # L2 distance from uniform distribution
                        uniform = 1.0 / base_model.n_neurons
                        lb_loss += ((freq - uniform) ** 2).sum() * base_model.n_neurons
                    lb_loss = lb_loss / len(routing_infos)

                # Total loss
                loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + load_balance_weight * lb_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Debug: Log gradient flow at specific steps
            if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                debug_logger.log_post_backward(model, epoch, step)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Accuracy calculation (only valid tokens)
        predictions = logits.argmax(dim=-1)
        valid_mask = (labels != -100)
        correct_predictions = (predictions == labels) & valid_mask

        correct = correct_predictions.sum().item()
        valid_tokens = valid_mask.sum().item()

        total_correct += correct
        total_valid_tokens += valid_tokens

        # Track total loss
        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        num_batches += 1
        step_acc = correct / valid_tokens if valid_tokens > 0 else 0.0
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{step_acc:.4f}"
        })

        # Accumulate for window logging
        window_loss += loss.item()
        window_acc_correct += correct
        window_acc_valid += valid_tokens
        window_count += 1

        # Log aggregated metrics every 100 steps
        if log_file and (step + 1) % log_interval == 0:
            avg_window_loss = window_loss / window_count
            avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

            with open(log_file, 'a') as f:
                f.write(f"epoch={epoch},step={step+1},loss={avg_window_loss:.6f},"
                       f"acc={avg_window_acc:.6f}\n")

            # Collect neuron metrics
            model.eval()
            with torch.no_grad():
                try:
                    _, neuron_indices = model(input_ids, return_activations=True)
                    neuron_metrics = compute_training_metrics(model, neuron_indices, device)
                    last_neuron_metrics = neuron_metrics

                    # Log neuron metrics to file
                    with open(log_file, 'a') as f:
                        f.write(f"METRICS,{epoch},{step+1},"
                               f"avg_usage={neuron_metrics['avg_usage']:.4f},"
                               f"avg_gini={neuron_metrics['avg_gini']:.4f},"
                               f"avg_entropy={neuron_metrics['avg_entropy']:.4f},"
                               f"avg_top10={neuron_metrics['avg_top10']:.4f},"
                               f"avg_top50={neuron_metrics['avg_top50']:.4f}")
                        # Add per-layer details
                        for i in range(len(neuron_indices)):
                            f.write(f",L{i}_usage={neuron_metrics[f'L{i}_usage']:.4f},"
                                   f"L{i}_gini={neuron_metrics[f'L{i}_gini']:.4f},"
                                   f"L{i}_entropy={neuron_metrics[f'L{i}_entropy']:.4f},"
                                   f"L{i}_top10={neuron_metrics[f'L{i}_top10']:.4f},"
                                   f"L{i}_top50={neuron_metrics[f'L{i}_top50']:.4f}")
                        f.write("\n")
                except Exception as e:
                    # If metrics collection fails, continue training
                    pass
            model.train()

            # Reset window
            window_loss = 0.0
            window_acc_correct = 0
            window_acc_valid = 0
            window_count = 0

    # Log remaining steps at end of epoch
    if log_file and window_count > 0:
        avg_window_loss = window_loss / window_count
        avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

        with open(log_file, 'a') as f:
            f.write(f"epoch={epoch},step={num_batches},loss={avg_window_loss:.6f},"
                   f"acc={avg_window_acc:.6f}\n")

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

    return avg_loss, avg_acc, last_neuron_metrics


def evaluate(model, dataloader, device, args, tokenizer=None):
    """Evaluate model with MLM masking"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)

            # Apply same MLM masking as training
            if tokenizer is not None:
                masked_input_ids, labels = apply_mlm_masking(input_ids, tokenizer)
            else:
                masked_input_ids = input_ids
                labels = input_ids.clone()

            logits = model(masked_input_ids)

            B, S, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * S, V),
                labels.view(B * S),
                ignore_index=-100
            )

            # Accuracy calculation
            predictions = logits.argmax(dim=-1)
            valid_mask = (labels != -100)
            correct_predictions = (predictions == labels) & valid_mask

            correct = correct_predictions.sum().item()
            valid_tokens = valid_mask.sum().item()

            total_correct += correct
            total_valid_tokens += valid_tokens

            batch_size, seq_len = input_ids.shape
            num_tokens = batch_size * seq_len
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    return avg_loss, avg_acc


def analyze_activations(model, input_ids, device):
    """Dynamic Neuron Transformer activation pattern analysis (v6.0+)"""
    model.eval()
    base_model = get_underlying_model(model)

    with torch.no_grad():
        # v7.5/v7.6: use return_routing_info, v6.0: use return_activations
        if is_v75_or_v76_model(model):
            logits, routing_infos = model(input_ids, return_routing_info=True)
            # Extract neuron_indices from routing_infos
            all_selected = [info['neuron_indices'] for info in routing_infos]
        else:
            _, all_selected = model(input_ids, return_activations=True)

    stats = {}
    for layer_idx, selected_idx in enumerate(all_selected):
        # selected_idx: [B, S, k]
        unique_neurons = torch.unique(selected_idx).numel()

        # Get total neurons from model
        layer = base_model.layers[layer_idx]

        # v7.5/v7.6: qkv_dynamic.n_neurons, v7.0: ffn.n_neurons, v6.0: router
        if hasattr(layer, 'qkv_dynamic') and hasattr(layer.qkv_dynamic, 'n_neurons'):
            total_neurons = layer.qkv_dynamic.n_neurons
        elif hasattr(layer, 'ffn') and hasattr(layer.ffn, 'n_neurons'):
            total_neurons = layer.ffn.n_neurons
        elif hasattr(layer, 'router') and hasattr(layer.router, 'n_neurons'):
            total_neurons = layer.router.n_neurons
        elif hasattr(layer, 'neuron_router') and hasattr(layer.neuron_router, 'n_neurons'):
            total_neurons = layer.neuron_router.n_neurons
        else:
            total_neurons = base_model.n_neurons

        usage_ratio = unique_neurons / total_neurons

        stats[f'layer_{layer_idx}'] = {
            'unique_neurons': unique_neurons,
            'total_neurons': total_neurons,
            'usage_ratio': usage_ratio,
            'k': selected_idx.shape[-1],
        }

    return stats


def compute_training_metrics(model, neuron_indices, device):
    """Compute detailed neuron usage metrics during training

    Args:
        model: DAWN model
        neuron_indices: List of [B, S, k] tensors (one per layer)
        device: torch device

    Returns:
        metrics: Dict with per-layer and aggregate metrics
    """
    metrics = {}

    # Get n_neurons from model
    if hasattr(model, '_orig_mod'):
        n_neurons = model._orig_mod.n_neurons
    else:
        n_neurons = model.n_neurons

    layer_usage = []
    layer_gini = []
    layer_top10 = []
    layer_top50 = []
    layer_entropy = []

    for layer_idx, selected_idx in enumerate(neuron_indices):
        # selected_idx: [B, S, k]
        flat_idx = selected_idx.reshape(-1)
        total_selections = flat_idx.numel()

        # 1. Usage rate (unique neurons used / total neurons)
        unique_neurons = torch.unique(flat_idx).numel()
        usage_rate = unique_neurons / n_neurons
        layer_usage.append(usage_rate)

        # 2. Selection frequency distribution
        counts = torch.bincount(flat_idx, minlength=n_neurons).float()
        freq = counts / (counts.sum() + 1e-10)

        # 3. Gini coefficient (0 = perfect equality, 1 = maximum inequality)
        sorted_counts = torch.sort(counts)[0]
        n = len(sorted_counts)
        index = torch.arange(1, n + 1, device=device, dtype=torch.float32)
        gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * sorted_counts.sum() + 1e-10)
        layer_gini.append(gini.item())

        # 4. Top-K concentration
        if n_neurons >= 10:
            top10 = torch.topk(counts, 10).values.sum() / (counts.sum() + 1e-10)
            layer_top10.append(top10.item())
        else:
            layer_top10.append(1.0)

        if n_neurons >= 50:
            top50 = torch.topk(counts, 50).values.sum() / (counts.sum() + 1e-10)
            layer_top50.append(top50.item())
        else:
            layer_top50.append(1.0)

        # 5. Entropy (higher = more uniform distribution)
        # Normalize to [0, 1] by dividing by log(n_neurons)
        entropy = -(freq * torch.log(freq + 1e-10)).sum()
        normalized_entropy = entropy / (torch.log(torch.tensor(n_neurons, dtype=torch.float32)) + 1e-10)
        layer_entropy.append(normalized_entropy.item())

        # Per-layer metrics
        metrics[f'L{layer_idx}_usage'] = usage_rate
        metrics[f'L{layer_idx}_gini'] = gini.item()
        metrics[f'L{layer_idx}_top10'] = layer_top10[-1]
        metrics[f'L{layer_idx}_top50'] = layer_top50[-1]
        metrics[f'L{layer_idx}_entropy'] = normalized_entropy.item()

    # Aggregate metrics
    metrics['avg_usage'] = sum(layer_usage) / len(layer_usage)
    metrics['avg_gini'] = sum(layer_gini) / len(layer_gini)
    metrics['avg_top10'] = sum(layer_top10) / len(layer_top10)
    metrics['avg_top50'] = sum(layer_top50) / len(layer_top50)
    metrics['avg_entropy'] = sum(layer_entropy) / len(layer_entropy)

    return metrics


def load_config(config_path):
    """Load config from YAML file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train DAWN (Dynamic Architecture With Neurons)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint folder to resume from (e.g., checkpoints/run_20240101_120000_1234)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Start training from scratch (disable auto-resume)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging to debug.txt (basis_up analysis, gradients, etc.)')
    cli_args = parser.parse_args()

    # Load config
    config_path = Path(PROJECT_ROOT) / cli_args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # Create args namespace from config
    class Args:
        pass
    args = Args()

    # Model (Dynamic Neuron Transformer)
    args.model_version = cfg['model'].get('model_version', '7.1')  # Default to latest
    args.d_model = cfg['model'].get('d_model', 512)
    args.n_layers = cfg['model'].get('n_layers', 6)
    args.n_heads = cfg['model'].get('n_heads', 8)
    args.n_neurons = cfg['model'].get('n_neurons', 1024)
    args.n_patterns = cfg['model'].get('n_patterns', 512)

    # Backward compatibility: neuron_k (new) vs k (old)
    args.neuron_k = cfg['model'].get('neuron_k', cfg['model'].get('k', 8))
    args.k = args.neuron_k  # Keep k for backward compatibility
    args.pattern_k = cfg['model'].get('pattern_k', 16)

    args.d_ff = cfg['model'].get('d_ff', None)  # Auto-calculate if None
    args.max_seq_len = cfg['model'].get('max_seq_len', 2048)
    args.dropout = cfg['model'].get('dropout', 0.1)
    args.pattern_dropout = cfg['model'].get('pattern_dropout', 0.0)

    # v6.0: Basis FFN parameters
    args.neuron_rank = cfg['model'].get('neuron_rank', None)  # v6.0: not used anymore (backward compat)
    args.n_basis = cfg['model'].get('n_basis', 8)
    args.basis_rank = cfg['model'].get('basis_rank', 64)
    args.mod_rank = cfg['model'].get('mod_rank', None)  # v5.0 compatibility (ignored)
    args.router_temperature = cfg['model'].get('router_temperature', None)  # v6.0 only (v7.0 ignores)

    # v7.9 NeuronCircuit parameters
    args.n_input = cfg['model'].get('n_input', 8)
    args.n_process = cfg['model'].get('n_process', 32)
    args.n_output = cfg['model'].get('n_output', 8)
    args.process_k = cfg['model'].get('process_k', 3)
    args.use_soft_selection = cfg['model'].get('use_soft_selection', True)

    # v8.0 KnowledgeNeurons parameters
    args.n_knowledge = cfg['model'].get('n_knowledge', 64)
    args.knowledge_k = cfg['model'].get('knowledge_k', 8)

    # Training
    args.batch_size = cfg['training']['batch_size']
    args.num_epochs = cfg['training']['num_epochs']
    args.lr = cfg['training']['lr']
    args.weight_decay = cfg['training']['weight_decay']
    args.warmup_epochs = cfg['training'].get('warmup_epochs', 1)

    # Regularization weights
    args.orthogonality_weight = cfg['training'].get('orthogonality_weight', 0.0)  # v6.0 compat
    args.diversity_weight = cfg['training'].get('diversity_weight', 0.0)  # v7.0: recipe diversity
    args.load_balance_weight = cfg['training'].get('load_balance_weight', 0.0)  # v7.0: load balance

    # Other
    args.use_amp = cfg.get('use_amp', True)
    args.checkpoint_dir = cfg.get('checkpoint_dir', 'checkpoints')
    args.log_dir = cfg.get('log_dir', 'logs')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    base_checkpoint_dir = Path(args.checkpoint_dir)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint loading logic
    latest_best_checkpoint = None
    checkpoint_dir = None

    if cli_args.resume:
        # Explicit resume from folder - use existing folder
        resume_folder = Path(cli_args.resume)
        if not resume_folder.is_absolute():
            resume_folder = Path(args.checkpoint_dir) / resume_folder.name

        best_ckpt = resume_folder / 'best_model.pt'
        if best_ckpt.exists():
            latest_best_checkpoint = best_ckpt
            checkpoint_dir = resume_folder  # Use existing folder
            print(f"\n✓ Resuming from: {latest_best_checkpoint}")
            print(f"✓ Continuing in same folder: {checkpoint_dir}")
        else:
            print(f"\n⚠️  Warning: Checkpoint not found at {best_ckpt}")
            print(f"    Starting from scratch instead.")

    elif not cli_args.from_scratch:
        # Auto-resume: find latest checkpoint and use its folder
        run_folders = sorted([
            d for d in base_checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ], reverse=True)

        if run_folders:
            latest_folder = run_folders[0]
            best_ckpt = latest_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                checkpoint_dir = latest_folder  # Use existing folder
                print(f"\n✓ Auto-resume: Found latest checkpoint: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")

    if cli_args.from_scratch:
        print(f"\n✓ Starting from scratch (--from-scratch)")

    # Create new run folder only if not resuming
    if checkpoint_dir is None:
        import random
        from datetime import timezone, timedelta
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime('%Y%m%d_%H%M%S')
        random_suffix = random.randint(1000, 9999)
        version = cfg['model'].get('model_version', DAWN.__version__)
        run_name = f"run_v{version}_{timestamp}_{random_suffix}"
        checkpoint_dir = base_checkpoint_dir / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created new run folder: {checkpoint_dir}")

        # Save config for new runs (add model version if not present)
        if 'model_version' not in cfg['model']:
            cfg['model']['model_version'] = DAWN.__version__
        with open(checkpoint_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

    log_dir = checkpoint_dir
    print(f"Run folder: {checkpoint_dir}")

    # ============================================================
    # STEP 1: Load checkpoint config FIRST (before logging)
    # ============================================================
    resume_checkpoint = None
    if latest_best_checkpoint:
        resume_checkpoint = latest_best_checkpoint

    checkpoint_config = None
    checkpoint_training_config = None
    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\n📌 Resuming from checkpoint: {resume_checkpoint}")

        # Try config.json first, then checkpoint file
        config_json_path = resume_checkpoint.parent / 'config.json'
        if config_json_path.exists():
            with open(config_json_path, 'r') as f:
                saved_cfg = json.load(f)
                checkpoint_config = saved_cfg.get('model')
                checkpoint_training_config = saved_cfg.get('training')
                print(f"✅ Loaded config.json from checkpoint folder")
        else:
            temp_checkpoint = torch.load(resume_checkpoint, map_location='cpu')
            if 'config' in temp_checkpoint:
                checkpoint_config = temp_checkpoint['config']
                print(f"✅ Loaded model config from checkpoint file")
            del temp_checkpoint

    # Update args from checkpoint config (if resuming)
    if checkpoint_config:
        args.model_version = checkpoint_config.get('model_version', args.model_version)
        args.d_model = checkpoint_config.get('d_model', args.d_model)
        args.n_layers = checkpoint_config.get('n_layers', args.n_layers)
        args.n_heads = checkpoint_config.get('n_heads', args.n_heads)
        args.n_neurons = checkpoint_config.get('n_neurons', args.n_neurons)
        args.k = checkpoint_config.get('neuron_k', args.k)
        args.n_basis = checkpoint_config.get('n_basis', args.n_basis)
        args.basis_rank = checkpoint_config.get('basis_rank', args.basis_rank)
        args.d_ff = checkpoint_config.get('d_ff', args.d_ff)
        args.max_seq_len = checkpoint_config.get('max_seq_len', args.max_seq_len)
        args.dropout = checkpoint_config.get('dropout', args.dropout)

        if checkpoint_training_config:
            args.orthogonality_weight = checkpoint_training_config.get('orthogonality_weight', args.orthogonality_weight)
            args.diversity_weight = checkpoint_training_config.get('diversity_weight', args.diversity_weight)
            args.load_balance_weight = checkpoint_training_config.get('load_balance_weight', args.load_balance_weight)

        print(f"   → Updated args from checkpoint config (v{args.model_version})")

    # ============================================================
    # STEP 2: Print configuration summary (using updated args)
    # ============================================================
    print(f"\n{'='*60}")
    model_version = getattr(args, 'model_version', '7.1')
    if model_version == 'baseline':
        print(f"Vanilla Transformer Baseline Training")
    else:
        print(f"DAWN (Dynamic Neuron Transformer) Training")
    print(f"{'='*60}")
    print(f"\nModel version: {model_version}")
    print(f"\nModel: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")

    if model_version != 'baseline':
        print(f"Neurons: n_neurons={args.n_neurons}, neuron_k={args.k}")

        if model_version in ["8.0", "8.1"]:
            # v8.0/v8.1: SharedNeurons + NeuronMemory (FFN 대체)
            rank = getattr(args, 'rank', args.basis_rank)
            n_input = getattr(args, 'n_input', 8)
            n_process = getattr(args, 'n_process', 32)
            n_output = getattr(args, 'n_output', 8)
            process_k = getattr(args, 'process_k', 3)
            n_knowledge = getattr(args, 'n_knowledge', 64)
            knowledge_k = getattr(args, 'knowledge_k', 8)
            print(f"SharedNeurons + NeuronMemory (v{model_version}): rank={rank}")
            print(f"  TransformNeurons (Shared):")
            print(f"    - InputNeuron: {n_input} × {args.d_model} × {rank}")
            print(f"    - ProcessNeuron: {n_process} × {rank} (Householder vectors)")
            print(f"    - OutputNeuron: {n_output} × {rank} × {args.d_model}")
            print(f"    - Process top-k: {process_k}")
            print(f"  KnowledgeNeurons (Shared):")
            print(f"    - K: {n_knowledge} × {rank}")
            print(f"    - V: {n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {knowledge_k}")
            if args.orthogonality_weight > 0:
                print(f"  - Orthogonality Loss (orth_weight={args.orthogonality_weight})")
        elif model_version == "7.9":
            # v7.9: NeuronCircuit with Householder Transformations
            rank = getattr(args, 'rank', args.basis_rank)
            n_input = getattr(args, 'n_input', 8)
            n_process = getattr(args, 'n_process', 32)
            n_output = getattr(args, 'n_output', 8)
            process_k = getattr(args, 'process_k', 3)
            print(f"NeuronCircuit with Householder (v7.9): rank={rank}")
            print(f"  - InputNeuron: {n_input} × {args.d_model} × {rank}")
            print(f"  - ProcessNeuron: {n_process} × {rank} (Householder vectors)")
            print(f"  - OutputNeuron: {n_output} × {rank} × {args.d_model}")
            print(f"  - Process top-k: {process_k}")
            if args.orthogonality_weight > 0:
                print(f"  - Orthogonality Loss (orth_weight={args.orthogonality_weight})")
        elif model_version == "7.8":
            # v7.8: Independent Neuron W_Q/K/V/O (No Basis Mixing)
            rank = getattr(args, 'rank', args.basis_rank)  # v7.8 uses 'rank', fallback to basis_rank
            print(f"Independent Neuron Projections (v7.8): rank={rank}")
            print(f"  - NeuronBank: {args.n_neurons} neurons × W_Q/K/V/O each")
            print(f"  - No basis mixing (condition number stability)")
            if args.orthogonality_weight > 0:
                print(f"  - Neuron W Orthogonality Loss (orth_weight={args.orthogonality_weight})")
        elif model_version == "7.7":
            # v7.7: QK/VO Basis Separation
            print(f"QK/VO Basis Separation (v7.7): n_basis={args.n_basis}, basis_rank={args.basis_rank}")
            print(f"  - basis_qk: Q/K share (gradient 2x)")
            print(f"  - basis_vo: V/O share (O uses transpose)")
            if args.orthogonality_weight > 0:
                print(f"  - Orthogonality Loss (orth_weight={args.orthogonality_weight})")
        elif model_version == "7.6":
            # v7.6: Independent O Basis
            print(f"Independent O Basis (v7.6): n_basis={args.n_basis}, basis_rank={args.basis_rank}")
            print(f"  - basis_down: Q/K/V projection")
            print(f"  - basis_up: O projection (independent)")
            if args.orthogonality_weight > 0:
                print(f"  - Orthogonality Loss (orth_weight={args.orthogonality_weight})")
        elif model_version == "7.5":
            print(f"Dynamic Q/K/V/O Generation (v7.5): n_basis={args.n_basis}, basis_rank={args.basis_rank}")
            if args.orthogonality_weight > 0:
                print(f"  - Learnable Basis (orth_weight={args.orthogonality_weight})")
            else:
                print(f"  - Fixed Basis")
            if args.load_balance_weight > 0:
                print(f"  - Load Balance Loss (lb_weight={args.load_balance_weight})")
            print(f"  - Simple Router (x only)")
        elif model_version == "7.4":
            print(f"TT Karcher Mean FFN: n_basis={args.n_basis}, basis_rank={args.basis_rank}")
        elif model_version == "7.2":
            print(f"FFN: Standard FFN with Neuron Augmentation (d_ff={args.d_ff})")
        else:
            if model_version == "7.1":
                basis_note = "v7.1: Symmetric Basis FFN"
            elif model_version == "7.0":
                basis_note = "v7.0: Fixed Orthogonal Basis"
            else:
                basis_note = "v6.0: Learned Basis"
            print(f"Basis FFN ({basis_note}): n_basis={args.n_basis}, basis_rank={args.basis_rank}")
    else:
        print(f"Standard FFN: d_ff={args.d_ff}")

    print(f"Training: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

    # Regularization summary
    reg_parts = []
    if args.orthogonality_weight > 0:
        reg_parts.append(f"orth={args.orthogonality_weight}")
    if args.diversity_weight > 0:
        reg_parts.append(f"div={args.diversity_weight}")
    if args.load_balance_weight > 0:
        reg_parts.append(f"lb={args.load_balance_weight}")
    if reg_parts:
        print(f"Regularization: {', '.join(reg_parts)}")

    # ============================================================
    # STEP 3: Load data
    # ============================================================
    print(f"\n{'='*60}")
    print("Loading data...")
    print(f"{'='*60}")
    train_loader, val_loader, tokenizer = load_data(
        data_config=cfg['data'],
        max_length=args.max_seq_len,
        batch_size=args.batch_size
    )

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ============================================================
    # STEP 4: Create model (using args)
    # ============================================================
    print(f"\n{'='*60}")
    print("Creating DAWN model...")
    print(f"{'='*60}")

    # Build model kwargs from args (already updated from checkpoint if resuming)
    model_version = getattr(args, 'model_version', '7.1')

    # Common parameters
    model_kwargs = {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'max_seq_len': args.max_seq_len,
        'dropout': args.dropout,
    }

    # Add DAWN-specific parameters
    if model_version != 'baseline':
        model_kwargs.update({
            'n_neurons': args.n_neurons,
            'neuron_k': args.k,
            'n_basis': args.n_basis,
            'basis_rank': args.basis_rank,
        })

        # v6.0 compatibility (ignored by v7.0+)
        if args.router_temperature is not None:
            model_kwargs['router_temperature'] = args.router_temperature
        if args.neuron_rank is not None:
            model_kwargs['neuron_rank'] = args.neuron_rank
        if args.mod_rank is not None:
            model_kwargs['mod_rank'] = args.mod_rank

        # v7.9 NeuronCircuit parameters
        if model_version == '7.9':
            model_kwargs.update({
                'n_input': args.n_input,
                'n_process': args.n_process,
                'n_output': args.n_output,
                'process_k': args.process_k,
                'use_soft_selection': args.use_soft_selection,
                'rank': args.basis_rank,  # v7.9 uses 'rank' instead of 'basis_rank'
            })

        # v8.0/v8.1 SharedNeurons + NeuronMemory parameters
        if model_version in ['8.0', '8.1']:
            model_kwargs.update({
                'n_input': args.n_input,
                'n_process': args.n_process,
                'n_output': args.n_output,
                'process_k': args.process_k,
                'n_knowledge': getattr(args, 'n_knowledge', 64),
                'knowledge_k': getattr(args, 'knowledge_k', 8),
                'rank': args.basis_rank,  # v8.0/v8.1 uses 'rank' instead of 'basis_rank'
            })

    # Create model
    if model_version in ['8.1', '8.0', '7.9', '7.8', '7.7', '7.6', '7.5', '7.4', '7.2', '7.1', '7.0', '6.0', 'baseline']:
        model = create_model_by_version(model_version, model_kwargs)
    else:
        model = DAWN(**model_kwargs)

    model = model.to(device)
    print(f"✅ Model created: v{getattr(model, '__version__', model_version)}")

    # PyTorch 2.0+ compilation for speed boost
    if hasattr(torch, 'compile'):
        print(f"\nCompiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print(f"  Model compiled successfully!")

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of layers: {args.n_layers}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )

    # Warmup + Cosine scheduler
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.num_epochs * len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=args.lr * 0.1
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    if args.use_amp:
        print(f"\nUsing Automatic Mixed Precision (AMP)")

    # Resume from checkpoint (weights loading)
    start_epoch = 1
    best_val_loss = float('inf')

    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\n{'='*60}")
        print("Loading checkpoint weights...")
        print(f"{'='*60}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        # Version check (should match if we used checkpoint config)
        checkpoint_version = checkpoint.get('model_version', 'unknown')
        current_version = getattr(model, '__version__', 'unknown')

        if checkpoint_config:
            # We used checkpoint config, so architecture should match perfectly
            print(f"✅ Architecture matches (both v{current_version})")
        else:
            # We used YAML config, check for mismatch
            print(f"📌 Checkpoint version: {checkpoint_version}")
            print(f"📌 Current model version: {current_version}")

            if checkpoint_version != current_version and checkpoint_version != 'unknown':
                print(f"\n⚠️  Version mismatch detected!")
                print(f"   Checkpoint: v{checkpoint_version} → Current: v{current_version}")
                print(f"   Will attempt partial loading (architecture-compatible parameters only)")

        # Load model state (strict if using checkpoint config)
        use_strict = checkpoint_config is not None
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], strict=use_strict
        )

        if use_strict:
            print(f"✅ Weights loaded successfully (strict mode)")

        # Categorize missing keys (only when not using strict mode)
        if not use_strict and missing_keys:
            # v5.0 new parameters
            v5_new_params = [k for k in missing_keys if any(x in k for x in
                ['neuron_A', 'neuron_B', 'basis_A', 'basis_B',
                 'neuron_coef', 'token_mod'])]
            # v4.5 old parameters that are now missing
            v4_old_params = [k for k in missing_keys if any(x in k for x in
                ['pattern_queries', 'pattern_up', 'neuron_q', 'neuron_k', 'neuron_v'])]
            other_missing = [k for k in missing_keys if k not in v5_new_params and k not in v4_old_params]

            if v5_new_params:
                print(f"\n✨ v5.0 new parameters (randomly initialized): {len(v5_new_params)}")
                print(f"   - Low-rank neurons, basis FFN, token modulation")
            if other_missing:
                print(f"\n⚠️  Other missing keys: {len(other_missing)}")
                if len(other_missing) <= 5:
                    for k in other_missing:
                        print(f"      - {k}")

        if not use_strict and unexpected_keys:
            # v4.5 parameters not in v5.0
            v4_deprecated = [k for k in unexpected_keys if any(x in k for x in
                ['pattern_queries', 'pattern_up', 'neuron_q', 'neuron_k', 'neuron_v',
                 'neuron_interaction', 'up_base', 'path_proj'])]
            other_unexpected = [k for k in unexpected_keys if k not in v4_deprecated]

            if v4_deprecated:
                print(f"\n♻️  v4.5 deprecated parameters (ignored): {len(v4_deprecated)}")
            if other_unexpected:
                print(f"\n⚠️  Other unexpected keys: {len(other_unexpected)}")
                if len(other_unexpected) <= 5:
                    for k in other_unexpected:
                        print(f"      - {k}")

        if not missing_keys and not unexpected_keys:
            print("\n✅ Perfect match! All parameters loaded successfully!")

        # Try to load optimizer state even with version mismatch
        try:
            if checkpoint_version != DAWN.__version__ and checkpoint_version != 'unknown':
                print(f"\n🔄 Loading optimizer state (cross-version)...")
                print(f"   New parameters will use default optimizer settings")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))

            print(f"✅ Optimizer/scheduler loaded successfully")
            print(f"✅ Resuming from epoch {start_epoch} (best val loss: {best_val_loss:.4f})")

        except Exception as e:
            print(f"\n⚠️  Could not load optimizer state: {str(e)[:100]}")
            print(f"   Starting with fresh optimizer (model weights preserved)")
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
            print(f"   Epoch count: {start_epoch}, Best val loss: {best_val_loss:.4f}")

    else:
        print(f"\n🆕 Starting fresh training (no checkpoint found)")

    # Checkpoint & Monitor
    ckpt_manager = CheckpointManager(str(checkpoint_dir), keep_best_n=3)
    monitor = TrainingMonitor(str(log_dir))

    # Training log file (append mode if resuming)
    training_log_file = checkpoint_dir / "training_log.txt"

    # Open in append mode if resuming, write mode if new
    log_mode = 'a' if latest_best_checkpoint else 'w'
    if log_mode == 'w':
        with open(training_log_file, 'w') as f:
            f.write("# Training Log\n")
            f.write("# Step logs: epoch,step,loss,acc\n")
            f.write("# Epoch summaries: EPOCH,epoch,train_loss,train_acc,val_loss,val_acc,lr,time\n")
            f.write("\n")
    else:
        # Append separator for resumed training
        with open(training_log_file, 'a') as f:
            f.write(f"\n# === Resumed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    # Debug logger (if --debug flag is set)
    debug_logger = None
    if cli_args.debug:
        debug_log_file = checkpoint_dir / "debug.txt"
        debug_logger = DebugLogger(str(debug_log_file))
        print(f"\n🔍 Debug mode enabled")
        print(f"  Debug log: {debug_log_file}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"  Training log: {training_log_file}")
    print(f"{'='*60}")

    # Get sample batch for debug logging
    sample_batch_for_debug = None
    if debug_logger:
        sample_batch_for_debug = next(iter(train_loader))['input_ids'][:1].to(device)
        # Log initial state (before any training)
        debug_logger.log_section(f"Initial State (Before Training)")
        debug_logger.log_epoch_summary(model, sample_batch_for_debug, epoch=0)

    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, neuron_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args,
            scaler, tokenizer, log_file=str(training_log_file),
            orthogonality_weight=args.orthogonality_weight,
            diversity_weight=args.diversity_weight,
            load_balance_weight=args.load_balance_weight,
            debug_logger=debug_logger
        )

        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, args, tokenizer)

        epoch_time = time.time() - epoch_start

        # Log
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        monitor.log_epoch(epoch, metrics)

        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {format_time(epoch_time)}")

        # Print neuron metrics if available
        if neuron_metrics is not None:
            print(f"  Neuron Metrics:")
            print(f"    Usage: {neuron_metrics['avg_usage']:.1%} | "
                  f"Gini: {neuron_metrics['avg_gini']:.3f} | "
                  f"Entropy: {neuron_metrics['avg_entropy']:.3f}")
            print(f"    Top-10: {neuron_metrics['avg_top10']:.2%} | "
                  f"Top-50: {neuron_metrics['avg_top50']:.2%}")
            # Per-layer breakdown
            n_layers = sum(1 for k in neuron_metrics.keys() if k.startswith('L') and k.endswith('_usage'))
            layer_strs = []
            for i in range(n_layers):
                layer_strs.append(
                    f"L{i}: U={neuron_metrics[f'L{i}_usage']:.0%} "
                    f"G={neuron_metrics[f'L{i}_gini']:.2f} "
                    f"E={neuron_metrics[f'L{i}_entropy']:.2f}"
                )
            print(f"    Per-layer usage: {' | '.join(layer_strs)}")

        # Write epoch summary to log
        with open(training_log_file, 'a') as f:
            f.write(f"EPOCH,{epoch},{train_loss:.6f},{train_acc:.6f},"
                   f"{val_loss:.6f},{val_acc:.6f},"
                   f"{optimizer.param_groups[0]['lr']:.6e},{epoch_time:.2f}\n")

        # Analyze activations periodically (skip for v7.5+ - uses different architecture)
        if epoch % 10 == 0:
            model_version = getattr(model, '__version__', None)
            if model_version not in ["7.5", "7.8", "7.9", "8.0", "8.1"]:
                sample_batch = next(iter(val_loader))
                sample_ids = sample_batch['input_ids'][:1].to(device)
                act_stats = analyze_activations(model, sample_ids, device)
                print(f"\n  Neuron Usage Analysis (Epoch {epoch}):")
                for layer_name, stats in act_stats.items():
                    print(f"    {layer_name}: {stats['unique_neurons']}/{stats['total_neurons']} neurons "
                          f"({stats['usage_ratio']:.2%} usage)")
            else:
                # v7.5/v7.8/v7.9/v8.x uses dynamic Q/K/V/O - activation analysis not applicable
                print(f"\n  (Neuron usage analysis skipped for v{model_version} - use analyze_dawn_v75.py instead)")

        # Debug: Log epoch summary for specific epochs
        if debug_logger and debug_logger.should_log_epoch(epoch):
            debug_logger.log_section(f"End of Epoch {epoch}")
            debug_logger.log_epoch_summary(model, sample_batch_for_debug, epoch)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  New best model! (val_loss: {best_val_loss:.4f})")

        ckpt_manager.save_checkpoint(
            model, optimizer, epoch, val_loss, metrics, is_best=is_best,
            scheduler=scheduler, scaler=scaler, model_config=model_kwargs
        )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
