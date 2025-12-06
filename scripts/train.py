"""
DAWN (Dynamic Architecture With Neurons) Training Script

Usage:
    # 기본 학습 (자동으로 최신 체크포인트 이어서 학습)
    python scripts/train.py

    # 처음부터 새로 시작
    python scripts/train.py --from-scratch

    # 특정 체크포인트 폴더에서 이어서 학습
    python scripts/train.py --resume checkpoints/run_20240101_120000_1234

    # 특정 .pt 파일에서 이어서 학습
    python scripts/train.py --resume /path/to/checkpoint_epoch1_step5000.pt

    # 커스텀 config 파일 사용
    python scripts/train.py --config configs/my_config.yaml

Checkpoint Options:
    (기본)           - 자동으로 최신 best_model.pt 탐색 후 이어서 학습
    --from-scratch   - 자동 탐색 비활성화, 처음부터 시작
    --resume <폴더>  - 지정한 폴더의 best_model.pt에서 이어서 학습
    --resume <파일>  - 지정한 .pt 파일에서 직접 이어서 학습
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

from models import create_model_by_version, print_version_info, normalize_version
from utils.training import CheckpointManager, TrainingMonitor, count_parameters, format_time
from utils.checkpoint import load_checkpoint_smart, load_optimizer_state, strip_compile_prefix
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


def is_modern_dawn_model(model):
    """Check if model is DAWN v10.0+"""
    base_model = get_underlying_model(model)

    # Check for v10+ structure
    if hasattr(base_model, '__version__') and base_model.__version__ in ["10.0", "13.0", "13.1", "13.2", "14.0"]:
        return True

    # Structure check: v10+ has layers with .attn and .memory
    if hasattr(base_model, 'layers') and len(base_model.layers) > 0:
        if hasattr(base_model.layers[0], 'attn') and hasattr(base_model.layers[0], 'memory'):
            return True

    return False


# Backward compatibility alias
def is_v75_or_v76_model(model):
    return is_modern_dawn_model(model)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, args, scaler=None, tokenizer=None, log_file=None,
                orthogonality_weight=0.0, diversity_weight=0.0, load_balance_weight=0.0, entropy_weight=0.0, process_norm_weight=0.0,
                debug_logger=None, ckpt_manager=None, model_config=None, start_step=0, global_step=0, total_steps=1):
    """Train for one epoch

    Args:
        start_step: Step to resume from within this epoch (default 0, start from beginning)
        global_step: Global training step counter (for v13.2 starvation decay)
        total_steps: Total training steps (for v13.2 starvation decay)
    """
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

    # Skip steps if resuming from middle of epoch
    if start_step > 0:
        print(f"  Skipping to step {start_step}...")

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", initial=start_step, total=len(dataloader))
    for step, batch in enumerate(pbar):
        # Skip already completed steps
        if step < start_step:
            continue
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

                # Check if v13.2+ (needs routing_info for usage logging)
                # v13.2: usage_ema_compress, v14: usage_ema_feature
                is_v13_2_plus = (hasattr(base_model, 'global_routers') and
                           hasattr(base_model.global_routers, 'neuron_router') and
                           (hasattr(base_model.global_routers.neuron_router, 'usage_ema_compress') or
                            hasattr(base_model.global_routers.neuron_router, 'usage_ema_feature')))

                # v10: DAWN model forward
                if load_balance_weight > 0 or entropy_weight > 0 or is_v13_2_plus:
                    ce_loss, logits, routing_infos = model(input_ids, labels, return_routing_info=True,
                                                           step=global_step, total_steps=total_steps)
                else:
                    ce_loss, logits = model(input_ids, labels, step=global_step, total_steps=total_steps)
                    routing_infos = None

                # Orthogonality loss
                orth_loss = 0.0
                if orthogonality_weight > 0 and hasattr(base_model, 'orthogonality_loss'):
                    orth_loss = base_model.orthogonality_loss()

                # Knowledge diversity loss
                diversity_loss = 0.0
                if diversity_weight > 0 and hasattr(base_model, 'knowledge_diversity_loss'):
                    diversity_loss = base_model.knowledge_diversity_loss()

                # Load balance loss (from model.aux_loss for v13.2+, fallback to old method)
                lb_loss = 0.0
                if load_balance_weight > 0:
                    if hasattr(base_model, 'aux_loss') and base_model.aux_loss != 0.0:
                        lb_loss = base_model.aux_loss
                    elif routing_infos is not None and hasattr(base_model, 'load_balance_loss'):
                        lb_loss = base_model.load_balance_loss(routing_infos)

                # Total loss
                loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + load_balance_weight * lb_loss

                # NaN/INF detection
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARNING] NaN/INF detected at step {step}!")
                    print(f"  ce_loss: {ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss}")
                    print(f"  orth_loss: {orth_loss.item() if torch.is_tensor(orth_loss) else orth_loss}")
                    print(f"  diversity_loss: {diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss}")
                    print(f"  lb_loss: {lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss}")
                    raise ValueError(f"NaN/INF loss detected at epoch {epoch}, step {step}")

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
            # Non-AMP training (CPU or no CUDA)
            base_model = get_underlying_model(model)

            # Check if v13.2+ (needs routing_info for usage logging)
            # v13.2: usage_ema_compress, v14: usage_ema_feature
            is_v13_2_plus = (hasattr(base_model, 'global_routers') and
                       hasattr(base_model.global_routers, 'neuron_router') and
                       (hasattr(base_model.global_routers.neuron_router, 'usage_ema_compress') or
                        hasattr(base_model.global_routers.neuron_router, 'usage_ema_feature')))

            # v10: DAWN model forward
            if load_balance_weight > 0 or entropy_weight > 0 or is_v13_2_plus:
                ce_loss, logits, routing_infos = model(input_ids, labels, return_routing_info=True,
                                                       step=global_step, total_steps=total_steps)
            else:
                ce_loss, logits = model(input_ids, labels, step=global_step, total_steps=total_steps)
                routing_infos = None

            # Orthogonality loss
            orth_loss = 0.0
            if orthogonality_weight > 0 and hasattr(base_model, 'orthogonality_loss'):
                orth_loss = base_model.orthogonality_loss()

            # Knowledge diversity loss
            diversity_loss = 0.0
            if diversity_weight > 0 and hasattr(base_model, 'knowledge_diversity_loss'):
                diversity_loss = base_model.knowledge_diversity_loss()

            # Load balance loss (from model.aux_loss for v13.2+, fallback to old method)
            lb_loss = 0.0
            if load_balance_weight > 0:
                if hasattr(base_model, 'aux_loss') and base_model.aux_loss != 0.0:
                    lb_loss = base_model.aux_loss
                elif routing_infos is not None and hasattr(base_model, 'load_balance_loss'):
                    lb_loss = base_model.load_balance_loss(routing_infos)

            # Total loss
            loss = ce_loss + orthogonality_weight * orth_loss + diversity_weight * diversity_loss + load_balance_weight * lb_loss

            # NaN/INF detection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/INF detected at step {step}!")
                print(f"  ce_loss: {ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss}")
                print(f"  orth_loss: {orth_loss.item() if torch.is_tensor(orth_loss) else orth_loss}")
                print(f"  diversity_loss: {diversity_loss.item() if torch.is_tensor(diversity_loss) else diversity_loss}")
                print(f"  lb_loss: {lb_loss.item() if torch.is_tensor(lb_loss) else lb_loss}")
                raise ValueError(f"NaN/INF loss detected at epoch {epoch}, step {step}")

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Debug: Log gradient flow at specific steps
            if debug_logger and debug_logger.should_log_epoch(epoch) and step in debug_log_steps:
                debug_logger.log_post_backward(model, epoch, step)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Increment global step counter (for v13.2 starvation decay)
        global_step += 1

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

        # Real-time entropy monitoring (every 200 steps)
        if (step + 1) % 200 == 0 and routing_infos is not None:
            with torch.no_grad():
                try:
                    # Layer 0 for routing analysis
                    attn = routing_infos[0].get('attention', routing_infos[0])

                    # Get all expand prefs
                    pref_Q = attn.get('expand_pref_Q')
                    pref_K = attn.get('expand_pref_K')
                    pref_V = attn.get('expand_pref_V')
                    pref_C = attn.get('compress_pref')

                    def calc_entropy_ratio(pref):
                        if pref is None:
                            return 0.0
                        ent = -(pref * (pref + 1e-8).log()).sum(-1).mean()
                        return (ent / math.log(pref.shape[-1]) * 100).item()

                    def calc_token_var(pref):
                        if pref is None:
                            return 0.0
                        return pref.var(dim=1).mean().item()

                    # Entropy ratios (C/Q/K/V)
                    ent_C = calc_entropy_ratio(pref_C)
                    ent_Q = calc_entropy_ratio(pref_Q)
                    ent_K = calc_entropy_ratio(pref_K)
                    ent_V = calc_entropy_ratio(pref_V)

                    # Token variance (C/Q/K/V)
                    var_C = calc_token_var(pref_C)
                    var_Q = calc_token_var(pref_Q)
                    var_K = calc_token_var(pref_K)
                    var_V = calc_token_var(pref_V)

                    # Attention ratio (attn_out_norm vs mem_out_norm per layer)
                    attn_ratios = []
                    for layer_info in routing_infos:
                        attn_norm = layer_info.get('attn_out_norm')
                        mem_norm = layer_info.get('mem_out_norm')
                        if attn_norm is not None and mem_norm is not None:
                            ratio = (attn_norm / (attn_norm + mem_norm + 1e-8) * 100).item()
                            attn_ratios.append(f"{ratio:.0f}")
                        else:
                            attn_ratios.append("-")
                    attn_str = "/".join(attn_ratios)

                    # Compute window average for display
                    avg_loss = window_loss / window_count if window_count > 0 else 0.0
                    avg_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

                    # Compact output with loss/acc
                    print(f"[{step+1}] Loss:{avg_loss:.4f} Acc:{avg_acc:.4f} | Ent C/Q/K/V:{ent_C:.0f}/{ent_Q:.0f}/{ent_K:.0f}/{ent_V:.0f} | TokVar:{var_C:.4f}/{var_Q:.4f}/{var_K:.4f}/{var_V:.4f} | Attn:{attn_str}")

                    # v13.2+: Usage EMA logging (v13.2: C/QK/V, v14: F/R/T with HRP)
                    if hasattr(base_model, 'global_routers') and hasattr(base_model.global_routers, 'neuron_router'):
                        router = base_model.global_routers.neuron_router

                        # Gini coefficient (0=equal, 1=one neuron dominates)
                        def gini(x):
                            x_sorted = torch.sort(x)[0]
                            n = x.numel()
                            idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                            return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()

                        # v14: FRTK naming with Homeostatic Routing Pressure
                        if hasattr(router, 'usage_ema_feature'):
                            ema_F = router.usage_ema_feature
                            ema_R = router.usage_ema_relational
                            ema_T = router.usage_ema_transfer

                            # Active neuron counts
                            active_F = (ema_F > 0.01).sum().item()
                            active_R = (ema_R > 0.01).sum().item()
                            active_T = (ema_T > 0.01).sum().item()
                            n_F, n_R, n_T = ema_F.numel(), ema_R.numel(), ema_T.numel()

                            # Gini coefficients
                            gini_F = gini(ema_F)
                            gini_R = gini(ema_R)
                            gini_T = gini(ema_T)

                            # HRP: imbalance and pressure_strength (same as model_v14.py)
                            imb_F = (ema_F.max() - ema_F.min()).item()
                            imb_R = (ema_R.max() - ema_R.min()).item()
                            imb_T = (ema_T.max() - ema_T.min()).item()
                            prs_F = 0.05 + 1.0 * imb_F  # base_floor=0.05, k=1.0
                            prs_R = 0.05 + 1.0 * imb_R
                            prs_T = 0.05 + 1.0 * imb_T

                            # Usage EMA distribution: min/max/mean
                            min_F, max_F, mean_F = ema_F.min().item(), ema_F.max().item(), ema_F.mean().item()
                            min_R, max_R, mean_R = ema_R.min().item(), ema_R.max().item(), ema_R.mean().item()
                            min_T, max_T, mean_T = ema_T.min().item(), ema_T.max().item(), ema_T.mean().item()

                            # Line 1: Active counts and Gini (same format as v13.2)
                            print(f"         HRP | Active F/R/T:{int(active_F)}/{n_F},{int(active_R)}/{n_R},{int(active_T)}/{n_T} | Gini:{gini_F:.2f}/{gini_R:.2f}/{gini_T:.2f}")
                            # Line 2: HRP pressure strength and imbalance
                            print(f"             Pressure F/R/T:{prs_F:.3f}/{prs_R:.3f}/{prs_T:.3f} | Imbalance:{imb_F:.3f}/{imb_R:.3f}/{imb_T:.3f}")
                            # Line 3: Usage EMA distribution
                            print(f"             EMA F:[{min_F:.3f},{mean_F:.3f},{max_F:.3f}] R:[{min_R:.3f},{mean_R:.3f},{max_R:.3f}] T:[{min_T:.3f},{mean_T:.3f},{max_T:.3f}]")

                        # v13.2: Compress/QK/V naming with starvation weight
                        elif hasattr(router, 'usage_ema_compress'):
                            ema_C = router.usage_ema_compress
                            ema_QK = router.usage_ema_expand_QK
                            ema_V = router.usage_ema_expand_V

                            starvation_weight = max(0.05, math.exp(-3.0 * global_step / total_steps))

                            # Active neuron counts
                            active_C = (ema_C > 0.01).sum().item()
                            active_QK = (ema_QK > 0.01).sum().item()
                            active_V = (ema_V > 0.01).sum().item()
                            n_C, n_QK, n_V = ema_C.numel(), ema_QK.numel(), ema_V.numel()

                            # Gini coefficients
                            gini_C = gini(ema_C)
                            gini_QK = gini(ema_QK)
                            gini_V = gini(ema_V)

                            # Imbalance (for comparison with v14 HRP)
                            imb_C = (ema_C.max() - ema_C.min()).item()
                            imb_QK = (ema_QK.max() - ema_QK.min()).item()
                            imb_V = (ema_V.max() - ema_V.min()).item()

                            # Usage EMA distribution: min/max/mean
                            min_C, max_C, mean_C = ema_C.min().item(), ema_C.max().item(), ema_C.mean().item()
                            min_QK, max_QK, mean_QK = ema_QK.min().item(), ema_QK.max().item(), ema_QK.mean().item()
                            min_V, max_V, mean_V = ema_V.min().item(), ema_V.max().item(), ema_V.mean().item()

                            # Line 1: Active counts and Gini
                            print(f"         Starv:{starvation_weight:.3f} | Active C/QK/V:{int(active_C)}/{n_C},{int(active_QK)}/{n_QK},{int(active_V)}/{n_V} | Gini:{gini_C:.2f}/{gini_QK:.2f}/{gini_V:.2f}")
                            # Line 2: Imbalance (comparable to v14)
                            print(f"             Imbalance C/QK/V:{imb_C:.3f}/{imb_QK:.3f}/{imb_V:.3f}")
                            # Line 3: Usage EMA distribution
                            print(f"             EMA C:[{min_C:.3f},{mean_C:.3f},{max_C:.3f}] QK:[{min_QK:.3f},{mean_QK:.3f},{max_QK:.3f}] V:[{min_V:.3f},{mean_V:.3f},{max_V:.3f}]")

                    # Knowledge neuron usage stats
                    try:
                        # Collect knowledge indices from all layers
                        all_knowledge_idx = []
                        for layer_info in routing_infos:
                            mem = layer_info.get('memory', {})
                            k_idx = mem.get('knowledge_indices')
                            if k_idx is not None:
                                all_knowledge_idx.append(k_idx.flatten())

                        if all_knowledge_idx:
                            all_idx = torch.cat(all_knowledge_idx)
                            n_knowledge = base_model.n_knowledge if hasattr(base_model, 'n_knowledge') else 80

                            # Count usage per knowledge neuron
                            usage_counts = torch.bincount(all_idx.long(), minlength=n_knowledge).float()
                            usage_freq = usage_counts / (usage_counts.sum() + 1e-8)

                            # Active: neurons used at least once
                            active_K = (usage_counts > 0).sum().item()

                            # Entropy (normalized)
                            ent_K = -(usage_freq * (usage_freq + 1e-8).log()).sum()
                            max_ent = math.log(n_knowledge)
                            ent_ratio_K = (ent_K / max_ent * 100).item()

                            # Gini
                            gini_K = gini(usage_freq)

                            print(f"         Knowledge: Active {int(active_K)}/{n_knowledge} | Ent:{ent_ratio_K:.0f}% | Gini:{gini_K:.2f}")
                    except Exception:
                        pass

                    # Warning if collapse detected
                    if min(ent_C, ent_Q, ent_K, ent_V) < 30:
                        print(f"  ⚠ WARNING: Router may be collapsing! (target: 60%)")
                    elif min(ent_C, ent_Q, ent_K, ent_V) > 80:
                        print(f"  ⚠ WARNING: Router too uniform! (target: 60%)")

                except Exception:
                    pass  # Skip if routing_infos format is different

        # Log aggregated metrics every 100 steps (same format as console output)
        if log_file and (step + 1) % log_interval == 0:
            avg_window_loss = window_loss / window_count
            avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

            with open(log_file, 'a') as f:
                # Basic loss/acc line
                f.write(f"[{step+1}] Loss:{avg_window_loss:.4f} Acc:{avg_window_acc:.4f}\n")

                # v14/v13.2: Add router metrics (same format as console)
                if hasattr(base_model, 'global_routers') and hasattr(base_model.global_routers, 'neuron_router'):
                    router = base_model.global_routers.neuron_router

                    def gini(x):
                        x_sorted = torch.sort(x)[0]
                        n = x.numel()
                        idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                        return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()

                    # v14: FRTK with HRP
                    if hasattr(router, 'usage_ema_feature'):
                        ema_F = router.usage_ema_feature
                        ema_R = router.usage_ema_relational
                        ema_T = router.usage_ema_transfer

                        active_F = (ema_F > 0.01).sum().item()
                        active_R = (ema_R > 0.01).sum().item()
                        active_T = (ema_T > 0.01).sum().item()
                        n_F, n_R, n_T = ema_F.numel(), ema_R.numel(), ema_T.numel()

                        gini_F, gini_R, gini_T = gini(ema_F), gini(ema_R), gini(ema_T)

                        imb_F = (ema_F.max() - ema_F.min()).item()
                        imb_R = (ema_R.max() - ema_R.min()).item()
                        imb_T = (ema_T.max() - ema_T.min()).item()
                        prs_F, prs_R, prs_T = 0.05 + 1.0 * imb_F, 0.05 + 1.0 * imb_R, 0.05 + 1.0 * imb_T

                        min_F, max_F, mean_F = ema_F.min().item(), ema_F.max().item(), ema_F.mean().item()
                        min_R, max_R, mean_R = ema_R.min().item(), ema_R.max().item(), ema_R.mean().item()
                        min_T, max_T, mean_T = ema_T.min().item(), ema_T.max().item(), ema_T.mean().item()

                        f.write(f"         HRP | Active F/R/T:{int(active_F)}/{n_F},{int(active_R)}/{n_R},{int(active_T)}/{n_T} | Gini:{gini_F:.2f}/{gini_R:.2f}/{gini_T:.2f}\n")
                        f.write(f"             Pressure F/R/T:{prs_F:.3f}/{prs_R:.3f}/{prs_T:.3f} | Imbalance:{imb_F:.3f}/{imb_R:.3f}/{imb_T:.3f}\n")
                        f.write(f"             EMA F:[{min_F:.3f},{mean_F:.3f},{max_F:.3f}] R:[{min_R:.3f},{mean_R:.3f},{max_R:.3f}] T:[{min_T:.3f},{mean_T:.3f},{max_T:.3f}]\n")

                    # v13.2: Compress/QK/V with starvation
                    elif hasattr(router, 'usage_ema_compress'):
                        ema_C = router.usage_ema_compress
                        ema_QK = router.usage_ema_expand_QK
                        ema_V = router.usage_ema_expand_V

                        starvation_weight = max(0.05, math.exp(-3.0 * global_step / total_steps))

                        active_C = (ema_C > 0.01).sum().item()
                        active_QK = (ema_QK > 0.01).sum().item()
                        active_V = (ema_V > 0.01).sum().item()
                        n_C, n_QK, n_V = ema_C.numel(), ema_QK.numel(), ema_V.numel()

                        gini_C, gini_QK, gini_V = gini(ema_C), gini(ema_QK), gini(ema_V)

                        imb_C = (ema_C.max() - ema_C.min()).item()
                        imb_QK = (ema_QK.max() - ema_QK.min()).item()
                        imb_V = (ema_V.max() - ema_V.min()).item()

                        min_C, max_C, mean_C = ema_C.min().item(), ema_C.max().item(), ema_C.mean().item()
                        min_QK, max_QK, mean_QK = ema_QK.min().item(), ema_QK.max().item(), ema_QK.mean().item()
                        min_V, max_V, mean_V = ema_V.min().item(), ema_V.max().item(), ema_V.mean().item()

                        f.write(f"         Starv:{starvation_weight:.3f} | Active C/QK/V:{int(active_C)}/{n_C},{int(active_QK)}/{n_QK},{int(active_V)}/{n_V} | Gini:{gini_C:.2f}/{gini_QK:.2f}/{gini_V:.2f}\n")
                        f.write(f"             Imbalance C/QK/V:{imb_C:.3f}/{imb_QK:.3f}/{imb_V:.3f}\n")
                        f.write(f"             EMA C:[{min_C:.3f},{mean_C:.3f},{max_C:.3f}] QK:[{min_QK:.3f},{mean_QK:.3f},{max_QK:.3f}] V:[{min_V:.3f},{mean_V:.3f},{max_V:.3f}]\n")

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

        # Save checkpoint every 1000 steps
        if ckpt_manager is not None and (step + 1) % 1000 == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
            step_metrics = {
                'train_loss': avg_loss,
                'train_acc': avg_acc,
                'step': step + 1,
            }
            ckpt_manager.save_checkpoint(
                model, optimizer, epoch, avg_loss, step_metrics, is_best=False,
                scheduler=scheduler, scaler=scaler, model_config=model_config,
                filename=f'checkpoint_epoch{epoch}_step{step+1}.pt',
                epoch_completed=False  # Mid-epoch checkpoint
            )
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{step_acc:.4f}",
                "ckpt": f"step{step+1}"
            })

    # Log remaining steps at end of epoch
    if log_file and window_count > 0:
        avg_window_loss = window_loss / window_count
        avg_window_acc = window_acc_correct / window_acc_valid if window_acc_valid > 0 else 0.0

        with open(log_file, 'a') as f:
            f.write(f"epoch={epoch},step={num_batches},loss={avg_window_loss:.6f},"
                   f"acc={avg_window_acc:.6f}\n")

    avg_loss = total_loss / total_tokens
    avg_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

    return avg_loss, avg_acc, last_neuron_metrics, global_step


def evaluate(model, dataloader, device, args, tokenizer=None, max_batches=200):
    """Evaluate model with MLM masking

    Args:
        max_batches: Maximum number of batches to evaluate (default 200 for faster eval)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    total_valid_tokens = 0

    # Clear CUDA cache before evaluation (helps with torch.compile memory)
    if device.type == 'cuda' if hasattr(device, 'type') else 'cuda' in str(device):
        torch.cuda.empty_cache()

    # Use original model if torch.compiled (avoids CUDA graph memory issues)
    eval_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False, total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)

            # Apply same MLM masking as training
            if tokenizer is not None:
                masked_input_ids, labels = apply_mlm_masking(input_ids, tokenizer)
            else:
                masked_input_ids = input_ids
                labels = input_ids.clone()

            logits = eval_model(masked_input_ids)

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
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for faster training (may cause issues with variable seq lengths)')
    # Training parameter overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch_size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    # Ablation options
    parser.add_argument('--skip-householder', action='store_true',
                        help='Ablation: skip Householder transforms (v8 only)')
    parser.add_argument('--gelu-only', action='store_true',
                        help='Ablation: add GELU after compress (v8 only)')
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
    args.model_version = cfg['model'].get('model_version', '9.0')  # Default to v9.0
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
    # v10+ uses 'rank' key, older versions use 'basis_rank'
    args.basis_rank = cfg['model'].get('basis_rank', cfg['model'].get('rank', 64))
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
    args.knowledge_rank = cfg['model'].get('knowledge_rank', None)  # None = use rank

    # SSM parameters
    args.state_dim = cfg['model'].get('state_dim', 64)

    # Legacy ablation parameters
    args.skip_householder = cfg['model'].get('skip_householder', False)
    args.compress_gelu = cfg['model'].get('compress_gelu', False)

    # Gradient checkpointing
    args.gradient_checkpointing = cfg['model'].get('gradient_checkpointing', False)

    # Top-k sparse routing
    args.top_k_compress = cfg['model'].get('top_k_compress', 16)
    args.top_k_expand = cfg['model'].get('top_k_expand', 8)

    # v13.1 QK/V separation parameters
    args.n_expand_QK = cfg['model'].get('n_expand_QK', 12)
    args.n_expand_V = cfg['model'].get('n_expand_V', 12)
    args.top_k_QK = cfg['model'].get('top_k_QK', 4)
    args.top_k_V = cfg['model'].get('top_k_V', 6)

    # v13.2 unified router parameters
    args.d_space = cfg['model'].get('d_space', 64)
    args.router_dropout = cfg['model'].get('router_dropout', 0.1)

    # v14.0 FRTK parameters (feature/relational/transfer naming)
    args.n_feature = cfg['model'].get('n_feature', cfg['model'].get('n_compress', 48))
    args.n_relational = cfg['model'].get('n_relational', cfg['model'].get('n_expand_QK', 12))
    args.n_transfer = cfg['model'].get('n_transfer', cfg['model'].get('n_expand_V', 12))
    args.top_k_feature = cfg['model'].get('top_k_feature', cfg['model'].get('top_k_compress', 8))
    args.top_k_relational = cfg['model'].get('top_k_relational', cfg['model'].get('top_k_QK', 4))
    args.top_k_transfer = cfg['model'].get('top_k_transfer', cfg['model'].get('top_k_V', 6))

    # v9.0 Compress/Expand/Reflection parameters
    args.n_compress = cfg['model'].get('n_compress', 4)
    args.n_expand = cfg['model'].get('n_expand', 4)
    args.n_reflect = cfg['model'].get('n_reflect', cfg['model'].get('n_process', 128))
    args.reflect_k = cfg['model'].get('reflect_k', cfg['model'].get('process_k', 3))
    # v9.1: separate reflection pools
    args.n_reflect_d = cfg['model'].get('n_reflect_d', cfg['model'].get('n_reflect', 64))
    args.n_reflect_r = cfg['model'].get('n_reflect_r', cfg['model'].get('n_reflect', 64))

    # Training
    args.batch_size = cfg['training']['batch_size']
    args.num_epochs = cfg['training']['num_epochs']
    args.lr = cfg['training']['lr']
    args.weight_decay = cfg['training']['weight_decay']

    # CLI overrides (takes precedence over config)
    if cli_args.epochs is not None:
        args.num_epochs = cli_args.epochs
        print(f"📌 CLI override: epochs={args.num_epochs}")
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size
        print(f"📌 CLI override: batch_size={args.batch_size}")
    if cli_args.lr is not None:
        args.lr = cli_args.lr
        print(f"📌 CLI override: lr={args.lr}")
    if cli_args.skip_householder:
        args.skip_householder = True
        print(f"📌 CLI override: skip_householder=True (ablation mode)")
    if cli_args.gelu_only:
        args.compress_gelu = True
        print(f"📌 CLI override: compress_gelu=True (GELU after compress)")
    args.warmup_epochs = cfg['training'].get('warmup_epochs', None)
    args.warmup_ratio = cfg['training'].get('warmup_ratio', None)  # Alternative to warmup_epochs

    # Regularization weights
    args.orthogonality_weight = cfg['training'].get('orthogonality_weight', 0.0)  # v6.0 compat
    args.diversity_weight = cfg['training'].get('diversity_weight', 0.0)  # v7.0: recipe diversity
    args.load_balance_weight = cfg['training'].get('load_balance_weight', 0.0)  # v7.0: load balance
    args.entropy_weight = cfg['training'].get('entropy_weight', 0.0)  # v13: router entropy loss
    args.process_norm_weight = cfg['training'].get('process_norm_weight', 0.0)  # v8.0: process neuron norm

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
        # Explicit resume - can be either a .pt file or a folder
        resume_path = Path(cli_args.resume)

        if resume_path.suffix == '.pt':
            # Direct .pt file path
            if resume_path.exists():
                latest_best_checkpoint = resume_path
                checkpoint_dir = resume_path.parent  # Use the folder containing the checkpoint
                print(f"\n✓ Resuming from checkpoint file: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")
            else:
                print(f"\n⚠️  Warning: Checkpoint file not found at {resume_path}")
                print(f"    Starting from scratch instead.")
        else:
            # Folder path - look for best_model.pt or latest checkpoint inside
            resume_folder = resume_path
            if not resume_folder.is_absolute():
                resume_folder = Path(args.checkpoint_dir) / resume_folder.name

            best_ckpt = resume_folder / 'best_model.pt'
            if best_ckpt.exists():
                latest_best_checkpoint = best_ckpt
                checkpoint_dir = resume_folder  # Use existing folder
                print(f"\n✓ Resuming from: {latest_best_checkpoint}")
                print(f"✓ Continuing in same folder: {checkpoint_dir}")
            else:
                # best_model.pt doesn't exist, look for checkpoint_epoch*_step*.pt
                checkpoint_files = sorted(
                    resume_folder.glob('checkpoint_epoch*_step*.pt'),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if checkpoint_files:
                    latest_best_checkpoint = checkpoint_files[0]
                    checkpoint_dir = resume_folder
                    print(f"\n✓ No best_model.pt found, using latest intermediate checkpoint:")
                    print(f"  {latest_best_checkpoint}")
                    print(f"✓ Continuing in same folder: {checkpoint_dir}")
                else:
                    print(f"\n⚠️  Warning: No checkpoints found in {resume_folder}")
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
        version = cfg['model'].get('model_version', '9.0')
        run_name = f"run_v{version}_{timestamp}_{random_suffix}"
        checkpoint_dir = base_checkpoint_dir / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Created new run folder: {checkpoint_dir}")

        # Save config for new runs (add model version if not present)
        if 'model_version' not in cfg['model']:
            cfg['model']['model_version'] = '9.0'
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
                # Also load top-level settings
                if 'use_amp' in saved_cfg:
                    args.use_amp = saved_cfg['use_amp']
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

        # v8.0+ / v10+ shared parameters
        args.rank = checkpoint_config.get('rank', getattr(args, 'rank', 64))
        args.basis_rank = args.rank  # Sync basis_rank with rank for model creation
        args.n_knowledge = checkpoint_config.get('n_knowledge', getattr(args, 'n_knowledge', 64))
        args.knowledge_k = checkpoint_config.get('knowledge_k', getattr(args, 'knowledge_k', 8))

        # v8.x specific parameters
        args.n_input = checkpoint_config.get('n_input', getattr(args, 'n_input', 8))
        args.n_process = checkpoint_config.get('n_process', getattr(args, 'n_process', 32))
        args.n_output = checkpoint_config.get('n_output', getattr(args, 'n_output', 8))
        args.process_k = checkpoint_config.get('process_k', getattr(args, 'process_k', 3))

        # v10.0 specific parameters
        args.n_compress = checkpoint_config.get('n_compress', getattr(args, 'n_compress', 4))
        args.n_expand = checkpoint_config.get('n_expand', getattr(args, 'n_expand', 4))

        if checkpoint_training_config:
            # Training hyperparameters (only if not overridden by CLI)
            if cli_args.batch_size is None:
                args.batch_size = checkpoint_training_config.get('batch_size', args.batch_size)
            if cli_args.epochs is None:
                args.num_epochs = checkpoint_training_config.get('num_epochs', args.num_epochs)
            if cli_args.lr is None:
                args.lr = checkpoint_training_config.get('lr', args.lr)
            args.warmup_ratio = checkpoint_training_config.get('warmup_ratio', args.warmup_ratio)
            args.weight_decay = checkpoint_training_config.get('weight_decay', args.weight_decay)
            # Loss weights
            args.orthogonality_weight = checkpoint_training_config.get('orthogonality_weight', args.orthogonality_weight)
            args.diversity_weight = checkpoint_training_config.get('diversity_weight', args.diversity_weight)
            args.load_balance_weight = checkpoint_training_config.get('load_balance_weight', args.load_balance_weight)
            args.entropy_weight = checkpoint_training_config.get('entropy_weight', args.entropy_weight)
            args.process_norm_weight = checkpoint_training_config.get('process_norm_weight', args.process_norm_weight)
            print(f"   → Training params: batch={args.batch_size}, epochs={args.num_epochs}, lr={args.lr}")

        print(f"   → Updated args from checkpoint config (v{args.model_version})")
        if args.model_version == '14.0':
            print(f"   → v14.0 FRTK params: n_feature={getattr(args, 'n_feature', 48)}, n_relational={getattr(args, 'n_relational', 12)}, n_transfer={getattr(args, 'n_transfer', 12)}, rank={args.basis_rank}")
        elif args.model_version in ['13.0', '13.1', '13.2']:
            print(f"   → v{args.model_version} params: n_compress={args.n_compress}, rank={args.basis_rank}, state_dim={getattr(args, 'state_dim', 64)}")
        elif args.model_version == '10.0':
            print(f"   → v10.0 params: n_compress={args.n_compress}, n_expand={args.n_expand}, rank={args.basis_rank}, n_knowledge={args.n_knowledge}")

    # ============================================================
    # STEP 2: Print configuration summary (using updated args)
    # ============================================================
    print(f"\n{'='*60}")
    model_version = getattr(args, 'model_version', '9.0')
    if model_version == 'baseline':
        print(f"Vanilla Transformer Baseline Training")
    else:
        print(f"DAWN (Dynamic Neuron Transformer) Training")
    print(f"{'='*60}")
    print(f"\nModel version: {model_version}")
    print(f"\nModel: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}")

    if model_version != 'baseline':
        if model_version == "14.0":
            # v14.0: FRTK Architecture with Homeostatic Routing
            rank = args.basis_rank
            knowledge_rank = getattr(args, 'knowledge_rank', None) or rank
            state_dim = getattr(args, 'state_dim', 64)
            d_head = args.d_model // args.n_heads
            n_feature = getattr(args, 'n_feature', 48)
            n_relational = getattr(args, 'n_relational', 12)
            n_transfer = getattr(args, 'n_transfer', 12)
            top_k_feature = getattr(args, 'top_k_feature', 8)
            top_k_relational = getattr(args, 'top_k_relational', 4)
            top_k_transfer = getattr(args, 'top_k_transfer', 6)
            d_space = getattr(args, 'd_space', 64)
            grad_ckpt = getattr(args, 'gradient_checkpointing', False)
            print(f"DAWN v{model_version}: rank={rank} - FRTK Architecture!")
            print(f"  FeatureNeurons (F): {n_feature} × {args.d_model} × {rank}")
            print(f"  RelationalNeurons (R): {n_relational} × {rank} × {args.d_model} (Q/K pool)")
            print(f"  TransferNeurons (T): {n_transfer} × {rank} × {args.d_model} (V pool)")
            print(f"  Global SSM: Selective mechanism (token-dependent delta, B_t)")
            print(f"  Unified Router: d_space={d_space} + Homeostatic Routing Pressure")
            print(f"  Context Enhancement: SSM context added to x")
            print(f"  Top-k Feature: {top_k_feature}/{n_feature}")
            print(f"  Top-k Relational: {top_k_relational}/{n_relational}")
            print(f"  Top-k Transfer: {top_k_transfer}/{n_transfer}")
            print(f"  SSM: state_dim={state_dim}")
            print(f"  FlashAttention: enabled (scaled_dot_product_attention)")
            print(f"  Gradient Checkpointing: {grad_ckpt}")
            print(f"  Load Balance: Switch Transformer style + HRP")
            print(f"  Architecture: Mamba SSM → Context + Unified Router (HRP) → FlashAttn")
            print(f"  Attention: d_model space (d_head={d_head})")
            print(f"  KnowledgeNeurons (K):")
            print(f"    - K: {args.n_knowledge} × {knowledge_rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "13.2":
            # v13.2: Unified Neuron Router
            rank = args.basis_rank
            knowledge_rank = getattr(args, 'knowledge_rank', None) or rank
            state_dim = getattr(args, 'state_dim', 64)
            d_head = args.d_model // args.n_heads
            top_k_compress = getattr(args, 'top_k_compress', 8)
            n_expand_QK = getattr(args, 'n_expand_QK', 12)
            n_expand_V = getattr(args, 'n_expand_V', 12)
            top_k_QK = getattr(args, 'top_k_QK', 4)
            top_k_V = getattr(args, 'top_k_V', 6)
            d_space = getattr(args, 'd_space', 64)
            grad_ckpt = getattr(args, 'gradient_checkpointing', False)
            print(f"SharedNeurons (v{model_version}): rank={rank} - Unified Router!")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank} (shared)")
            print(f"  expand_neurons_QK: {n_expand_QK} × {rank} × {args.d_model} (Q/K pool)")
            print(f"  expand_neurons_V: {n_expand_V} × {rank} × {args.d_model} (V pool)")
            print(f"  Global SSM: Selective mechanism (token-dependent delta, B_t)")
            print(f"  Unified Router: d_space={d_space} (all neurons in same space)")
            print(f"  Context Enhancement: SSM context added to x")
            print(f"  Top-k Compress: {top_k_compress}/{args.n_compress}")
            print(f"  Top-k QK: {top_k_QK}/{n_expand_QK}")
            print(f"  Top-k V: {top_k_V}/{n_expand_V}")
            print(f"  SSM: state_dim={state_dim}")
            print(f"  FlashAttention: enabled (scaled_dot_product_attention)")
            print(f"  Gradient Checkpointing: {grad_ckpt}")
            print(f"  Load Balance: Switch Transformer style")
            print(f"  Architecture: Selective SSM → Context + Unified Router → FlashAttn")
            print(f"  Attention: d_model space (d_head={d_head})")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {knowledge_rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "13.1":
            # v13.1: Separate QK/V Expand Pools
            rank = args.basis_rank
            knowledge_rank = getattr(args, 'knowledge_rank', None) or rank
            state_dim = getattr(args, 'state_dim', 64)
            d_head = args.d_model // args.n_heads
            top_k_compress = getattr(args, 'top_k_compress', 8)
            n_expand_QK = getattr(args, 'n_expand_QK', 12)
            n_expand_V = getattr(args, 'n_expand_V', 12)
            top_k_QK = getattr(args, 'top_k_QK', 4)
            top_k_V = getattr(args, 'top_k_V', 6)
            grad_ckpt = getattr(args, 'gradient_checkpointing', False)
            print(f"SharedNeurons (v{model_version}): rank={rank} - QK/V Separated!")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank} (shared)")
            print(f"  expand_neurons_QK: {n_expand_QK} × {rank} × {args.d_model} (Q/K shared)")
            print(f"  expand_neurons_V: {n_expand_V} × {rank} × {args.d_model} (V separate)")
            print(f"  Global SSM: Selective mechanism (token-dependent delta, B_t)")
            print(f"  Global Routers: compress + Q + K + V + memory")
            print(f"  Context Enhancement: SSM context added to x")
            print(f"  Top-k Compress: {top_k_compress}/{args.n_compress}")
            print(f"  Top-k QK: {top_k_QK}/{n_expand_QK}")
            print(f"  Top-k V: {top_k_V}/{n_expand_V}")
            print(f"  SSM: state_dim={state_dim}")
            print(f"  FlashAttention: enabled (scaled_dot_product_attention)")
            print(f"  Gradient Checkpointing: {grad_ckpt}")
            print(f"  Load Balance: Switch Transformer style")
            print(f"  Architecture: Selective SSM → Context + QK/V Routers → FlashAttn")
            print(f"  Attention: d_model space (d_head={d_head})")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {knowledge_rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "13.0":
            # v13.0: Final Architecture - Selective SSM + Context + Top-k
            rank = args.basis_rank
            knowledge_rank = getattr(args, 'knowledge_rank', None) or rank
            state_dim = getattr(args, 'state_dim', 64)
            d_head = args.d_model // args.n_heads
            top_k_compress = getattr(args, 'top_k_compress', 8)
            top_k_expand = getattr(args, 'top_k_expand', 4)
            grad_ckpt = getattr(args, 'gradient_checkpointing', False)
            print(f"SharedNeurons (v{model_version}): rank={rank} - FINAL ARCHITECTURE!")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank} (shared)")
            print(f"  expand_neurons_pool: {args.n_expand} × {rank} × {args.d_model} (1 shared pool for Q/K/V)")
            print(f"  Global SSM: Selective mechanism (token-dependent delta, B_t)")
            print(f"  Global Routers: 5 (compress, expand_Q/K/V, memory) + Top-k")
            print(f"  Context Enhancement: SSM context added to x")
            print(f"  Top-k Compress: {top_k_compress}/{args.n_compress}")
            print(f"  Top-k Expand: {top_k_expand}/{args.n_expand}")
            print(f"  SSM: state_dim={state_dim}")
            print(f"  FlashAttention: enabled (scaled_dot_product_attention)")
            print(f"  Gradient Checkpointing: {grad_ckpt}")
            print(f"  Load Balance: Switch Transformer style")
            print(f"  Architecture: Selective SSM → Context + Top-k Routers → FlashAttn")
            print(f"  Attention: d_model space (d_head={d_head})")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {knowledge_rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "10.0":
            # v10.0: Simplified Compress/Expand
            rank = args.basis_rank
            knowledge_rank = getattr(args, 'knowledge_rank', None) or rank
            print(f"SharedNeurons (v{model_version}): rank={rank} - No Householder!")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank} (Q/K/V/M shared)")
            print(f"  ExpandNeurons: {args.n_expand} × {rank} × {args.d_model} (O shared)")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {knowledge_rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "9.1":
            # v9.1: hard selection + gated reflection + separate pools
            rank = args.basis_rank
            print(f"SharedNeurons (v{model_version}): rank={rank}")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank} (hard selection)")
            print(f"  ExpandNeurons: {args.n_expand} × {rank} × {args.d_model} (hard selection)")
            print(f"  ReflectionNeurons (gated, separate pools):")
            print(f"    - reflect_d: {args.n_reflect_d} × {args.d_model}")
            print(f"    - reflect_r: {args.n_reflect_r} × {rank}")
            print(f"    - Reflect top-k: {args.reflect_k}")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version == "9.0":
            # v9.0: CompressNeurons + ExpandNeurons + ReflectionNeurons
            rank = args.basis_rank
            print(f"SharedNeurons (v{model_version}): rank={rank}")
            print(f"  CompressNeurons: {args.n_compress} × {args.d_model} × {rank}")
            print(f"  ExpandNeurons: {args.n_expand} × {rank} × {args.d_model}")
            print(f"  ReflectionNeurons (unified pool):")
            print(f"    - reflect_d: {args.n_reflect} × {args.d_model}")
            print(f"    - reflect_r: {args.n_reflect} × {rank}")
            print(f"    - Reflect top-k: {args.reflect_k}")
            print(f"  KnowledgeNeurons:")
            print(f"    - K: {args.n_knowledge} × {rank}")
            print(f"    - V: {args.n_knowledge} × {args.d_model}")
            print(f"    - Knowledge top-k: {args.knowledge_k}")
        elif model_version in ["8.0", "8.1", "8.2", "8.3"]:
            # v8.x: SharedNeurons + NeuronMemory
            rank = getattr(args, 'rank', args.basis_rank)
            n_input = getattr(args, 'n_input', 8)
            n_process = getattr(args, 'n_process', 32)
            n_output = getattr(args, 'n_output', 8)
            process_k = getattr(args, 'process_k', 3)
            n_knowledge = getattr(args, 'n_knowledge', 64)
            knowledge_k = getattr(args, 'knowledge_k', 8)
            print(f"SharedNeurons + NeuronMemory (v{model_version}): rank={rank}")
            print(f"  TransformNeurons: input={n_input}, process={n_process}, output={n_output}")
            print(f"  KnowledgeNeurons: {n_knowledge} (top-k: {knowledge_k})")
        else:
            print(f"⚠️  Unsupported version: {model_version}")
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
    if args.entropy_weight > 0:
        reg_parts.append(f"ent={args.entropy_weight}")
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
    model_version = getattr(args, 'model_version', '9.0')

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

    # Add version-specific parameters
    if model_version == '13.1':
        # v13.1: Separate QK/V Expand Pools
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand_QK': getattr(args, 'n_expand_QK', 12),
            'n_expand_V': getattr(args, 'n_expand_V', 12),
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'knowledge_rank': args.knowledge_rank,  # None = use rank
            'rank': args.basis_rank,
            'state_dim': getattr(args, 'state_dim', 64),
            'top_k_compress': getattr(args, 'top_k_compress', 8),
            'top_k_QK': getattr(args, 'top_k_QK', 4),
            'top_k_V': getattr(args, 'top_k_V', 6),
            'gradient_checkpointing': args.gradient_checkpointing,
        })
    elif model_version == '14.0':
        # v14.0: FRTK Architecture with Homeostatic Routing
        model_kwargs.update({
            'n_feature': getattr(args, 'n_feature', 48),
            'n_relational': getattr(args, 'n_relational', 12),
            'n_transfer': getattr(args, 'n_transfer', 12),
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'knowledge_rank': args.knowledge_rank,  # None = use rank
            'rank': args.basis_rank,
            'state_dim': getattr(args, 'state_dim', 64),
            'top_k_feature': getattr(args, 'top_k_feature', 8),
            'top_k_relational': getattr(args, 'top_k_relational', 4),
            'top_k_transfer': getattr(args, 'top_k_transfer', 6),
            'd_space': getattr(args, 'd_space', 64),
            'router_dropout': getattr(args, 'router_dropout', 0.1),
            'gradient_checkpointing': args.gradient_checkpointing,
        })
    elif model_version == '13.2':
        # v13.2: Unified Neuron Router
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand_QK': getattr(args, 'n_expand_QK', 12),
            'n_expand_V': getattr(args, 'n_expand_V', 12),
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'knowledge_rank': args.knowledge_rank,  # None = use rank
            'rank': args.basis_rank,
            'state_dim': getattr(args, 'state_dim', 64),
            'top_k_compress': getattr(args, 'top_k_compress', 8),
            'top_k_QK': getattr(args, 'top_k_QK', 4),
            'top_k_V': getattr(args, 'top_k_V', 6),
            'd_space': getattr(args, 'd_space', 64),
            'router_dropout': getattr(args, 'router_dropout', 0.1),
            'gradient_checkpointing': args.gradient_checkpointing,
        })
    elif model_version == '13.0':
        # v13.0: Final Architecture - Selective SSM + Context + Top-k
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand': args.n_expand,
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'knowledge_rank': args.knowledge_rank,  # None = use rank
            'rank': args.basis_rank,
            'state_dim': getattr(args, 'state_dim', 64),
            'top_k_compress': getattr(args, 'top_k_compress', 8),
            'top_k_expand': getattr(args, 'top_k_expand', 4),
            'gradient_checkpointing': args.gradient_checkpointing,
        })
    elif model_version == '10.0':
        # v10.0: Simplified Compress/Expand (No Householder)
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand': args.n_expand,
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'knowledge_rank': args.knowledge_rank,  # None = use rank
            'rank': args.basis_rank,
        })
    elif model_version == '9.1':
        # v9.1: hard selection + gated reflection + separate pools
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand': args.n_expand,
            'n_reflect_d': args.n_reflect_d,
            'n_reflect_r': args.n_reflect_r,
            'reflect_k': args.reflect_k,
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'rank': args.basis_rank,
        })
    elif model_version == '9.0':
        # v9.0: CompressNeurons + ExpandNeurons + ReflectionNeurons
        model_kwargs.update({
            'n_compress': args.n_compress,
            'n_expand': args.n_expand,
            'n_reflect': args.n_reflect,
            'reflect_k': args.reflect_k,
            'n_knowledge': args.n_knowledge,
            'knowledge_k': args.knowledge_k,
            'rank': args.basis_rank,
        })
    elif model_version in ['8.0', '8.1', '8.2', '8.3']:
        # v8.x: SharedNeurons + NeuronMemory
        model_kwargs.update({
            'n_input': args.n_input,
            'n_process': args.n_process,
            'n_output': args.n_output,
            'process_k': args.process_k,
            'n_knowledge': getattr(args, 'n_knowledge', 64),
            'knowledge_k': getattr(args, 'knowledge_k', 8),
            'rank': args.basis_rank,
            'skip_householder': getattr(args, 'skip_householder', False),  # Ablation
            'compress_gelu': getattr(args, 'compress_gelu', False),  # GELU after compress
        })

    # Create model
    model = create_model_by_version(model_version, model_kwargs)

    model = model.to(device)
    print(f"✅ Model created: v{getattr(model, '__version__', model_version)}")

    # NOTE: torch.compile() is applied AFTER checkpoint loading to avoid _orig_mod. prefix issues

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
    total_steps = args.num_epochs * len(train_loader)

    # Support both warmup_ratio and warmup_epochs
    if args.warmup_ratio is not None:
        warmup_steps = int(total_steps * args.warmup_ratio)
    elif args.warmup_epochs is not None:
        warmup_steps = args.warmup_epochs * len(train_loader)
    else:
        warmup_steps = len(train_loader)  # Default: 1 epoch

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
    start_step = 0  # Step within epoch to resume from

    if resume_checkpoint and resume_checkpoint.exists():
        print(f"\n{'='*60}")
        print("Loading checkpoint weights...")
        print(f"{'='*60}")

        # Use smart checkpoint loading with version awareness
        use_strict = checkpoint_config is not None
        checkpoint, load_info = load_checkpoint_smart(
            model, str(resume_checkpoint), device=device,
            strict=use_strict, verbose=True
        )

        # Load optimizer, scheduler, and scaler states
        start_epoch, best_val_loss, start_step = load_optimizer_state(
            optimizer, checkpoint, scheduler=scheduler, scaler=scaler, verbose=True
        )
    else:
        print(f"\n🆕 Starting fresh training (no checkpoint found)")

    # PyTorch 2.0+ compilation for speed boost (optional)
    # Applied AFTER checkpoint loading to avoid _orig_mod. prefix issues
    if cli_args.compile and hasattr(torch, 'compile'):
        print(f"\nCompiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
        print(f"  Model compiled successfully!")

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

    # v13.2: Calculate total_steps for starvation decay
    steps_per_epoch = len(train_loader)
    total_steps = args.num_epochs * steps_per_epoch
    global_step = (start_epoch - 1) * steps_per_epoch + start_step  # Resume from correct step

    for epoch in range(start_epoch, args.num_epochs + 1):
        # Clear CUDA cache at start of each epoch (helps with torch.compile memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        # Determine start_step for this epoch (only non-zero for first epoch when resuming)
        epoch_start_step = start_step if epoch == start_epoch else 0

        # Train
        train_loss, train_acc, neuron_metrics, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args,
            scaler, tokenizer, log_file=str(training_log_file),
            orthogonality_weight=args.orthogonality_weight,
            diversity_weight=args.diversity_weight,
            load_balance_weight=args.load_balance_weight,
            entropy_weight=args.entropy_weight,
            process_norm_weight=args.process_norm_weight,
            debug_logger=debug_logger,
            ckpt_manager=ckpt_manager,
            model_config=model_kwargs,
            start_step=epoch_start_step,
            global_step=global_step,
            total_steps=total_steps
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

        # v13.2: Print router metrics
        base_model = get_underlying_model(model)
        if hasattr(base_model, 'global_routers') and hasattr(base_model.global_routers, 'neuron_router'):
            router = base_model.global_routers.neuron_router
            if hasattr(router, 'usage_ema_compress'):
                starvation_weight = max(0.05, math.exp(-3.0 * global_step / total_steps))
                # Active count
                active_C = (router.usage_ema_compress > 0.01).sum().item()
                active_QK = (router.usage_ema_expand_QK > 0.01).sum().item()
                active_V = (router.usage_ema_expand_V > 0.01).sum().item()
                n_C = router.usage_ema_compress.numel()
                n_QK = router.usage_ema_expand_QK.numel()
                n_V = router.usage_ema_expand_V.numel()
                # Gini coefficient
                def gini(x):
                    x_sorted = torch.sort(x)[0]
                    n = x.numel()
                    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                    return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()
                gini_C = gini(router.usage_ema_compress)
                gini_QK = gini(router.usage_ema_expand_QK)
                gini_V = gini(router.usage_ema_expand_V)
                print(f"  Router: Starv={starvation_weight:.3f} | Active C/QK/V: {int(active_C)}/{n_C}, {int(active_QK)}/{n_QK}, {int(active_V)}/{n_V} | Gini: {gini_C:.2f}/{gini_QK:.2f}/{gini_V:.2f}")

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
                   f"{optimizer.param_groups[0]['lr']:.6e},{epoch_time:.2f}")

            # v13.2: Add starvation and usage EMA to epoch summary
            base_model = get_underlying_model(model)
            if hasattr(base_model, 'global_routers') and hasattr(base_model.global_routers, 'neuron_router'):
                router = base_model.global_routers.neuron_router
                if hasattr(router, 'usage_ema_compress'):
                    starvation_weight = max(0.05, math.exp(-3.0 * global_step / total_steps))
                    # Active count
                    active_C = (router.usage_ema_compress > 0.01).sum().item()
                    active_QK = (router.usage_ema_expand_QK > 0.01).sum().item()
                    active_V = (router.usage_ema_expand_V > 0.01).sum().item()
                    # Gini coefficient
                    def gini(x):
                        x_sorted = torch.sort(x)[0]
                        n = x.numel()
                        idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                        return (2 * (idx * x_sorted).sum() / (n * x_sorted.sum() + 1e-8) - (n + 1) / n).item()
                    gini_C = gini(router.usage_ema_compress)
                    gini_QK = gini(router.usage_ema_expand_QK)
                    gini_V = gini(router.usage_ema_expand_V)
                    f.write(f",starv={starvation_weight:.3f},active_C={int(active_C)},"
                           f"active_QK={int(active_QK)},active_V={int(active_V)},"
                           f"gini_C={gini_C:.3f},gini_QK={gini_QK:.3f},gini_V={gini_V:.3f}")
            f.write("\n")

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
