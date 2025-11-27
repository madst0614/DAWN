"""
DAWN v8.x Householder 효과 분석

분석 내용:
1. Householder 적용 시 v·x 내적값 분포 (mean, std, histogram)
2. 라우터가 선택한 process neuron 조합들의 cosine similarity
3. Input neuron 통과 후 vs Process neuron 통과 후 벡터 변화량 비교
4. 레이어별/컴포넌트별(Q/K/V/O) breakdown

Usage:
    python scripts/analyze_householder_effect.py --checkpoint <path> --num-batches 100
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import create_model_by_version
from utils.data import load_data


class HouseholderAnalyzer:
    """Householder 변환 효과 분석기"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.stats = defaultdict(list)

        # Hook storage
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Compressor/Expander에 hook 등록"""
        for layer_idx, layer in enumerate(self.model.layers):
            # NeuronAttention의 Compressor들 (Q, K, V)
            attn = layer.attention
            for name, compressor in [('Q', attn.compressor_Q),
                                      ('K', attn.compressor_K),
                                      ('V', attn.compressor_V)]:
                hook = self._make_compressor_hook(layer_idx, name, compressor)
                self.hooks.append(compressor.register_forward_hook(hook))

            # Expander (O)
            expander = attn.expander_O
            hook = self._make_expander_hook(layer_idx, 'O', expander)
            self.hooks.append(expander.register_forward_hook(hook))

    def _make_compressor_hook(self, layer_idx, comp_name, compressor):
        """Compressor forward hook 생성"""
        def hook(module, input, output):
            x = input[0]  # [B, S, d_model]
            x_compressed, routing_info = output

            # 분석 수행
            self._analyze_compressor(
                layer_idx, comp_name, x, x_compressed,
                routing_info, compressor
            )
        return hook

    def _make_expander_hook(self, layer_idx, comp_name, expander):
        """Expander forward hook 생성"""
        def hook(module, input, output):
            x = input[0]  # [B, S, rank]
            x_expanded, routing_info = output

            # 분석 수행
            self._analyze_expander(
                layer_idx, comp_name, x, x_expanded,
                routing_info, expander
            )
        return hook

    def _analyze_compressor(self, layer_idx, comp_name, x, x_compressed, routing_info, compressor):
        """Compressor 분석

        1. Input neuron 통과 후 변화량
        2. Process neuron (Householder) 적용 시 v·x 내적
        3. Process neuron 선택 조합의 cosine similarity
        """
        B, S, D = x.shape
        shared = compressor.shared_neurons

        # === 1. Input neuron 통과 후 재계산 ===
        input_weights = routing_info['input_weights']  # [B, S, n_input]
        all_proj = torch.einsum('bsd,ndr->bsnr', x, shared.input_neurons)
        x_after_input = (all_proj * input_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, rank]

        # Input neuron 변화량 (압축이므로 차원이 다름 - norm 비교)
        input_change = x_after_input.norm(dim=-1).mean().item()
        x_norm = x.norm(dim=-1).mean().item()

        # === 2. Process neuron (Householder) v·x 내적 분석 ===
        process_indices = routing_info['process_indices']  # [B, S, k]
        process_k = process_indices.shape[-1]
        rank = x_after_input.shape[-1]

        # 선택된 process neurons 가져오기
        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, process_k, rank)
        selected_v = shared.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        # 각 Householder 단계별 v·x 내적 계산
        vdotx_list = []
        x_current = x_after_input.clone()

        for i in range(process_k):
            v = selected_v[:, :, i, :]  # [B, S, rank]

            # v 정규화 (모델과 동일한 방식)
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
            v_normalized = v / v_norm_sq.sqrt()

            # v·x 내적
            vTx = (x_current * v_normalized).sum(dim=-1)  # [B, S]
            vdotx_list.append(vTx.detach().cpu())

            # Householder 적용
            x_current = x_current - 2 * v_normalized * vTx.unsqueeze(-1)

        # v·x 통계
        all_vdotx = torch.cat([v.flatten() for v in vdotx_list])

        # === 3. 선택된 process neuron 조합의 cosine similarity ===
        # 각 토큰별로 선택된 k개 뉴런들끼리의 유사도
        selected_v_flat = selected_v.view(B * S, process_k, rank)  # [B*S, k, rank]
        selected_v_normalized = F.normalize(selected_v_flat, dim=-1)

        # k x k similarity matrix per token
        sim_matrix = torch.bmm(selected_v_normalized, selected_v_normalized.transpose(1, 2))  # [B*S, k, k]

        # 상삼각 (대각선 제외) 추출
        triu_indices = torch.triu_indices(process_k, process_k, offset=1)
        pairwise_sim = sim_matrix[:, triu_indices[0], triu_indices[1]]  # [B*S, k*(k-1)/2]

        # === 4. Input vs Process 변화량 비교 ===
        # Input neuron 통과 후와 Process neuron 통과 후의 cos_sim
        cos_sim_input_process = F.cosine_similarity(
            x_after_input.view(-1, rank),
            x_compressed.view(-1, rank),
            dim=-1
        )

        # 저장
        key_prefix = f"L{layer_idx}_{comp_name}"
        self.stats[f"{key_prefix}_vdotx_mean"].append(all_vdotx.mean().item())
        self.stats[f"{key_prefix}_vdotx_std"].append(all_vdotx.std().item())
        self.stats[f"{key_prefix}_vdotx_abs_mean"].append(all_vdotx.abs().mean().item())
        self.stats[f"{key_prefix}_vdotx_hist"].append(all_vdotx.numpy())

        self.stats[f"{key_prefix}_neuron_sim_mean"].append(pairwise_sim.mean().item())
        self.stats[f"{key_prefix}_neuron_sim_std"].append(pairwise_sim.std().item())
        self.stats[f"{key_prefix}_neuron_sim_hist"].append(pairwise_sim.detach().cpu().numpy().flatten())

        self.stats[f"{key_prefix}_input_norm"].append(x_norm)
        self.stats[f"{key_prefix}_after_input_norm"].append(input_change)
        self.stats[f"{key_prefix}_cos_sim_input_process"].append(cos_sim_input_process.mean().item())

        # 각 Householder step별 변화량
        for i, vdotx in enumerate(vdotx_list):
            self.stats[f"{key_prefix}_vdotx_step{i}_mean"].append(vdotx.mean().item())
            self.stats[f"{key_prefix}_vdotx_step{i}_abs_mean"].append(vdotx.abs().mean().item())

    def _analyze_expander(self, layer_idx, comp_name, x, x_expanded, routing_info, expander):
        """Expander 분석 (O projection)"""
        B, S, rank = x.shape
        shared = expander.shared_neurons

        # === Process neuron (Householder) v·x 내적 분석 ===
        process_indices = routing_info['process_indices']  # [B, S, k]
        process_k = process_indices.shape[-1]

        idx_expanded = process_indices.unsqueeze(-1).expand(B, S, process_k, rank)
        selected_v = shared.process_neurons.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1)
        selected_v = selected_v.gather(2, idx_expanded)  # [B, S, k, rank]

        vdotx_list = []
        x_current = x.clone()

        for i in range(process_k):
            v = selected_v[:, :, i, :]
            v_norm_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8
            v_normalized = v / v_norm_sq.sqrt()
            vTx = (x_current * v_normalized).sum(dim=-1)
            vdotx_list.append(vTx.detach().cpu())
            x_current = x_current - 2 * v_normalized * vTx.unsqueeze(-1)

        x_after_process = x_current  # Process 후 결과

        all_vdotx = torch.cat([v.flatten() for v in vdotx_list])

        # 선택된 neuron 조합 similarity
        selected_v_flat = selected_v.view(B * S, process_k, rank)
        selected_v_normalized = F.normalize(selected_v_flat, dim=-1)
        sim_matrix = torch.bmm(selected_v_normalized, selected_v_normalized.transpose(1, 2))
        triu_indices = torch.triu_indices(process_k, process_k, offset=1)
        pairwise_sim = sim_matrix[:, triu_indices[0], triu_indices[1]]

        # Input과 Process 후의 cos_sim
        cos_sim_input_process = F.cosine_similarity(
            x.view(-1, rank),
            x_after_process.view(-1, rank),
            dim=-1
        )

        key_prefix = f"L{layer_idx}_{comp_name}"
        self.stats[f"{key_prefix}_vdotx_mean"].append(all_vdotx.mean().item())
        self.stats[f"{key_prefix}_vdotx_std"].append(all_vdotx.std().item())
        self.stats[f"{key_prefix}_vdotx_abs_mean"].append(all_vdotx.abs().mean().item())
        self.stats[f"{key_prefix}_vdotx_hist"].append(all_vdotx.numpy())

        self.stats[f"{key_prefix}_neuron_sim_mean"].append(pairwise_sim.mean().item())
        self.stats[f"{key_prefix}_neuron_sim_std"].append(pairwise_sim.std().item())
        self.stats[f"{key_prefix}_neuron_sim_hist"].append(pairwise_sim.detach().cpu().numpy().flatten())

        self.stats[f"{key_prefix}_cos_sim_input_process"].append(cos_sim_input_process.mean().item())

        for i, vdotx in enumerate(vdotx_list):
            self.stats[f"{key_prefix}_vdotx_step{i}_mean"].append(vdotx.mean().item())
            self.stats[f"{key_prefix}_vdotx_step{i}_abs_mean"].append(vdotx.abs().mean().item())

    def remove_hooks(self):
        """등록된 hooks 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_summary(self):
        """통계 요약"""
        summary = {}

        # 각 metric별 평균
        for key, values in self.stats.items():
            if 'hist' in key:
                # Histogram 데이터는 concat
                summary[key] = np.concatenate(values)
            else:
                summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }

        return summary


def analyze_process_neuron_pool(model):
    """전체 process neuron pool 분석"""
    shared = model.shared_neurons
    process_neurons = shared.process_neurons.detach()  # [n_process, rank]

    # 정규화
    process_normalized = F.normalize(process_neurons, dim=-1)

    # Pairwise similarity
    sim_matrix = process_normalized @ process_normalized.T  # [n_process, n_process]

    # 대각선 제외
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]

    # Norm 분포
    norms = process_neurons.norm(dim=-1)

    return {
        'pool_sim_mean': off_diag.mean().item(),
        'pool_sim_std': off_diag.std().item(),
        'pool_sim_max': off_diag.max().item(),
        'pool_sim_min': off_diag.min().item(),
        'pool_norm_mean': norms.mean().item(),
        'pool_norm_std': norms.std().item(),
        'pool_norm_min': norms.min().item(),
        'pool_norm_max': norms.max().item(),
        'sim_matrix': sim_matrix.cpu().numpy(),
    }


def plot_results(summary, pool_stats, output_dir):
    """결과 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. v·x 분포 히스토그램 (전체)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Layer별, Component별 v·x 분포
    components = ['Q', 'K', 'V', 'O']
    for idx, comp in enumerate(components):
        ax = axes[idx // 2, idx % 2]

        all_vdotx = []
        for key in summary:
            if f'_{comp}_vdotx_hist' in key:
                all_vdotx.append(summary[key])

        if all_vdotx:
            combined = np.concatenate(all_vdotx)
            ax.hist(combined, bins=100, alpha=0.7, density=True)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_title(f'{comp} v·x Distribution (mean={combined.mean():.4f})')
            ax.set_xlabel('v·x')
            ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_dir / 'vdotx_distribution.png', dpi=150)
    plt.close()

    # 2. 선택된 neuron 조합 similarity 분포
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, comp in enumerate(components):
        ax = axes[idx // 2, idx % 2]

        all_sim = []
        for key in summary:
            if f'_{comp}_neuron_sim_hist' in key:
                all_sim.append(summary[key])

        if all_sim:
            combined = np.concatenate(all_sim)
            ax.hist(combined, bins=50, alpha=0.7, density=True)
            ax.set_title(f'{comp} Selected Neuron Pairwise Sim (mean={combined.mean():.4f})')
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_dir / 'selected_neuron_similarity.png', dpi=150)
    plt.close()

    # 3. Process neuron pool similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pool_stats['sim_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Process Neuron Pool Similarity\n(off-diag mean={pool_stats["pool_sim_mean"]:.4f})')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Neuron Index')
    plt.tight_layout()
    plt.savefig(output_dir / 'process_neuron_pool_sim.png', dpi=150)
    plt.close()

    # 4. Input vs Process 변화량 비교
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = []
    q_sims, k_sims, v_sims, o_sims = [], [], [], []

    for layer_idx in range(10):  # 최대 10 레이어
        for comp, sim_list in [('Q', q_sims), ('K', k_sims), ('V', v_sims), ('O', o_sims)]:
            key = f'L{layer_idx}_{comp}_cos_sim_input_process'
            if key in summary:
                sim_list.append(summary[key]['mean'])
                if comp == 'Q':
                    layers.append(layer_idx)

    if layers:
        x = np.arange(len(layers))
        width = 0.2
        ax.bar(x - 1.5*width, q_sims, width, label='Q', alpha=0.8)
        ax.bar(x - 0.5*width, k_sims, width, label='K', alpha=0.8)
        ax.bar(x + 0.5*width, v_sims, width, label='V', alpha=0.8)
        ax.bar(x + 1.5*width, o_sims, width, label='O', alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity (Input → After Process)')
        ax.set_title('Vector Change by Householder Transform')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.legend()
        ax.set_ylim(0.9, 1.0)  # cos_sim이 높으면 변화가 적음

    plt.tight_layout()
    plt.savefig(output_dir / 'input_vs_process_change.png', dpi=150)
    plt.close()

    # 5. Step별 v·x 절대값 평균
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, comp in enumerate(components):
        ax = axes[idx // 2, idx % 2]

        step_means = defaultdict(list)
        for key in summary:
            if f'_{comp}_vdotx_step' in key and '_abs_mean' in key:
                step_num = int(key.split('_step')[1].split('_')[0])
                step_means[step_num].append(summary[key]['mean'])

        if step_means:
            steps = sorted(step_means.keys())
            means = [np.mean(step_means[s]) for s in steps]
            ax.bar(steps, means, alpha=0.8)
            ax.set_xlabel('Householder Step')
            ax.set_ylabel('|v·x| Mean')
            ax.set_title(f'{comp} |v·x| by Householder Step')

    plt.tight_layout()
    plt.savefig(output_dir / 'vdotx_by_step.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Householder Transform Effect in DAWN v8.x')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches to analyze')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 체크포인트 로드
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'best_model.pt'

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Config 로드
    config_path = checkpoint_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_config = cfg['model']
        data_config = cfg['data']
    else:
        raise FileNotFoundError(f"config.json not found in {checkpoint_path.parent}")

    model_version = model_config.get('model_version', '8.0')
    print(f"Model version: {model_version}")

    # 모델 생성
    model_kwargs = {
        'vocab_size': 30522,
        'd_model': model_config.get('d_model', 256),
        'n_layers': model_config.get('n_layers', 4),
        'n_heads': model_config.get('n_heads', 4),
        'rank': model_config.get('rank', 64),
        'n_input': model_config.get('n_input', 8),
        'n_process': model_config.get('n_process', 32),
        'n_output': model_config.get('n_output', 8),
        'process_k': model_config.get('process_k', 3),
        'n_knowledge': model_config.get('n_knowledge', 64),
        'knowledge_k': model_config.get('knowledge_k', 8),
        'max_seq_len': model_config.get('max_seq_len', 128),
        'dropout': 0.0,  # 분석 시 dropout 비활성화
    }

    model = create_model_by_version(model_version, model_kwargs)

    # Handle torch.compile() wrapped checkpoints (keys have "_orig_mod." prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # 데이터 로드
    print("\nLoading data...")
    _, val_loader, tokenizer = load_data(
        data_config=data_config,
        max_length=model_config.get('max_seq_len', 128),
        batch_size=32
    )

    # Process neuron pool 분석
    print("\nAnalyzing process neuron pool...")
    pool_stats = analyze_process_neuron_pool(model)
    print(f"  Pool similarity: mean={pool_stats['pool_sim_mean']:.4f}, "
          f"std={pool_stats['pool_sim_std']:.4f}")
    print(f"  Pool norms: mean={pool_stats['pool_norm_mean']:.4f}, "
          f"std={pool_stats['pool_norm_std']:.4f}")

    # 분석기 생성
    analyzer = HouseholderAnalyzer(model, device)

    # 분석 실행
    print(f"\nAnalyzing {args.num_batches} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=args.num_batches)):
            if batch_idx >= args.num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            _ = model(input_ids)

    # Hooks 제거
    analyzer.remove_hooks()

    # 결과 요약
    print("\nGenerating summary...")
    summary = analyzer.get_summary()

    # 주요 결과 출력
    print("\n" + "=" * 60)
    print("HOUSEHOLDER EFFECT ANALYSIS RESULTS")
    print("=" * 60)

    # 레이어/컴포넌트별 요약
    for layer_idx in range(model_kwargs['n_layers']):
        print(f"\n--- Layer {layer_idx} ---")
        for comp in ['Q', 'K', 'V', 'O']:
            prefix = f"L{layer_idx}_{comp}"

            vdotx_key = f"{prefix}_vdotx_abs_mean"
            sim_key = f"{prefix}_neuron_sim_mean"
            cos_key = f"{prefix}_cos_sim_input_process"

            if vdotx_key in summary:
                print(f"  {comp}:")
                print(f"    |v·x| mean: {summary[vdotx_key]['mean']:.4f}")
                print(f"    Selected neuron sim: {summary[sim_key]['mean']:.4f}")
                print(f"    cos(input, after_process): {summary[cos_key]['mean']:.4f}")

    # 결과 저장
    output_dir = args.output_dir or str(checkpoint_path.parent / 'householder_analysis')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 시각화
    print("\nGenerating plots...")
    plot_results(summary, pool_stats, output_dir)

    # JSON 저장 (histogram 제외)
    json_summary = {}
    for key, value in summary.items():
        if 'hist' not in key:
            json_summary[key] = value
    json_summary['pool_stats'] = {k: v for k, v in pool_stats.items() if k != 'sim_matrix'}

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(json_summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("\nKey findings:")
    print(f"  - Process neuron pool similarity: {pool_stats['pool_sim_mean']:.4f}")

    # cos_sim이 1에 가까우면 변화가 적음
    avg_cos_sim = np.mean([
        summary[k]['mean'] for k in summary
        if 'cos_sim_input_process' in k and isinstance(summary[k], dict)
    ])
    print(f"  - Average cos(input, after_process): {avg_cos_sim:.4f}")
    print(f"    → Householder 변환으로 인한 평균 변화: {1 - avg_cos_sim:.4f}")


if __name__ == '__main__':
    main()
