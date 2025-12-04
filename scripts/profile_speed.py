"""
DAWN Speed Profiling - 병목 지점 측정
"""

import torch
import torch.nn as nn
import time
import sys
sys.path.insert(0, '/content/DAWN')

from models.model_v13 import DAWN


def profile_forward(model, input_ids, n_runs=10, warmup=3):
    """Forward pass 프로파일링"""
    device = next(model.parameters()).device

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, return_routing_info=True)

    torch.cuda.synchronize()

    # Measure
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_ids, return_routing_info=True)
    torch.cuda.synchronize()

    return (time.time() - start) / n_runs * 1000  # ms


def profile_components(model, input_ids):
    """각 컴포넌트별 시간 측정"""
    device = next(model.parameters()).device
    B, S = input_ids.shape

    times = {}

    # 1. Embedding
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = model.token_emb(input_ids) + model.pos_emb(positions)
    torch.cuda.synchronize()
    times['embedding'] = (time.time() - start) / 10 * 1000

    # Get x for next steps
    with torch.no_grad():
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = model.token_emb(input_ids) + model.pos_emb(positions)

    # 2. SSM (global_ssm)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            importance, context, raw_importance = model.global_ssm(x)
    torch.cuda.synchronize()
    times['ssm'] = (time.time() - start) / 10 * 1000

    # Get SSM outputs
    with torch.no_grad():
        importance, context, raw_importance = model.global_ssm(x)
        x_with_context = x + context

    # 3. Router (attention weights)
    normed_x = model.layers[0].norm1(x_with_context)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            compress_w, expand_Q, expand_K, expand_V, _ = \
                model.global_routers.get_attention_weights(normed_x, importance)
    torch.cuda.synchronize()
    times['router'] = (time.time() - start) / 10 * 1000

    # Get router outputs
    with torch.no_grad():
        compress_w, expand_Q, expand_K, expand_V, _ = \
            model.global_routers.get_attention_weights(normed_x, importance)

    # 4. NeuronCircuit (attention)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            attn_out, _ = model.layers[0].attn(normed_x, compress_w, expand_Q, expand_K, expand_V)
    torch.cuda.synchronize()
    times['attention'] = (time.time() - start) / 10 * 1000

    # 5. Memory
    normed_x2 = model.layers[0].norm2(x_with_context)
    with torch.no_grad():
        memory_w, _ = model.global_routers.get_memory_weights(normed_x2, importance)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            mem_out, _ = model.layers[0].memory(normed_x2, memory_w)
    torch.cuda.synchronize()
    times['memory'] = (time.time() - start) / 10 * 1000

    # 6. LM Head
    with torch.no_grad():
        x_final = model.norm(x_with_context)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            logits = model.lm_head(x_final)
    torch.cuda.synchronize()
    times['lm_head'] = (time.time() - start) / 10 * 1000

    # 7. Full layer (for comparison)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _, _ = model.layers[0](x_with_context, importance, model.global_routers)
    torch.cuda.synchronize()
    times['full_layer'] = (time.time() - start) / 10 * 1000

    return times


def profile_layer_breakdown(model, input_ids):
    """레이어별 시간 측정"""
    device = next(model.parameters()).device
    B, S = input_ids.shape

    layer_times = []

    with torch.no_grad():
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = model.token_emb(input_ids) + model.pos_emb(positions)

    for layer_idx, layer in enumerate(model.layers):
        # SSM
        with torch.no_grad():
            importance, context, _ = model.global_ssm(x)
            x_ctx = x + context

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                x_out, _ = layer(x_ctx, importance, model.global_routers)
        torch.cuda.synchronize()
        layer_time = (time.time() - start) / 10 * 1000
        layer_times.append(layer_time)

        # Update x for next layer
        with torch.no_grad():
            x, _ = layer(x_ctx, importance, model.global_routers)

    return layer_times


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})

    model = DAWN(
        vocab_size=config.get('vocab_size', 30522),
        d_model=config.get('d_model', 320),
        n_layers=config.get('n_layers', 12),
        n_heads=config.get('n_heads', 4),
        rank=config.get('rank', 64),
        max_seq_len=config.get('max_seq_len', 512),
        n_compress=config.get('n_compress', 288),
        n_expand=config.get('n_expand', 72),
        n_knowledge=config.get('n_knowledge', 320),
        knowledge_k=config.get('knowledge_k', 14),
        state_dim=config.get('state_dim', 64),
        top_k_compress=config.get('top_k_compress', 48),
        top_k_expand=config.get('top_k_expand', 24),
        gradient_checkpointing=False,  # 프로파일링에서는 끔
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nModel: {config.get('n_layers', 12)} layers, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"Input: batch={args.batch_size}, seq_len={args.seq_len}")

    # Create dummy input
    input_ids = torch.randint(0, 30522, (args.batch_size, args.seq_len), device=device)

    print("\n" + "="*60)
    print("1. TOTAL FORWARD TIME")
    print("="*60)

    total_time = profile_forward(model, input_ids)
    print(f"Total forward: {total_time:.2f} ms")
    print(f"Throughput: {args.batch_size * args.seq_len / total_time * 1000:.0f} tokens/sec")

    print("\n" + "="*60)
    print("2. COMPONENT BREAKDOWN (single layer)")
    print("="*60)

    times = profile_components(model, input_ids)

    total_component = sum(times.values()) - times['full_layer']
    print(f"\n{'Component':<15} {'Time (ms)':<12} {'%':<8}")
    print("-" * 35)
    for name, t in sorted(times.items(), key=lambda x: -x[1]):
        if name != 'full_layer':
            pct = t / total_component * 100
            print(f"{name:<15} {t:<12.2f} {pct:<8.1f}")
    print("-" * 35)
    print(f"{'Sum':<15} {total_component:<12.2f}")
    print(f"{'Full layer':<15} {times['full_layer']:<12.2f}")

    print("\n" + "="*60)
    print("3. LAYER-BY-LAYER TIME")
    print("="*60)

    layer_times = profile_layer_breakdown(model, input_ids)

    print(f"\n{'Layer':<10} {'Time (ms)':<12}")
    print("-" * 25)
    for i, t in enumerate(layer_times):
        print(f"L{i:<9} {t:<12.2f}")
    print("-" * 25)
    print(f"{'Total':<10} {sum(layer_times):<12.2f}")
    print(f"{'Average':<10} {sum(layer_times)/len(layer_times):<12.2f}")

    print("\n" + "="*60)
    print("4. BOTTLENECK ANALYSIS")
    print("="*60)

    # 병목 분석
    ssm_pct = times['ssm'] / times['full_layer'] * 100
    router_pct = times['router'] / times['full_layer'] * 100
    attn_pct = times['attention'] / times['full_layer'] * 100
    mem_pct = times['memory'] / times['full_layer'] * 100

    print(f"\nPer-layer breakdown:")
    print(f"  SSM:       {ssm_pct:.1f}%")
    print(f"  Router:    {router_pct:.1f}%")
    print(f"  Attention: {attn_pct:.1f}%")
    print(f"  Memory:    {mem_pct:.1f}%")

    # SSM이 레이어마다 호출되므로
    ssm_total = times['ssm'] * config.get('n_layers', 12)
    print(f"\nSSM overhead ({config.get('n_layers', 12)} layers): {ssm_total:.2f} ms ({ssm_total/total_time*100:.1f}% of total)")

    # Baseline 대비 예측
    baseline_estimate = total_time - ssm_total - times['router'] * config.get('n_layers', 12)
    print(f"\nEstimated time without SSM/Router: {baseline_estimate:.2f} ms")
    print(f"Overhead ratio: {total_time / baseline_estimate:.2f}x")


if __name__ == '__main__':
    main()
