#!/usr/bin/env python3
"""
Specific Neuron Analysis Script (DAWN v15 compatible)

Usage:
    python scripts/analyze_neuron.py --neuron_id 8 --checkpoint path/to/model.pt
    python scripts/analyze_neuron.py --neuron_id 8 --checkpoint path/to/model.pt --num_batches 50
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import json

# Optional: spaCy for POS tagging
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False
    print("Warning: spaCy not available, skipping POS analysis")


def load_model_and_tokenizer(checkpoint_path, device):
    """Load model from checkpoint"""
    from pathlib import Path
    from transformers import BertTokenizer
    from models import create_model_by_version

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    # Detect version from path or state_dict
    path_str = str(checkpoint_path).lower()
    if 'v15' in path_str:
        version = '15.0'
    elif 'v14' in path_str:
        version = '14.0'
    elif 'v13' in path_str:
        version = '13.0'
    elif 'baseline' in path_str or 'vbaseline' in path_str:
        version = 'baseline'
    else:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        keys_str = ' '.join(state_dict.keys())
        if 'knowledge_encoder' in keys_str:
            version = '15.0'
        elif 'feature_neurons' in keys_str:
            version = '14.0'
        elif 'context_proj' in keys_str:
            version = '13.0'
        else:
            version = config.get('model_version', '15.0')

    print(f"Model version: {version}")

    # Import appropriate model
    if version == 'baseline' or version == 'vbaseline':
        from baseline_transformer import VanillaTransformer
        model = VanillaTransformer(**config)
    else:
        # Build model kwargs
        model_kwargs = {
            'vocab_size': config.get('vocab_size', 30522),
            'd_model': config.get('d_model', 320),
            'n_layers': config.get('n_layers', 4),
            'n_heads': config.get('n_heads', 4),
            'rank': config.get('rank', 64),
            'max_seq_len': config.get('max_seq_len', 512),
            'n_compress': config.get('n_compress', 48),
            'n_expand': config.get('n_expand', 12),
            'n_knowledge': config.get('n_knowledge', 80),
            'dropout': config.get('dropout', 0.1),
            'state_dim': config.get('state_dim', 64),
        }

        if version.startswith('15'):
            model_kwargs['n_feature'] = config.get('n_feature', 48)
            model_kwargs['n_relational'] = config.get('n_relational', 12)
            model_kwargs['n_value'] = config.get('n_value', 12)
            model_kwargs['knowledge_rank'] = config.get('knowledge_rank', 128)
            model_kwargs['coarse_k'] = config.get('coarse_k', 20)
            model_kwargs['fine_k'] = config.get('fine_k', 10)
        elif version.startswith('14'):
            model_kwargs['n_feature'] = config.get('n_feature', config.get('n_compress', 48))
            model_kwargs['n_relational'] = config.get('n_relational', config.get('n_expand', 12))
            model_kwargs['n_transfer'] = config.get('n_transfer', config.get('n_expand', 12))

        model = create_model_by_version(version, model_kwargs)

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, config, version


class DAWNNeuronAnalyzer:
    """Neuron analyzer for DAWN v15 models"""

    def __init__(self, model, tokenizer, device, neuron_id, neuron_type='feature'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type  # 'feature', 'relational', 'value', 'knowledge'

        model_config = model.get_config() if hasattr(model, 'get_config') else {}
        self.n_layers = model_config.get('n_layers', len(model.layers) if hasattr(model, 'layers') else 12)
        self.n_feature = model_config.get('n_feature', 48)
        self.n_relational = model_config.get('n_relational', 12)
        self.n_value = model_config.get('n_value', 12)
        self.n_knowledge = model_config.get('n_knowledge', 80)

        print(f"DAWN Neuron Analyzer")
        print(f"  Layers: {self.n_layers}")
        print(f"  Feature neurons: {self.n_feature}")
        print(f"  Relational neurons: {self.n_relational}")
        print(f"  Value neurons: {self.n_value}")
        print(f"  Knowledge neurons: {self.n_knowledge}")
        print(f"  Analyzing: {neuron_type} neuron #{neuron_id}")

    def analyze_routing_weights(self, dataloader, num_batches=50, max_seq_len=128):
        """Analyze routing weights for specific neuron"""
        print(f"\n{'='*60}")
        print(f"1. Routing Weight Analysis for {self.neuron_type} Neuron {self.neuron_id}")
        print(f"{'='*60}")

        token_weights = defaultdict(list)  # token_id -> list of weights
        layer_weights = defaultdict(list)  # layer_idx -> list of weights

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Analyzing routing")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Truncate sequence for memory
                input_ids = input_ids[:, :max_seq_len]
                B, S = input_ids.shape

                # Forward with routing info
                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    # Without labels: (logits, routing_infos)
                    # With labels: (loss, logits, routing_infos)
                    if isinstance(outputs, tuple):
                        if len(outputs) == 2:
                            routing_info_list = outputs[1]  # (logits, routing_infos)
                        elif len(outputs) >= 3:
                            routing_info_list = outputs[2]  # (loss, logits, routing_infos)
                        else:
                            continue
                    else:
                        continue
                except Exception as e:
                    print(f"Forward error: {e}")
                    continue

                # Extract weights for target neuron
                for layer_idx, routing_info in enumerate(routing_info_list):
                    if not isinstance(routing_info, dict):
                        continue

                    attn_info = routing_info.get('attention', {})

                    # Get appropriate weights based on neuron type
                    if self.neuron_type == 'feature':
                        weights = attn_info.get('feature_weights', attn_info.get('neuron_weights'))
                    elif self.neuron_type == 'relational':
                        weights = attn_info.get('relational_weights_Q')
                    elif self.neuron_type == 'value':
                        weights = attn_info.get('value_weights')
                    else:
                        weights = None

                    if weights is None:
                        continue

                    # weights shape: [B, N] or [B, S, N]
                    if weights.dim() == 2 and weights.shape[1] > self.neuron_id:
                        neuron_w = weights[:, self.neuron_id].cpu()  # [B]
                        layer_weights[layer_idx].extend(neuron_w.tolist())
                    elif weights.dim() == 3 and weights.shape[2] > self.neuron_id:
                        neuron_w = weights[:, :, self.neuron_id].cpu()  # [B, S]
                        layer_weights[layer_idx].extend(neuron_w.mean(dim=1).tolist())

                        # Token-level analysis
                        for b in range(B):
                            for s in range(S):
                                token_id = input_ids[b, s].item()
                                token_weights[token_id].append(neuron_w[b, s].item())

                # Clear cache
                torch.cuda.empty_cache()

        # Report layer-wise weights
        print(f"\nLayer-wise mean weight for {self.neuron_type} neuron {self.neuron_id}:")
        print("-" * 50)
        layer_results = []
        for layer_idx in sorted(layer_weights.keys()):
            weights = layer_weights[layer_idx]
            mean_w = sum(weights) / len(weights) if weights else 0
            std_w = (sum((w - mean_w)**2 for w in weights) / len(weights))**0.5 if weights else 0
            print(f"  Layer {layer_idx:2d}: mean={mean_w:.6f}, std={std_w:.6f}, n={len(weights)}")
            layer_results.append({'layer': layer_idx, 'mean': mean_w, 'std': std_w})

        # Report top tokens
        if token_weights:
            print(f"\nTop 50 tokens with highest weight for neuron {self.neuron_id}:")
            print("-" * 50)
            token_mean = {tid: sum(ws)/len(ws) for tid, ws in token_weights.items() if len(ws) >= 3}
            sorted_tokens = sorted(token_mean.items(), key=lambda x: x[1], reverse=True)[:50]

            token_results = []
            for i, (token_id, mean_w) in enumerate(sorted_tokens):
                token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                count = len(token_weights[token_id])
                print(f"  {i+1:2d}. '{token:15s}' | weight={mean_w:.6f} | count={count}")
                token_results.append({'token': token, 'token_id': token_id, 'weight': mean_w, 'count': count})

            return {'layer_weights': layer_results, 'top_tokens': token_results}

        return {'layer_weights': layer_results}

    def analyze_all_neurons_usage(self, dataloader, num_batches=30, max_seq_len=128):
        """Analyze usage of all neurons (find which are most/least used)"""
        print(f"\n{'='*60}")
        print(f"2. All {self.neuron_type.title()} Neurons Usage Analysis")
        print(f"{'='*60}")

        if self.neuron_type == 'feature':
            n_neurons = self.n_feature
        elif self.neuron_type == 'relational':
            n_neurons = self.n_relational
        elif self.neuron_type == 'value':
            n_neurons = self.n_value
        else:
            n_neurons = self.n_knowledge

        neuron_total_weight = torch.zeros(n_neurons)
        neuron_count = 0

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Usage analysis")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                input_ids = input_ids[:, :max_seq_len]

                try:
                    outputs = self.model(input_ids, return_routing_info=True)
                    if isinstance(outputs, tuple):
                        if len(outputs) == 2:
                            routing_info_list = outputs[1]
                        elif len(outputs) >= 3:
                            routing_info_list = outputs[2]
                        else:
                            continue
                    else:
                        continue
                except:
                    continue

                for routing_info in routing_info_list:
                    if not isinstance(routing_info, dict):
                        continue

                    attn_info = routing_info.get('attention', {})

                    if self.neuron_type == 'feature':
                        weights = attn_info.get('feature_weights', attn_info.get('neuron_weights'))
                    elif self.neuron_type == 'relational':
                        weights = attn_info.get('relational_weights_Q')
                    elif self.neuron_type == 'value':
                        weights = attn_info.get('value_weights')
                    else:
                        weights = None

                    if weights is None:
                        continue

                    # Aggregate
                    if weights.dim() == 2:
                        neuron_total_weight[:weights.shape[1]] += weights.sum(dim=0).cpu()
                    elif weights.dim() == 3:
                        neuron_total_weight[:weights.shape[2]] += weights.sum(dim=(0,1)).cpu()
                    neuron_count += weights.shape[0]

                torch.cuda.empty_cache()

        # Normalize
        if neuron_count > 0:
            neuron_avg = neuron_total_weight / neuron_count
        else:
            print("Warning: No routing data collected!")
            return {'neuron_usage': [], 'target_rank': -1}

        # Report
        sorted_idx = torch.argsort(neuron_avg, descending=True)
        print(f"\nNeuron usage ranking (avg weight across all tokens):")
        print("-" * 50)
        print("Top 10 most used:")
        for i, idx in enumerate(sorted_idx[:10]):
            marker = " <-- TARGET" if idx.item() == self.neuron_id else ""
            print(f"  {i+1:2d}. Neuron {idx.item():3d}: {neuron_avg[idx]:.6f}{marker}")

        print("\nBottom 10 least used:")
        for i, idx in enumerate(sorted_idx[-10:]):
            marker = " <-- TARGET" if idx.item() == self.neuron_id else ""
            print(f"  {i+1:2d}. Neuron {idx.item():3d}: {neuron_avg[idx]:.6f}{marker}")

        # Where is our target neuron?
        target_rank = (sorted_idx == self.neuron_id).nonzero()
        if len(target_rank) > 0:
            rank = target_rank[0].item() + 1
            print(f"\n** Neuron {self.neuron_id} rank: {rank}/{n_neurons} (avg weight: {neuron_avg[self.neuron_id]:.6f})")

        return {
            'neuron_usage': neuron_avg.tolist(),
            'target_rank': rank if len(target_rank) > 0 else -1
        }

    def analyze_ablation(self, test_sentences=None, max_seq_len=64):
        """Compare predictions with vs without this neuron (simplified)"""
        print(f"\n{'='*60}")
        print(f"3. Simplified Ablation Analysis")
        print(f"{'='*60}")
        print("(Note: Full ablation requires model modification, showing routing impact instead)")

        if test_sentences is None:
            test_sentences = [
                "The cat sat on the mat",
                "She went to the store to buy",
                "The quick brown fox jumps",
                "I love to read books",
            ]

        self.model.eval()
        results = []

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for sentence in test_sentences:
                tokens = self.tokenizer.encode(sentence, return_tensors='pt', max_length=max_seq_len, truncation=True)
                tokens = tokens.to(self.device)

                try:
                    outputs = self.model(tokens, return_routing_info=True)
                    if isinstance(outputs, tuple):
                        if len(outputs) == 2:
                            routing_info_list = outputs[1]
                        elif len(outputs) >= 3:
                            routing_info_list = outputs[2]
                        else:
                            routing_info_list = []
                    else:
                        routing_info_list = []
                except Exception as e:
                    print(f"Error: {e}")
                    continue

                # Get weights for target neuron across layers
                neuron_weights_per_layer = []
                for layer_idx, routing_info in enumerate(routing_info_list):
                    if not isinstance(routing_info, dict):
                        continue
                    attn_info = routing_info.get('attention', {})
                    if self.neuron_type == 'feature':
                        weights = attn_info.get('feature_weights', attn_info.get('neuron_weights'))
                    else:
                        weights = attn_info.get(f'{self.neuron_type}_weights')

                    if weights is not None and weights.shape[-1] > self.neuron_id:
                        if weights.dim() == 2:
                            w = weights[0, self.neuron_id].item()
                        else:
                            w = weights[0, :, self.neuron_id].mean().item()
                        neuron_weights_per_layer.append(w)

                avg_weight = sum(neuron_weights_per_layer) / len(neuron_weights_per_layer) if neuron_weights_per_layer else 0

                print(f"\nInput: '{sentence}'")
                print(f"  Neuron {self.neuron_id} avg weight: {avg_weight:.6f}")
                print(f"  Per-layer: {[f'{w:.4f}' for w in neuron_weights_per_layer]}")

                results.append({
                    'sentence': sentence,
                    'avg_weight': avg_weight,
                    'per_layer': neuron_weights_per_layer
                })

                torch.cuda.empty_cache()

        return results

    def run_full_analysis(self, dataloader, num_batches=50, max_seq_len=128, save_path=None):
        """Run all analyses"""
        results = {
            'neuron_id': self.neuron_id,
            'neuron_type': self.neuron_type,
            'n_layers': self.n_layers,
        }

        # 1. Routing weights analysis
        results['routing_analysis'] = self.analyze_routing_weights(
            dataloader, num_batches=num_batches, max_seq_len=max_seq_len)

        # 2. All neurons usage
        results['usage_analysis'] = self.analyze_all_neurons_usage(
            dataloader, num_batches=num_batches//2, max_seq_len=max_seq_len)

        # 3. Ablation (simplified)
        results['ablation'] = self.analyze_ablation(max_seq_len=max_seq_len)

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {save_path}")

        return results


class TransformerNeuronAnalyzer:
    """Neuron analyzer for standard Transformer (baseline) models"""

    def __init__(self, model, tokenizer, device, neuron_id):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neuron_id = neuron_id

        model_config = model.get_config() if hasattr(model, 'get_config') else getattr(model, 'config', {})
        self.n_layers = len(model.layers) if hasattr(model, 'layers') else model_config.get('n_layers', 12)
        self.d_ff = model_config.get('d_ff', 1024)

        print(f"Transformer Neuron Analyzer")
        print(f"  Layers: {self.n_layers}, d_ff: {self.d_ff}")
        print(f"  Analyzing neuron: {neuron_id}")

    def analyze_top_tokens(self, dataloader, num_batches=50, top_k=50, max_seq_len=128):
        """Find tokens that most strongly activate this neuron"""
        print(f"\n{'='*60}")
        print(f"1. Top Activated Tokens for FFN Neuron {self.neuron_id}")
        print(f"{'='*60}")

        token_activations = defaultdict(list)
        activations_cache = {}
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                if output.shape[-1] > self.neuron_id:
                    activations_cache[name] = output[:, :, self.neuron_id].detach()
            return hook

        # Register hooks on w_up layers
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'w_up'):
                    h = layer.ffn.w_up.register_forward_hook(make_hook(f'layer_{i}'))
                    handles.append(h)

        self.model.eval()
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Scanning")):
                    if batch_idx >= num_batches:
                        break

                    if isinstance(batch, (list, tuple)):
                        input_ids = batch[0].to(self.device)
                    else:
                        input_ids = batch.to(self.device)

                    input_ids = input_ids[:, :max_seq_len]
                    activations_cache.clear()

                    # Forward (hooks capture activations)
                    _ = self.model(input_ids)

                    if not activations_cache:
                        continue

                    # Aggregate across layers
                    all_acts = torch.stack(list(activations_cache.values()), dim=0)
                    mean_acts = all_acts.mean(dim=0).cpu()  # [batch, seq]

                    for b in range(input_ids.shape[0]):
                        for s in range(input_ids.shape[1]):
                            token_id = input_ids[b, s].item()
                            token_activations[token_id].append(mean_acts[b, s].item())

                    torch.cuda.empty_cache()

        finally:
            for h in handles:
                h.remove()

        # Report
        token_mean = {tid: sum(acts)/len(acts) for tid, acts in token_activations.items() if len(acts) >= 3}
        sorted_tokens = sorted(token_mean.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_k} tokens:")
        print("-" * 50)
        results = []
        for i, (token_id, act) in enumerate(sorted_tokens[:top_k]):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            count = len(token_activations[token_id])
            print(f"  {i+1:2d}. '{token:15s}' | act={act:8.4f} | n={count}")
            results.append({'token': token, 'activation': act, 'count': count})

        return results

    def run_full_analysis(self, dataloader, num_batches=50, max_seq_len=128, save_path=None):
        results = {
            'neuron_id': self.neuron_id,
            'model_type': 'Transformer',
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
        }

        results['top_tokens'] = self.analyze_top_tokens(
            dataloader, num_batches=num_batches, max_seq_len=max_seq_len)

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {save_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze specific neuron behavior')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron index to analyze')
    parser.add_argument('--neuron_type', type=str, default='feature',
                       choices=['feature', 'relational', 'value', 'knowledge'],
                       help='Neuron type for DAWN models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data', help='Path to data')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (small for memory)')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, tokenizer, config, version = load_model_and_tokenizer(args.checkpoint, device)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    from torch.utils.data import DataLoader, TensorDataset

    val_path = os.path.join(args.data_path, 'val', 'c4', 'c4_val_50M.pt')
    if os.path.exists(val_path):
        raw_data = torch.load(val_path, weights_only=False)

        if isinstance(raw_data, dict):
            data = raw_data.get('tokens', raw_data.get('input_ids', None))
            if data is None:
                data = list(raw_data.values())[0]
        else:
            data = raw_data

        if data.dim() == 1:
            seq_len = args.max_seq_len
            n_seqs = data.shape[0] // seq_len
            data = data[:n_seqs * seq_len].view(n_seqs, seq_len)

        dataset = TensorDataset(data[:5000])  # Limit for speed
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"  Loaded {len(dataset)} sequences")
    else:
        print(f"  Warning: {val_path} not found, using random data")
        vocab_size = config.get('vocab_size', 30522)
        random_data = torch.randint(0, vocab_size, (1000, args.max_seq_len))
        dataset = TensorDataset(random_data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Create appropriate analyzer
    if version.startswith('15') or version.startswith('14'):
        analyzer = DAWNNeuronAnalyzer(model, tokenizer, device, args.neuron_id, args.neuron_type)
    else:
        analyzer = TransformerNeuronAnalyzer(model, tokenizer, device, args.neuron_id)

    # Run analysis
    output_path = args.output or f'neuron_{args.neuron_id}_analysis.json'
    results = analyzer.run_full_analysis(
        dataloader,
        num_batches=args.num_batches,
        max_seq_len=args.max_seq_len,
        save_path=output_path
    )

    print(f"\n{'='*60}")
    print(f"Analysis Complete for Neuron {args.neuron_id}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
