#!/usr/bin/env python3
"""
Specific Neuron Analysis Script

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
    from transformers import BertTokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model_version = config.get('model_version', 'v15')

    print(f"Model version: {model_version}")

    # Import appropriate model
    if model_version == 'baseline' or model_version == 'vbaseline':
        from baseline_transformer import VanillaTransformer
        model = VanillaTransformer(**config)
    else:
        from model import DAWN
        model = DAWN(**config)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, config


def get_neuron_activations_hook(model, neuron_id, layer_idx=None):
    """
    Hook to capture neuron activations from FFN layers
    Returns activations for specific neuron across all positions
    """
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            # output: [batch, seq, d_ff] after up projection
            if len(output.shape) == 3:
                activations[name] = output[:, :, neuron_id].detach().cpu()
        return hook

    handles = []

    # For DAWN model - hook into router or FFN
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            if layer_idx is not None and i != layer_idx:
                continue
            # Try to hook FFN up projection
            if hasattr(layer, 'ffn'):
                if hasattr(layer.ffn, 'w_up'):
                    h = layer.ffn.w_up.register_forward_hook(hook_fn(f'layer_{i}'))
                    handles.append(h)
                elif hasattr(layer.ffn, 'router'):
                    # For DAWN with router
                    h = layer.ffn.router.register_forward_hook(hook_fn(f'layer_{i}_router'))
                    handles.append(h)

    return activations, handles


class NeuronAnalyzer:
    def __init__(self, model, tokenizer, device, neuron_id):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neuron_id = neuron_id

        # Check model type
        self.is_dawn = hasattr(model, 'layers') and hasattr(model.layers[0], 'ffn') and \
                       hasattr(model.layers[0].ffn, 'router') if hasattr(model, 'layers') else False

        # Get dimensions
        if self.is_dawn:
            self.n_layers = len(model.layers)
            self.d_ff = model.config.get('d_ff', model.config.get('n_basis', 32))
        else:
            self.n_layers = len(model.layers) if hasattr(model, 'layers') else model.n_layers
            self.d_ff = model.config.get('d_ff', 1024)

        print(f"Model type: {'DAWN' if self.is_dawn else 'Transformer'}")
        print(f"Layers: {self.n_layers}, d_ff/n_basis: {self.d_ff}")
        print(f"Analyzing neuron: {neuron_id}")

    def analyze_top_tokens(self, dataloader, num_batches=100, top_k=100):
        """Find tokens that most strongly activate this neuron"""
        print(f"\n{'='*60}")
        print(f"1. Top Activated Tokens for Neuron {self.neuron_id}")
        print(f"{'='*60}")

        token_activations = defaultdict(list)  # token_id -> list of activations

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Scanning tokens")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Get activations via forward with hooks
                activations = self._get_activations(input_ids)

                if not activations:
                    continue

                # Aggregate across layers (mean or max)
                all_acts = torch.stack(list(activations.values()), dim=0)  # [layers, batch, seq]
                mean_acts = all_acts.mean(dim=0)  # [batch, seq]

                # Record token -> activation
                for b in range(input_ids.shape[0]):
                    for s in range(input_ids.shape[1]):
                        token_id = input_ids[b, s].item()
                        act_val = mean_acts[b, s].item()
                        token_activations[token_id].append(act_val)

        # Compute mean activation per token
        token_mean_act = {}
        for token_id, acts in token_activations.items():
            if len(acts) >= 5:  # Minimum occurrences
                token_mean_act[token_id] = sum(acts) / len(acts)

        # Sort by activation
        sorted_tokens = sorted(token_mean_act.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_k} tokens (highest activation):")
        print("-" * 50)
        results = []
        for i, (token_id, act) in enumerate(sorted_tokens[:top_k]):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            count = len(token_activations[token_id])
            print(f"  {i+1:3d}. '{token:15s}' | act={act:7.4f} | count={count}")
            results.append({'rank': i+1, 'token': token, 'token_id': token_id,
                          'activation': act, 'count': count})

        print(f"\nBottom {20} tokens (lowest activation):")
        print("-" * 50)
        for i, (token_id, act) in enumerate(sorted_tokens[-20:]):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            count = len(token_activations[token_id])
            print(f"  {i+1:3d}. '{token:15s}' | act={act:7.4f} | count={count}")

        return results

    def analyze_layer_importance(self, dataloader, num_batches=50, top_k=8):
        """Analyze how often this neuron is in top-k per layer"""
        print(f"\n{'='*60}")
        print(f"2. Layer-wise Importance of Neuron {self.neuron_id}")
        print(f"{'='*60}")

        layer_stats = defaultdict(lambda: {'in_topk': 0, 'total': 0, 'mean_act': [], 'mean_rank': []})

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Layer analysis")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Get per-layer activations
                layer_activations = self._get_all_neuron_activations(input_ids)

                for layer_name, acts in layer_activations.items():
                    # acts: [batch, seq, d_ff]
                    if acts is None:
                        continue

                    B, S, D = acts.shape

                    # Check if neuron_id is in top-k for each position
                    for b in range(B):
                        for s in range(S):
                            pos_acts = acts[b, s]  # [d_ff]

                            # Rank of this neuron
                            sorted_indices = torch.argsort(pos_acts, descending=True)
                            rank = (sorted_indices == self.neuron_id).nonzero()
                            if len(rank) > 0:
                                rank = rank[0].item() + 1
                            else:
                                rank = D

                            layer_stats[layer_name]['total'] += 1
                            layer_stats[layer_name]['mean_act'].append(pos_acts[self.neuron_id].item())
                            layer_stats[layer_name]['mean_rank'].append(rank)

                            if rank <= top_k:
                                layer_stats[layer_name]['in_topk'] += 1

        print(f"\nNeuron {self.neuron_id} importance per layer (top-{top_k}):")
        print("-" * 70)
        print(f"{'Layer':<20} | {'In Top-K %':>10} | {'Mean Act':>10} | {'Mean Rank':>10}")
        print("-" * 70)

        results = []
        for layer_name in sorted(layer_stats.keys()):
            stats = layer_stats[layer_name]
            topk_pct = 100 * stats['in_topk'] / max(stats['total'], 1)
            mean_act = sum(stats['mean_act']) / max(len(stats['mean_act']), 1)
            mean_rank = sum(stats['mean_rank']) / max(len(stats['mean_rank']), 1)

            print(f"{layer_name:<20} | {topk_pct:>9.2f}% | {mean_act:>10.4f} | {mean_rank:>10.2f}")
            results.append({
                'layer': layer_name,
                'topk_pct': topk_pct,
                'mean_activation': mean_act,
                'mean_rank': mean_rank
            })

        return results

    def analyze_pos_activation(self, dataloader, num_batches=30):
        """Analyze activation by Part-of-Speech"""
        if not HAS_SPACY:
            print("\n[Skipping POS analysis - spaCy not available]")
            return None

        print(f"\n{'='*60}")
        print(f"3. POS-based Activation for Neuron {self.neuron_id}")
        print(f"{'='*60}")

        pos_activations = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="POS analysis")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                # Get activations
                activations = self._get_activations(input_ids)
                if not activations:
                    continue

                all_acts = torch.stack(list(activations.values()), dim=0)
                mean_acts = all_acts.mean(dim=0)  # [batch, seq]

                # Decode and get POS tags
                for b in range(input_ids.shape[0]):
                    tokens = self.tokenizer.convert_ids_to_tokens(input_ids[b].cpu().tolist())
                    text = self.tokenizer.decode(input_ids[b].cpu().tolist(), skip_special_tokens=False)

                    try:
                        doc = nlp(text)
                        # Align spaCy tokens with BERT tokens (approximate)
                        spacy_pos = [token.pos_ for token in doc]

                        for s, token in enumerate(tokens):
                            if token in ['[CLS]', '[SEP]', '[PAD]']:
                                continue
                            # Simple alignment: use position ratio
                            spacy_idx = min(int(s * len(spacy_pos) / len(tokens)), len(spacy_pos) - 1)
                            pos = spacy_pos[spacy_idx] if spacy_idx < len(spacy_pos) else 'X'

                            act_val = mean_acts[b, s].item()
                            pos_activations[pos].append(act_val)
                    except:
                        continue

        print(f"\nActivation by POS tag:")
        print("-" * 50)
        print(f"{'POS':<10} | {'Mean Act':>10} | {'Std':>10} | {'Count':>8}")
        print("-" * 50)

        results = []
        sorted_pos = sorted(pos_activations.items(),
                           key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0,
                           reverse=True)

        for pos, acts in sorted_pos:
            if len(acts) < 10:
                continue
            mean_act = sum(acts) / len(acts)
            std_act = (sum((a - mean_act)**2 for a in acts) / len(acts)) ** 0.5
            print(f"{pos:<10} | {mean_act:>10.4f} | {std_act:>10.4f} | {len(acts):>8}")
            results.append({
                'pos': pos,
                'mean_activation': mean_act,
                'std': std_act,
                'count': len(acts)
            })

        # Interpretation
        print("\nPOS Categories:")
        print("  Content words: NOUN, VERB, ADJ, ADV")
        print("  Function words: DET, ADP, PRON, AUX, CONJ, SCONJ")
        print("  Punctuation: PUNCT")

        return results

    def analyze_ablation(self, test_sentences=None):
        """Compare predictions with vs without this neuron"""
        print(f"\n{'='*60}")
        print(f"4. Ablation Study: Neuron {self.neuron_id}")
        print(f"{'='*60}")

        if test_sentences is None:
            test_sentences = [
                "The cat sat on the [MASK]",
                "She went to the [MASK] to buy groceries",
                "The quick brown [MASK] jumps over the lazy dog",
                "I love to [MASK] books in my free time",
                "The [MASK] is shining brightly today",
                "He is a very [MASK] person",
                "They decided to [MASK] the meeting",
                "The [MASK] was delicious",
            ]

        print("\nPrediction comparison (with vs without neuron):")
        print("-" * 70)

        results = []
        self.model.eval()

        for sentence in test_sentences:
            print(f"\nInput: {sentence}")

            # Tokenize
            tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
            mask_pos = (tokens == self.tokenizer.mask_token_id).nonzero()

            if len(mask_pos) == 0:
                # No mask token, use last position for next-token prediction
                mask_idx = tokens.shape[1] - 1
            else:
                mask_idx = mask_pos[0, 1].item()

            with torch.no_grad():
                # Normal forward
                output_normal = self.model(tokens)
                if isinstance(output_normal, tuple):
                    logits_normal = output_normal[0] if isinstance(output_normal[0], torch.Tensor) and output_normal[0].dim() == 3 else output_normal[1]
                else:
                    logits_normal = output_normal

                probs_normal = F.softmax(logits_normal[0, mask_idx], dim=-1)
                top_normal = torch.topk(probs_normal, k=5)

                # Ablated forward (zero out neuron)
                logits_ablated = self._forward_with_ablation(tokens)
                probs_ablated = F.softmax(logits_ablated[0, mask_idx], dim=-1)
                top_ablated = torch.topk(probs_ablated, k=5)

            # Display
            normal_tokens = self.tokenizer.convert_ids_to_tokens(top_normal.indices.cpu().tolist())
            ablated_tokens = self.tokenizer.convert_ids_to_tokens(top_ablated.indices.cpu().tolist())

            print(f"  With neuron {self.neuron_id}:    {normal_tokens}")
            print(f"  Without neuron {self.neuron_id}: {ablated_tokens}")

            # Measure KL divergence
            kl_div = F.kl_div(
                F.log_softmax(logits_ablated[0, mask_idx], dim=-1),
                probs_normal,
                reduction='sum'
            ).item()
            print(f"  KL divergence: {kl_div:.4f}")

            results.append({
                'sentence': sentence,
                'with_neuron': normal_tokens,
                'without_neuron': ablated_tokens,
                'kl_divergence': kl_div
            })

        return results

    def _get_activations(self, input_ids):
        """Get neuron activations using hooks"""
        activations = {}
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                # After GELU activation in FFN
                if len(output.shape) == 3 and output.shape[-1] > self.neuron_id:
                    activations[name] = output[:, :, self.neuron_id].detach().cpu()
            return hook

        # Register hooks
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'ffn'):
                    if hasattr(layer.ffn, 'w_up'):
                        h = layer.ffn.w_up.register_forward_hook(make_hook(f'layer_{i}'))
                        handles.append(h)

        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return activations

    def _get_all_neuron_activations(self, input_ids):
        """Get all neuron activations (not just one)"""
        activations = {}
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                if len(output.shape) == 3:
                    activations[name] = output.detach().cpu()
            return hook

        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'ffn'):
                    if hasattr(layer.ffn, 'w_up'):
                        h = layer.ffn.w_up.register_forward_hook(make_hook(f'layer_{i}'))
                        handles.append(h)

        try:
            with torch.no_grad():
                _ = self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return activations

    def _forward_with_ablation(self, input_ids):
        """Forward pass with neuron zeroed out"""
        activations_to_modify = {}
        handles = []

        def make_ablation_hook(name):
            def hook(module, input, output):
                # Zero out the specific neuron
                modified = output.clone()
                if len(modified.shape) == 3 and modified.shape[-1] > self.neuron_id:
                    modified[:, :, self.neuron_id] = 0
                return modified
            return hook

        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'ffn'):
                    if hasattr(layer.ffn, 'w_up'):
                        h = layer.ffn.w_up.register_forward_hook(make_ablation_hook(f'layer_{i}'))
                        handles.append(h)

        try:
            output = self.model(input_ids)
            if isinstance(output, tuple):
                logits = output[0] if isinstance(output[0], torch.Tensor) and output[0].dim() == 3 else output[1]
            else:
                logits = output
        finally:
            for h in handles:
                h.remove()

        return logits

    def run_full_analysis(self, dataloader, num_batches=50, save_path=None):
        """Run all analyses"""
        results = {
            'neuron_id': self.neuron_id,
            'model_type': 'DAWN' if self.is_dawn else 'Transformer',
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
        }

        # 1. Top tokens
        results['top_tokens'] = self.analyze_top_tokens(dataloader, num_batches=num_batches)

        # 2. Layer importance
        results['layer_importance'] = self.analyze_layer_importance(dataloader, num_batches=num_batches//2)

        # 3. POS analysis
        results['pos_analysis'] = self.analyze_pos_activation(dataloader, num_batches=num_batches//3)

        # 4. Ablation
        results['ablation'] = self.analyze_ablation()

        # Save results
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {save_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Analyze specific neuron behavior')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron index to analyze')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data', help='Path to data')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to analyze')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    from torch.utils.data import DataLoader, TensorDataset

    # Try to load validation data
    val_path = os.path.join(args.data_path, 'val', 'c4', 'c4_val_50M.pt')
    if os.path.exists(val_path):
        data = torch.load(val_path)
        if data.dim() == 1:
            seq_len = config.get('max_seq_len', 512)
            n_seqs = data.shape[0] // seq_len
            data = data[:n_seqs * seq_len].view(n_seqs, seq_len)
        dataset = TensorDataset(data[:10000])  # Limit for speed
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"  Loaded {len(dataset)} sequences")
    else:
        print(f"  Warning: {val_path} not found, using random data")
        vocab_size = config.get('vocab_size', 30522)
        random_data = torch.randint(0, vocab_size, (1000, 512))
        dataset = TensorDataset(random_data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create analyzer
    analyzer = NeuronAnalyzer(model, tokenizer, device, args.neuron_id)

    # Run analysis
    output_path = args.output or f'neuron_{args.neuron_id}_analysis.json'
    results = analyzer.run_full_analysis(dataloader, num_batches=args.num_batches, save_path=output_path)

    print(f"\n{'='*60}")
    print(f"Analysis Complete for Neuron {args.neuron_id}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
