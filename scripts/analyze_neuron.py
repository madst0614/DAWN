#!/usr/bin/env python3
"""
DAWN v15 Neuron Deep Analysis Script

Analyzes what a specific neuron encodes by:
1. Weight matrix analysis (cosine similarity with embeddings)
2. Actual activation patterns during forward pass
3. POS-based activation statistics

Usage:
    python scripts/analyze_neuron.py --neuron_id 8 --checkpoint path/to/model.pt
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
    print("Warning: spaCy not available, POS analysis will be skipped")


def load_model_and_tokenizer(checkpoint_path, device):
    """Load model from checkpoint"""
    from transformers import BertTokenizer
    from models import create_model_by_version

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    path_str = str(checkpoint_path).lower()
    if 'v15' in path_str:
        version = '15.0'
    elif 'v14' in path_str:
        version = '14.0'
    else:
        version = config.get('model_version', '15.0')

    print(f"Model version: {version}")

    model_kwargs = {
        'vocab_size': config.get('vocab_size', 30522),
        'd_model': config.get('d_model', 320),
        'n_layers': config.get('n_layers', 4),
        'n_heads': config.get('n_heads', 4),
        'rank': config.get('rank', 64),
        'max_seq_len': config.get('max_seq_len', 512),
        'n_feature': config.get('n_feature', 48),
        'n_relational': config.get('n_relational', 12),
        'n_value': config.get('n_value', 12),
        'n_knowledge': config.get('n_knowledge', 80),
        'dropout': config.get('dropout', 0.1),
        'state_dim': config.get('state_dim', 64),
        'knowledge_rank': config.get('knowledge_rank', 128),
        'coarse_k': config.get('coarse_k', 20),
        'fine_k': config.get('fine_k', 10),
    }

    model = create_model_by_version(version, model_kwargs)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, config


class DAWNNeuronAnalyzer:
    """Deep analysis of DAWN v15 feature neurons"""

    def __init__(self, model, tokenizer, device, neuron_id, neuron_type='feature'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type

        config = model.get_config()
        self.n_layers = config.get('n_layers', 12)
        self.n_feature = config.get('n_feature', 48)
        self.d_model = config.get('d_model', 320)
        self.rank = config.get('rank', 64)

        print(f"\nDAWN Neuron Deep Analyzer")
        print(f"  Model: d_model={self.d_model}, rank={self.rank}")
        print(f"  Feature neurons: {self.n_feature}")
        print(f"  Analyzing: {neuron_type} neuron #{neuron_id}")

    def analyze_weight_matrix(self, top_k=100):
        """Analyze neuron's weight matrix - what token directions it responds to"""
        print(f"\n{'='*60}")
        print(f"1. Weight Matrix Analysis for Neuron {self.neuron_id}")
        print(f"{'='*60}")

        # Get feature neuron weight: [n_feature, d_model, rank]
        W_feature = self.model.shared_neurons.feature_neurons.data
        W_n = W_feature[self.neuron_id]  # [d_model, rank]

        # Get token embeddings: [vocab, d_model]
        embed = self.model.token_emb.weight.data  # [vocab, d_model]

        print(f"\nW_{self.neuron_id} shape: {W_n.shape}")
        print(f"Embedding shape: {embed.shape}")

        # Compute how much each token activates this neuron
        # h = x @ W_n → [vocab, rank]
        # activation_strength = ||h||
        with torch.no_grad():
            h = embed @ W_n  # [vocab, rank]
            activation_strength = h.norm(dim=1)  # [vocab]

            # Also compute direction (mean of h normalized)
            h_normalized = F.normalize(h, dim=1)

        # Top tokens by activation strength
        top_indices = activation_strength.topk(top_k).indices
        bottom_indices = activation_strength.topk(top_k, largest=False).indices

        print(f"\nTop {top_k} tokens that activate neuron {self.neuron_id}:")
        print("-" * 60)
        top_results = []
        for i, idx in enumerate(top_indices):
            token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
            strength = activation_strength[idx].item()
            print(f"  {i+1:3d}. '{token:20s}' | strength={strength:.4f}")
            top_results.append({'rank': i+1, 'token': token, 'token_id': idx.item(), 'strength': strength})

        print(f"\nBottom {20} tokens (lowest activation):")
        print("-" * 60)
        for i, idx in enumerate(bottom_indices[:20]):
            token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
            strength = activation_strength[idx].item()
            print(f"  {i+1:3d}. '{token:20s}' | strength={strength:.4f}")

        # Analyze W_n characteristics
        print(f"\nNeuron {self.neuron_id} weight statistics:")
        print(f"  W norm: {W_n.norm():.4f}")
        print(f"  W mean: {W_n.mean():.6f}")
        print(f"  W std: {W_n.std():.4f}")

        # Compare with other neurons
        print(f"\nComparison with other neurons:")
        all_norms = W_feature.view(self.n_feature, -1).norm(dim=1)
        sorted_norms = all_norms.argsort(descending=True)
        rank_of_n = (sorted_norms == self.neuron_id).nonzero()[0].item() + 1
        print(f"  Neuron {self.neuron_id} norm rank: {rank_of_n}/{self.n_feature}")
        print(f"  Top 5 by norm: {sorted_norms[:5].tolist()}")

        return {
            'top_tokens': top_results,
            'neuron_norm': W_n.norm().item(),
            'norm_rank': rank_of_n
        }

    def analyze_activation_patterns(self, dataloader, num_batches=50, max_seq_len=128):
        """Capture actual neuron activations during forward pass"""
        print(f"\n{'='*60}")
        print(f"2. Activation Pattern Analysis for Neuron {self.neuron_id}")
        print(f"{'='*60}")

        # Use tensors for accumulation (faster than dict)
        vocab_size = self.model.token_emb.weight.shape[0]
        token_sum = torch.zeros(vocab_size, device=self.device)
        token_sum_sq = torch.zeros(vocab_size, device=self.device)
        token_count = torch.zeros(vocab_size, device=self.device)

        # Get feature neuron weight
        W_n = self.model.shared_neurons.feature_neurons[self.neuron_id].data  # [d_model, rank]

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Capturing activations")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                input_ids = input_ids[:, :max_seq_len]
                B, S = input_ids.shape

                # Get embeddings
                positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
                x = self.model.token_emb(input_ids) + self.model.pos_emb(positions)

                # Compute neuron activation: h = x @ W_n
                h = torch.einsum('bsd,dr->bsr', x, W_n)  # [B, S, rank]
                activation_norm = h.norm(dim=-1)  # [B, S]

                # Vectorized accumulation
                flat_ids = input_ids.view(-1)  # [B*S]
                flat_acts = activation_norm.view(-1).float()  # [B*S]

                token_sum.scatter_add_(0, flat_ids, flat_acts)
                token_sum_sq.scatter_add_(0, flat_ids, flat_acts ** 2)
                token_count.scatter_add_(0, flat_ids, torch.ones_like(flat_acts))

        # Compute mean and std
        valid_mask = token_count >= 5
        token_mean = torch.zeros_like(token_sum)
        token_std = torch.zeros_like(token_sum)

        token_mean[valid_mask] = token_sum[valid_mask] / token_count[valid_mask]
        variance = (token_sum_sq[valid_mask] / token_count[valid_mask]) - (token_mean[valid_mask] ** 2)
        token_std[valid_mask] = variance.clamp(min=0).sqrt()

        # Get top tokens
        valid_indices = valid_mask.nonzero().squeeze(-1)
        valid_means = token_mean[valid_indices]
        sorted_order = valid_means.argsort(descending=True)

        print(f"\nTop 50 tokens by actual activation (||x @ W_{self.neuron_id}||):")
        print("-" * 60)
        results = []
        for i in range(min(50, len(sorted_order))):
            idx = valid_indices[sorted_order[i]].item()
            mean_act = token_mean[idx].item()
            std_act = token_std[idx].item()
            count = int(token_count[idx].item())
            token = self.tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"  {i+1:3d}. '{token:20s}' | act={mean_act:.4f} +/- {std_act:.4f} | n={count}")
            results.append({'token': token, 'mean_activation': mean_act, 'std': std_act, 'count': count})

        print(f"\nBottom 20 tokens by activation:")
        for i in range(min(20, len(sorted_order))):
            idx = valid_indices[sorted_order[-(i+1)]].item()
            mean_act = token_mean[idx].item()
            token = self.tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"  {i+1:3d}. '{token:20s}' | act={mean_act:.4f}")

        # Overall statistics
        total_count = token_count.sum().item()
        overall_mean = token_sum.sum().item() / total_count if total_count > 0 else 0
        print(f"\nOverall activation statistics:")
        print(f"  Mean: {overall_mean:.4f}")
        print(f"  Total tokens: {int(total_count)}")

        return {'top_tokens': results}

    def analyze_pos_patterns(self, dataloader, num_batches=30, max_seq_len=128):
        """Analyze activation by Part-of-Speech"""
        if not HAS_SPACY:
            print("\n[Skipping POS analysis - spaCy not available]")
            return None

        print(f"\n{'='*60}")
        print(f"3. POS-based Activation Analysis for Neuron {self.neuron_id}")
        print(f"{'='*60}")

        pos_activations = defaultdict(list)
        W_n = self.model.shared_neurons.feature_neurons[self.neuron_id].data

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="POS analysis")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                input_ids = input_ids[:, :max_seq_len]
                B, S = input_ids.shape

                # Compute activations
                positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, S)
                x = self.model.token_emb(input_ids) + self.model.pos_emb(positions)
                h = torch.einsum('bsd,dr->bsr', x, W_n)
                activation_norm = h.norm(dim=-1).cpu()  # [B, S]

                # Batch decode texts
                input_ids_cpu = input_ids.cpu()
                texts = [self.tokenizer.decode(input_ids_cpu[b].tolist(), skip_special_tokens=True) for b in range(B)]
                all_tokens = [self.tokenizer.convert_ids_to_tokens(input_ids_cpu[b].tolist()) for b in range(B)]

                # Batch POS tagging with nlp.pipe()
                try:
                    docs = list(nlp.pipe(texts, batch_size=256))
                except:
                    continue

                for b, doc in enumerate(docs):
                    tokens = all_tokens[b]
                    spacy_tokens = [(t.text, t.pos_) for t in doc]

                    # Align (approximate)
                    for s, bert_token in enumerate(tokens):
                        if bert_token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                            continue

                        # Find closest spaCy token
                        clean_token = bert_token.replace('##', '')
                        pos = 'X'
                        for sp_text, sp_pos in spacy_tokens:
                            if clean_token.lower() in sp_text.lower() or sp_text.lower() in clean_token.lower():
                                pos = sp_pos
                                break

                        act = activation_norm[b, s].item()
                        pos_activations[pos].append(act)

                torch.cuda.empty_cache()

        # Report
        print(f"\nActivation by POS tag:")
        print("-" * 60)
        print(f"{'POS':<12} | {'Mean':>10} | {'Std':>10} | {'Count':>8}")
        print("-" * 60)

        results = []
        sorted_pos = sorted(
            [(pos, acts) for pos, acts in pos_activations.items() if len(acts) >= 20],
            key=lambda x: sum(x[1])/len(x[1]),
            reverse=True
        )

        for pos, acts in sorted_pos:
            mean_act = sum(acts) / len(acts)
            std_act = (sum((a - mean_act)**2 for a in acts) / len(acts)) ** 0.5
            print(f"{pos:<12} | {mean_act:>10.4f} | {std_act:>10.4f} | {len(acts):>8}")
            results.append({'pos': pos, 'mean': mean_act, 'std': std_act, 'count': len(acts)})

        # Interpretation
        print("\n📊 POS Categories:")
        print("  Function words: DET, ADP, PRON, AUX, CCONJ, SCONJ, PART")
        print("  Content words: NOUN, VERB, ADJ, ADV, PROPN")
        print("  Other: PUNCT, NUM, SYM, X")

        return results

    def analyze_context_sensitivity(self, test_sentences=None):
        """Analyze how neuron activates in context"""
        print(f"\n{'='*60}")
        print(f"4. Context Sensitivity Analysis")
        print(f"{'='*60}")

        if test_sentences is None:
            test_sentences = [
                "The cat sat on the mat.",
                "A dog ran through the park.",
                "She quickly walked to the store.",
                "The big red ball bounced high.",
                "He is reading an interesting book.",
                "They went to buy some groceries.",
            ]

        W_n = self.model.shared_neurons.feature_neurons[self.neuron_id].data
        results = []

        self.model.eval()
        with torch.no_grad():
            for sentence in test_sentences:
                tokens = self.tokenizer(sentence, return_tensors='pt', padding=True)
                input_ids = tokens['input_ids'].to(self.device)

                # Get activations
                S = input_ids.shape[1]
                positions = torch.arange(S, device=self.device).unsqueeze(0)
                x = self.model.token_emb(input_ids) + self.model.pos_emb(positions)
                h = torch.einsum('bsd,dr->bsr', x, W_n)
                activation_norm = h.norm(dim=-1)[0].cpu()  # [S]

                # Display
                token_strs = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                print(f"\n\"{sentence}\"")
                print("-" * 50)

                token_acts = []
                for i, (tok, act) in enumerate(zip(token_strs, activation_norm)):
                    if tok not in ['[CLS]', '[SEP]', '[PAD]']:
                        bar = '█' * int(act.item() * 10)
                        print(f"  {tok:15s} | {act.item():.4f} | {bar}")
                        token_acts.append({'token': tok, 'activation': act.item()})

                results.append({'sentence': sentence, 'tokens': token_acts})

        return results

    def analyze_routing_activation(self, dataloader, num_batches=50, max_seq_len=64):
        """Analyze actual routing weights during forward pass"""
        print(f"\n{'='*60}")
        print(f"5. Routing Weight Analysis (Forward Pass)")
        print(f"{'='*60}")

        # Vectorized accumulation
        vocab_size = self.model.token_emb.weight.shape[0]
        token_sum = torch.zeros(vocab_size, device=self.device)
        token_count = torch.zeros(vocab_size, device=self.device)

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Routing analysis")):
                if batch_idx >= num_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0].to(self.device)
                else:
                    input_ids = batch.to(self.device)

                input_ids = input_ids[:, :max_seq_len]
                B, S = input_ids.shape

                # Forward with routing info
                outputs = self.model(input_ids, return_routing_info=True)

                # Get routing_info_list
                if isinstance(outputs, tuple):
                    if len(outputs) == 2:
                        routing_info_list = outputs[1]
                    elif len(outputs) >= 3:
                        routing_info_list = outputs[2]
                    else:
                        continue
                else:
                    continue

                # Layer 0 routing weights
                if routing_info_list and len(routing_info_list) > 0:
                    info = routing_info_list[0]  # First layer
                    if isinstance(info, dict):
                        attn_info = info.get('attention', {})
                        weights = attn_info.get('feature_weights', attn_info.get('neuron_weights'))

                        if weights is not None:
                            flat_ids = input_ids.reshape(-1)  # [B*S]

                            if weights.dim() == 3 and weights.shape[2] > self.neuron_id:
                                w_n = weights[:, :, self.neuron_id].reshape(-1).float()  # [B*S]
                            elif weights.dim() == 2 and weights.shape[1] > self.neuron_id:
                                w_n = weights[:, self.neuron_id].unsqueeze(1).expand(B, S).reshape(-1).float()
                            else:
                                continue

                            token_sum.scatter_add_(0, flat_ids, w_n)
                            token_count.scatter_add_(0, flat_ids, torch.ones_like(w_n))

        # Compute mean
        valid_mask = token_count >= 5
        if not valid_mask.any():
            print("Warning: No routing data collected")
            return None

        token_mean = torch.zeros_like(token_sum)
        token_mean[valid_mask] = token_sum[valid_mask] / token_count[valid_mask]

        # Get sorted results
        valid_indices = valid_mask.nonzero().squeeze(-1)
        valid_means = token_mean[valid_indices]
        sorted_order = valid_means.argsort(descending=True)

        print(f"\nTop 30 tokens by routing weight:")
        print("-" * 50)
        results = []
        for i in range(min(30, len(sorted_order))):
            idx = valid_indices[sorted_order[i]].item()
            weight = token_mean[idx].item()
            count = int(token_count[idx].item())
            token = self.tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"  {i+1:2d}. '{token:15s}' | weight={weight:.6f} | n={count}")
            results.append({'token': token, 'weight': weight, 'count': count})

        print(f"\nBottom 30 tokens by routing weight:")
        for i in range(min(30, len(sorted_order))):
            idx = valid_indices[sorted_order[-(i+1)]].item()
            weight = token_mean[idx].item()
            token = self.tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"  {i+1:2d}. '{token:15s}' | weight={weight:.6f}")

        return results

    def run_full_analysis(self, dataloader, num_batches=50, max_seq_len=128, save_path=None, skip_pos=False):
        """Run complete analysis"""
        results = {
            'neuron_id': self.neuron_id,
            'neuron_type': self.neuron_type,
            'd_model': self.d_model,
            'rank': self.rank,
            'n_feature': self.n_feature,
        }

        # 1. Weight matrix analysis
        results['weight_analysis'] = self.analyze_weight_matrix()

        # 2. Activation patterns
        results['activation_patterns'] = self.analyze_activation_patterns(
            dataloader, num_batches=num_batches, max_seq_len=max_seq_len)

        # 3. POS analysis (optional, slow)
        if skip_pos:
            print("\n[Skipping POS analysis (--skip-pos)]")
            results['pos_analysis'] = None
        else:
            results['pos_analysis'] = self.analyze_pos_patterns(
                dataloader, num_batches=num_batches//2, max_seq_len=max_seq_len)

        # 4. Context sensitivity
        results['context_analysis'] = self.analyze_context_sensitivity()

        # 5. Routing weights (actual forward pass)
        results['routing_analysis'] = self.analyze_routing_activation(
            dataloader, num_batches=num_batches//2, max_seq_len=64)

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n✅ Results saved to: {save_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Deep analysis of DAWN neuron')
    parser.add_argument('--neuron_id', type=int, required=True, help='Neuron index to analyze')
    parser.add_argument('--neuron_type', type=str, default='feature',
                       choices=['feature', 'relational', 'value', 'knowledge'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/data')
    parser.add_argument('--num_batches', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-pos', action='store_true', help='Skip POS analysis (slow)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    from torch.utils.data import DataLoader, TensorDataset

    val_path = os.path.join(args.data_path, 'val', 'c4', 'c4_val_50M.pt')
    if os.path.exists(val_path):
        raw_data = torch.load(val_path, weights_only=False)
        if isinstance(raw_data, dict):
            data = raw_data.get('tokens', raw_data.get('input_ids', list(raw_data.values())[0]))
        else:
            data = raw_data

        if data.dim() == 1:
            seq_len = args.max_seq_len
            n_seqs = data.shape[0] // seq_len
            data = data[:n_seqs * seq_len].view(n_seqs, seq_len)

        dataset = TensorDataset(data[:5000])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"  Loaded {len(dataset)} sequences")
    else:
        print(f"  Warning: {val_path} not found, using random data")
        vocab_size = config.get('vocab_size', 30522)
        random_data = torch.randint(0, vocab_size, (1000, args.max_seq_len))
        dataset = TensorDataset(random_data)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Create analyzer
    analyzer = DAWNNeuronAnalyzer(model, tokenizer, device, args.neuron_id, args.neuron_type)

    # Run analysis
    output_path = args.output or f'neuron_{args.neuron_id}_deep_analysis.json'
    results = analyzer.run_full_analysis(
        dataloader,
        num_batches=args.num_batches,
        max_seq_len=args.max_seq_len,
        save_path=output_path,
        skip_pos=args.skip_pos
    )

    print(f"\n{'='*60}")
    print(f"Analysis Complete for Neuron {args.neuron_id}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
