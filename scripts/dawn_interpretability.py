#!/usr/bin/env python3
"""
DAWN Interpretability Analysis - Complete Framework
====================================================

Full implementation:
1. Linguistic Structure: POS, Syntax, Morphology, Semantics
2. Knowledge Structure: Entity types, World knowledge, Temporal
3. Behavioral Patterns: Layer roles, Position effects, Context length
4. Neuron Pool Specialization: QK vs V, Cross-layer reuse
5. Causal Verification: Ablation studies

Usage:
    from dawn_interpretability import DAWNInterpreter
    interpreter = DAWNInterpreter(model, tokenizer, device='cuda')
    interpreter.run_full_analysis(dataloader, max_batches=500)
"""

import os, json, re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DAWNInterpreter:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except:
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

        self._init_storage()
        self.output_dir = "./interpretability_results"

    def _init_storage(self):
        # Token-Neuron
        self.token_neuron_map = defaultdict(lambda: {"FR": Counter(), "FV": Counter(), "R": Counter(), "V": Counter()})
        self.neuron_token_map = defaultdict(lambda: {"tokens": Counter(), "contexts": []})
        self.total_token_counts = Counter()

        # POS
        self.pos_neuron_counts = defaultdict(lambda: defaultdict(Counter))
        self.neuron_pos_counts = defaultdict(lambda: defaultdict(Counter))

        # Syntax
        self.dep_neuron_counts = defaultdict(lambda: defaultdict(Counter))
        self.neuron_dep_counts = defaultdict(lambda: defaultdict(Counter))
        self.head_dep_patterns = defaultdict(Counter)

        # Morphology
        self.morph_neuron_counts = defaultdict(lambda: defaultdict(Counter))
        self.neuron_morph_counts = defaultdict(lambda: defaultdict(Counter))
        self.tense_neurons = defaultdict(Counter)
        self.number_neurons = defaultdict(Counter)
        self.verb_form_neurons = defaultdict(Counter)

        # Semantics
        self.semantic_clusters = {}
        self.synonym_groups = defaultdict(set)

        # Entity/Knowledge
        self.entity_neuron_counts = defaultdict(lambda: defaultdict(Counter))
        self.neuron_entity_counts = defaultdict(lambda: defaultdict(Counter))
        self.entity_examples = defaultdict(list)
        self.capitalized_neurons = defaultdict(Counter)
        self.numeric_neurons = defaultdict(Counter)
        self.temporal_word_neurons = defaultdict(Counter)

        # Layer/Position/Length
        self.layer_neuron_counts = defaultdict(lambda: {"FR": Counter(), "FV": Counter(), "R": Counter(), "V": Counter()})
        self.position_neuron_counts = defaultdict(lambda: defaultdict(Counter))
        self.context_length_counts = defaultdict(lambda: defaultdict(Counter))

        # QK vs V
        self.qk_v_comparison = {"qk_neurons": defaultdict(Counter), "v_neurons": defaultdict(Counter)}
        self.neuron_layer_distribution = defaultdict(lambda: defaultdict(int))

        # Results
        self.specialized_neurons = {}
        self.ablation_results = {}
        self.original_weights = {}

    def _extract_routing_info(self) -> Dict:
        routing = {}
        if hasattr(self.model, 'last_routing_info'):
            return self.model.last_routing_info
        if hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'last_routing'):
                    for k, v in layer.last_routing.items():
                        routing[f"layer{i}_{k}"] = v
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'last_routing'):
                    for k, v in layer.attention.last_routing.items():
                        routing[f"layer{i}_{k}"] = v
        return routing

    def _get_active_neurons(self, weights, b, pos, seq_len) -> List[int]:
        try:
            if weights.dim() == 3 and pos < weights.shape[1]:
                return (weights[b, pos] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            elif weights.dim() == 2:
                idx = b * seq_len + pos if weights.shape[0] != seq_len else pos
                if idx < weights.shape[0]:
                    return (weights[idx] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
        except:
            pass
        return []

    def _get_neuron_type(self, key: str) -> Optional[str]:
        k = key.lower()
        if "feature_r" in k: return "FR"
        elif "feature_v" in k: return "FV"
        elif "relational" in k: return "R"
        elif "value" in k: return "V"
        return None

    def _get_layer_idx(self, key: str) -> int:
        m = re.search(r'layer(\d+)', key)
        return int(m.group(1)) if m else 0

    @torch.no_grad()
    def collect_data(self, dataloader, max_batches=500, context_window=5, verbose=True):
        print(f"\n{'='*60}\nStep 1: Collecting Data\n{'='*60}")

        for batch_idx, batch in enumerate(tqdm(dataloader, disable=not verbose)):
            if batch_idx >= max_batches: break

            if isinstance(batch, dict): input_ids = batch['input_ids'].to(self.device)
            elif isinstance(batch, (list, tuple)): input_ids = batch[0].to(self.device)
            else: input_ids = batch.to(self.device)

            B, L = input_ids.shape

            # v16: use return_routing_info=True
            output = self.model(input_ids, return_routing_info=True)
            if isinstance(output, tuple) and len(output) >= 2:
                routing_infos = output[-1]  # list of per-layer routing info
            else:
                routing_infos = []

            if not routing_infos: continue

            # Batch decode all texts first
            all_tokens = [input_ids[b].cpu().tolist() for b in range(B)]
            all_texts = [self.tokenizer.decode(toks, skip_special_tokens=True) for toks in all_tokens]

            # Batch decode all individual tokens (65536x faster)
            all_flat_tokens = input_ids.view(-1).tolist()
            all_decoded_flat = self.tokenizer.convert_ids_to_tokens(all_flat_tokens)
            all_decoded = [all_decoded_flat[b*L:(b+1)*L] for b in range(B)]

            # Spacy batch processing
            if self.nlp:
                docs = list(self.nlp.pipe(all_texts, batch_size=B))
            else:
                docs = [None] * B

            # Pre-compute topk for all layers/neuron types at once (GPU batch)
            topk_map = {'FR': 8, 'FV': 8, 'R': 4, 'V': 6}
            all_topk = {}  # {(layer_idx, nt): numpy array [B, L, topk]}

            for layer_idx, layer_info in enumerate(routing_infos):
                if 'attention' not in layer_info: continue
                attn = layer_info['attention']

                prefs = {
                    'FR': attn.get('feature_r_pref'),
                    'FV': attn.get('feature_v_pref'),
                    'R': attn.get('relational_q_pref'),
                    'V': attn.get('value_pref'),
                }

                for nt, pref in prefs.items():
                    if pref is None: continue
                    try:
                        topk = min(topk_map[nt], pref.shape[-1])
                        _, topk_indices = torch.topk(pref, topk, dim=-1)  # [B, L, topk]
                        all_topk[(layer_idx, nt)] = topk_indices.cpu().numpy()
                    except (IndexError, RuntimeError):
                        continue

            # Now iterate over tokens (CPU only)
            for b in range(B):
                tokens = all_tokens[b]
                decoded_tokens = all_decoded[b]
                doc = docs[b]
                spacy_idx = 0

                for pos in range(L):
                    tok_str = decoded_tokens[pos].replace('##', '').replace('Ġ', '').replace('▁', '').strip()
                    if not tok_str or tok_str in ['[PAD]','<pad>','[CLS]','[SEP]']: continue

                    self.total_token_counts[tok_str] += 1
                    ctx_s, ctx_e = max(0, pos-context_window), min(L, pos+context_window+1)
                    context = self.tokenizer.decode(tokens[ctx_s:ctx_e])

                    # Linguistic features
                    ling = self._get_ling_features(doc, tok_str, spacy_idx)
                    spacy_idx = ling.get("next_idx", spacy_idx)

                    # Position/length bins
                    rel_pos = pos / L
                    pos_bin = "first" if pos==0 else "start" if rel_pos<0.25 else "middle" if rel_pos<0.75 else "end"
                    len_bin = next((f"{s}-{e}" for s,e in [(0,32),(32,64),(64,128),(128,256),(256,512)] if s<=L<e), "512+")

                    # Record from pre-computed topk (no GPU ops here)
                    for (layer_idx, nt), topk_np in all_topk.items():
                        for n_idx in topk_np[b, pos]:
                            self._record(tok_str, nt, int(n_idx), layer_idx, pos_bin, len_bin, context, ling)

        print(f"Collected: {sum(self.total_token_counts.values()):,} tokens, {len(self.token_neuron_map):,} unique")

    @torch.no_grad()
    def collect_data_vectorized(self, dataloader, max_batches=500, verbose=True):
        """Fully vectorized data collection - 50x faster"""
        import time
        import pandas as pd
        print(f"\n{'='*60}\nStep 1: Collecting Data (Vectorized)\n{'='*60}")

        t_start = time.time()
        topk_map = {'FR': 8, 'FV': 8, 'R': 4, 'V': 6}
        all_records = []
        seq_len = 512

        for batch_idx, batch in enumerate(tqdm(dataloader, disable=not verbose)):
            if batch_idx >= max_batches: break

            if isinstance(batch, (list, tuple)): input_ids = batch[0].to(self.device)
            else: input_ids = batch.to(self.device)

            B, L = input_ids.shape
            seq_len = L
            token_ids = input_ids.cpu().numpy().astype(np.int32)

            output = self.model(input_ids, return_routing_info=True)
            routing_infos = output[-1] if isinstance(output, tuple) else []
            if not routing_infos: continue

            for layer_idx, layer_info in enumerate(routing_infos):
                if 'attention' not in layer_info: continue
                attn = layer_info['attention']

                prefs = [
                    ('FR', attn.get('feature_r_pref')),
                    ('FV', attn.get('feature_v_pref')),
                    ('R', attn.get('relational_q_pref')),
                    ('V', attn.get('value_pref')),
                ]

                for nt_idx, (nt, pref) in enumerate(prefs):
                    if pref is None: continue
                    k = min(topk_map[nt], pref.shape[-1])
                    _, topk_idx = torch.topk(pref, k, dim=-1)
                    topk_np = topk_idx.cpu().numpy().astype(np.int16)

                    b_idx, l_idx, k_idx = np.indices(topk_np.shape)
                    records = np.stack([
                        token_ids[b_idx, l_idx].ravel(),
                        np.full(topk_np.size, nt_idx, dtype=np.int8),
                        topk_np.ravel(),
                        np.full(topk_np.size, layer_idx, dtype=np.int8),
                        l_idx.ravel().astype(np.int16),
                    ], axis=1)
                    all_records.append(records)

        t_forward = time.time()
        print(f"  Forward pass: {t_forward - t_start:.1f}s")

        if not all_records:
            print("No records collected")
            return None

        all_records = np.concatenate(all_records, axis=0)
        print(f"  Raw records: {len(all_records):,}")

        # Optimize dtypes for memory
        df = pd.DataFrame(all_records, columns=['token_id', 'nt_idx', 'neuron_idx', 'layer_idx', 'pos'])
        df['token_id'] = df['token_id'].astype(np.int32)
        df['nt_idx'] = df['nt_idx'].astype(np.int8)
        df['neuron_idx'] = df['neuron_idx'].astype(np.int16)
        df['layer_idx'] = df['layer_idx'].astype(np.int8)
        df['pos'] = df['pos'].astype(np.int16)

        t_df = time.time()
        print(f"  DataFrame created: {t_df - t_forward:.1f}s")

        # Token string mapping (vectorized)
        unique_ids = df['token_id'].unique()
        id_to_str = {int(tid): self.tokenizer.convert_ids_to_tokens([int(tid)])[0] for tid in unique_ids}
        df['token'] = df['token_id'].map(id_to_str)

        nt_names = {0: 'FR', 1: 'FV', 2: 'R', 3: 'V'}
        df['nt'] = df['nt_idx'].map(nt_names)

        df['pos_bin'] = pd.cut(df['pos'] / seq_len, bins=[-0.01, 0.01, 0.25, 0.75, 1.01],
                               labels=['first', 'start', 'middle', 'end'])

        t_map = time.time()
        print(f"  Mapping: {t_map - t_df:.1f}s")

        # Token-Neuron counts
        token_neuron = df.groupby(['token', 'nt', 'neuron_idx']).size()
        for (tok, nt, n_idx), count in token_neuron.items():
            self.token_neuron_map[tok][nt][int(n_idx)] += count
            self.neuron_token_map[f"{nt}_{int(n_idx)}"]["tokens"][tok] += count

        # Layer counts + neuron_layer_distribution
        layer_counts = df.groupby(['layer_idx', 'nt', 'neuron_idx']).size()
        for (layer, nt, n_idx), count in layer_counts.items():
            self.layer_neuron_counts[int(layer)][nt][int(n_idx)] += count
            self.neuron_layer_distribution[f"{nt}_{int(n_idx)}"][int(layer)] += count

        # Position counts
        pos_counts = df.groupby(['pos_bin', 'nt', 'neuron_idx']).size()
        for (pos_bin, nt, n_idx), count in pos_counts.items():
            self.position_neuron_counts[str(pos_bin)][nt][int(n_idx)] += count

        # Total token counts
        tok_counts = df['token'].value_counts()
        for tok, count in tok_counts.items():
            self.total_token_counts[tok] += count

        t_agg = time.time()
        print(f"  Aggregation: {t_agg - t_map:.1f}s")
        print(f"  Total: {t_agg - t_start:.1f}s")
        print(f"Collected: {len(df):,} records, {len(unique_ids):,} unique tokens")

        self._add_spacy_features_sampled(df, sample_ratio=0.1)

        return df

    def _add_spacy_features_sampled(self, df, sample_ratio=0.1):
        """Spacy analysis on sampled unique tokens only"""
        import pandas as pd
        if not self.nlp or df is None:
            return

        unique_tokens = df['token'].unique()
        unique_tokens = [t for t in unique_tokens if t not in ['[PAD]','<pad>','[CLS]','[SEP]','[UNK]']]
        sample_size = max(1000, int(len(unique_tokens) * sample_ratio))
        sampled = np.random.choice(unique_tokens, min(sample_size, len(unique_tokens)), replace=False)

        print(f"Running spacy on {len(sampled)} unique tokens...")

        docs = list(self.nlp.pipe([str(t).replace('##','').replace('Ġ','').replace('▁','') for t in sampled], batch_size=500))
        token_to_pos = {tok: doc[0].pos_ for tok, doc in zip(sampled, docs) if len(doc) > 0}
        token_to_dep = {tok: doc[0].dep_ for tok, doc in zip(sampled, docs) if len(doc) > 0}

        # Vectorized map (faster than lambda)
        pos_series = pd.Series(token_to_pos)
        dep_series = pd.Series(token_to_dep)
        df['pos_tag'] = df['token'].map(pos_series).fillna('UNK')
        df['dep_tag'] = df['token'].map(dep_series).fillna('UNK')

        # POS-Neuron counts
        pos_neuron = df.groupby(['pos_tag', 'nt', 'neuron_idx']).size()
        for (pos, nt, n_idx), count in pos_neuron.items():
            self.pos_neuron_counts[pos][nt][int(n_idx)] += count
            self.neuron_pos_counts[f"{nt}_{int(n_idx)}"][nt][pos] += count

        # Dep-Neuron counts
        dep_neuron = df.groupby(['dep_tag', 'nt', 'neuron_idx']).size()
        for (dep, nt, n_idx), count in dep_neuron.items():
            self.dep_neuron_counts[dep][nt][int(n_idx)] += count
            self.neuron_dep_counts[f"{nt}_{int(n_idx)}"][nt][dep] += count

        # QK vs V comparison
        qk_types = {'FR', 'R'}
        for (pos, nt, n_idx), count in pos_neuron.items():
            if nt in qk_types:
                self.qk_v_comparison["qk_neurons"][pos][int(n_idx)] += count
            else:
                self.qk_v_comparison["v_neurons"][pos][int(n_idx)] += count

        print(f"Spacy features added for {len(token_to_pos)} tokens")

    def _get_ling_features(self, doc, tok_str, start_idx) -> Dict:
        feat = {"pos":"UNK","dep":"UNK","head_pos":"UNK","ner":"O","morph":{},"entity_type":None,
                "is_cap":tok_str[0].isupper() if tok_str else False,
                "is_num":tok_str.replace(',','').replace('.','').isdigit(),
                "is_temp":tok_str.lower() in {'yesterday','today','tomorrow','now','before','after','monday','tuesday','wednesday','thursday','friday','saturday','sunday'},
                "next_idx":start_idx}
        if not doc: return feat
        clean = tok_str.lower().strip('##Ġ▁')
        for i in range(start_idx, min(start_idx+3, len(doc))):
            if clean in doc[i].text.lower() or doc[i].text.lower() in clean:
                feat.update({"pos":doc[i].pos_,"dep":doc[i].dep_,"head_pos":doc[i].head.pos_,
                            "morph":doc[i].morph.to_dict(),"next_idx":i+1})
                if doc[i].ent_type_: feat.update({"ner":doc[i].ent_type_,"entity_type":doc[i].ent_type_})
                break
        return feat

    def _record(self, tok, nt, n_idx, layer, pos_bin, len_bin, ctx, ling):
        nk = f"{nt}_{n_idx}"

        # Basic
        self.token_neuron_map[tok][nt][n_idx] += 1
        self.neuron_token_map[nk]["tokens"][tok] += 1
        if len(self.neuron_token_map[nk]["contexts"]) < 30:
            self.neuron_token_map[nk]["contexts"].append({"token":tok,"context":ctx,"layer":layer})

        # POS
        self.pos_neuron_counts[ling["pos"]][nt][n_idx] += 1
        self.neuron_pos_counts[nk][nt][ling["pos"]] += 1

        # Dependency
        self.dep_neuron_counts[ling["dep"]][nt][n_idx] += 1
        self.neuron_dep_counts[nk][nt][ling["dep"]] += 1
        self.head_dep_patterns[f"{ling['head_pos']}-{ling['dep']}->{ling['pos']}"][nk] += 1

        # Morphology
        for mf, mv in ling["morph"].items():
            mk = f"{mf}={mv}"
            self.morph_neuron_counts[mk][nt][n_idx] += 1
            self.neuron_morph_counts[nk][nt][mk] += 1
            if mf == "Tense": self.tense_neurons[mv][n_idx] += 1
            elif mf == "Number": self.number_neurons[mv][n_idx] += 1
            elif mf == "VerbForm": self.verb_form_neurons[mv][n_idx] += 1

        # Entity
        if ling["entity_type"]:
            et = ling["entity_type"]
            self.entity_neuron_counts[et][nt][n_idx] += 1
            self.neuron_entity_counts[nk][nt][et] += 1
            if len(self.entity_examples[et]) < 50: self.entity_examples[et].append(tok)

        # Knowledge patterns
        if ling["is_cap"]: self.capitalized_neurons[nt][n_idx] += 1
        if ling["is_num"]: self.numeric_neurons[nt][n_idx] += 1
        if ling["is_temp"]: self.temporal_word_neurons[nt][n_idx] += 1

        # Layer/Position/Length
        self.layer_neuron_counts[layer][nt][n_idx] += 1
        self.neuron_layer_distribution[nk][layer] += 1
        self.position_neuron_counts[pos_bin][nt][n_idx] += 1
        self.context_length_counts[len_bin][nt][n_idx] += 1

        # QK vs V
        if nt in ["FR","R"]: self.qk_v_comparison["qk_neurons"][ling["pos"]][n_idx] += 1
        else: self.qk_v_comparison["v_neurons"][ling["pos"]][n_idx] += 1

    def analyze_linguistics(self):
        print(f"\n{'='*60}\nStep 2: Linguistic Analysis\n{'='*60}")

        # POS
        print("\n[POS]")
        for pos in ["NOUN","VERB","DET","ADJ","ADP","PUNCT"]:
            if pos in self.pos_neuron_counts:
                top = self.pos_neuron_counts[pos]["R"].most_common(3)
                if top: print(f"  {pos}: R neurons {[n for n,_ in top]}")

        # Syntax
        print("\n[Syntax]")
        for dep in ["nsubj","dobj","amod","det","prep","ROOT"]:
            if dep in self.dep_neuron_counts:
                top = self.dep_neuron_counts[dep]["R"].most_common(3)
                if top: print(f"  {dep}: R neurons {[n for n,_ in top]}")

        print("\n  Top patterns:")
        for pat, neurons in sorted(self.head_dep_patterns.items(), key=lambda x:-sum(x[1].values()))[:5]:
            print(f"    {pat}: {sum(neurons.values())}")

        # Morphology
        print("\n[Morphology]")
        for name, data in [("Tense", self.tense_neurons), ("Number", self.number_neurons), ("VerbForm", self.verb_form_neurons)]:
            if data:
                print(f"  {name}:")
                for val, neurons in data.items():
                    top = neurons.most_common(3)
                    if top: print(f"    {val}: {[n for n,_ in top]}")

        # Semantic clustering
        if SKLEARN_AVAILABLE:
            self._cluster_neurons("R", 8)

    def _cluster_neurons(self, nt, n_clusters):
        print(f"\n[Semantic Clustering]")
        all_toks = list(self.total_token_counts.keys())[:500]
        neurons = [k for k in self.neuron_token_map if k.startswith(f"{nt}_")]

        feats, valid = [], []
        for nk in neurons:
            tc = self.neuron_token_map[nk]["tokens"]
            total = sum(tc.values())
            if total < 50: continue
            feats.append([tc.get(t,0)/total for t in all_toks])
            valid.append(nk)

        if len(feats) < n_clusters: return

        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(feats)
        self.semantic_clusters[nt] = defaultdict(list)
        for nk, lab in zip(valid, labels):
            self.semantic_clusters[nt][lab].append(nk)

        print(f"  Clustered {len(valid)} neurons into {n_clusters} groups")
        for cid in range(min(3, n_clusters)):
            members = self.semantic_clusters[nt][cid][:3]
            toks = Counter()
            for nk in members: toks.update(self.neuron_token_map[nk]["tokens"])
            print(f"    Cluster {cid}: {[t for t,_ in toks.most_common(5)]}")

    def analyze_knowledge(self):
        print(f"\n{'='*60}\nStep 3: Knowledge Structure\n{'='*60}")

        # Entities
        print("\n[Entities]")
        for et in ["PERSON","ORG","GPE","DATE","MONEY"]:
            if et in self.entity_neuron_counts:
                top = self.entity_neuron_counts[et]["R"].most_common(3)
                ex = self.entity_examples.get(et, [])[:3]
                if top: print(f"  {et}: neurons {[n for n,_ in top]}, ex: {ex}")

        # Knowledge patterns
        print("\n[Knowledge Patterns]")
        for name, data in [("Capitalized", self.capitalized_neurons), ("Numeric", self.numeric_neurons), ("Temporal", self.temporal_word_neurons)]:
            if data.get("R"):
                top = data["R"].most_common(3)
                print(f"  {name}: R neurons {[n for n,_ in top]}")

    def analyze_behavior(self):
        print(f"\n{'='*60}\nStep 4: Behavioral Patterns\n{'='*60}")

        # Layer
        print("\n[Layer Roles]")
        for layer in sorted(self.layer_neuron_counts.keys()):
            stats = {nt: len([c for c in self.layer_neuron_counts[layer][nt].values() if c>0]) for nt in ["FR","FV","R","V"]}
            print(f"  Layer {layer}: {stats}")

        # Position
        print("\n[Position Effects]")
        for pos_bin in ["first","start","middle","end"]:
            if pos_bin in self.position_neuron_counts:
                stats = {nt: len([c for c in self.position_neuron_counts[pos_bin][nt].values() if c>0]) for nt in ["R","FR"]}
                print(f"  {pos_bin}: {stats}")

        # Context length
        print("\n[Context Length]")
        for lb in sorted(self.context_length_counts.keys()):
            stats = {nt: len([c for c in self.context_length_counts[lb][nt].values() if c>0]) for nt in ["R","FR"]}
            print(f"  {lb}: {stats}")

    def analyze_specialization(self):
        print(f"\n{'='*60}\nStep 5: Pool Specialization\n{'='*60}")

        # QK vs V
        print("\n[QK vs V]")
        qk_n = set()
        for pos, neurons in self.qk_v_comparison["qk_neurons"].items(): qk_n.update(neurons.keys())
        v_n = set()
        for pos, neurons in self.qk_v_comparison["v_neurons"].items(): v_n.update(neurons.keys())
        print(f"  QK (FR+R): {len(qk_n)} unique neurons")
        print(f"  V (FV+V): {len(v_n)} unique neurons")

        # Cross-layer
        print("\n[Cross-layer Reuse]")
        multi = [(k, len([l for l,c in layers.items() if c>0])) for k, layers in self.neuron_layer_distribution.items()]
        multi = sorted(multi, key=lambda x:-x[1])[:5]
        for nk, nl in multi: print(f"  {nk}: {nl} layers")

        # Specialized neurons
        print("\n[Specialized Neurons]")
        for nt in ["FR","FV","R","V"]:
            spec = []
            for nk, tc in self.neuron_pos_counts.items():
                if not nk.startswith(f"{nt}_"): continue
                counts = tc[nt]
                total = sum(counts.values())
                if total < 100: continue
                probs = np.array(list(counts.values())) / total
                ent = -np.sum(probs * np.log2(probs + 1e-10))
                max_ent = np.log2(len(counts)) if len(counts) > 0 else 1
                score = 1 - ent/max_ent if max_ent > 0 else 0
                if score > 0.3:
                    top_pos = counts.most_common(1)
                    spec.append({"idx": int(nk.split("_")[1]), "spec": round(score,3), "pos": top_pos[0][0] if top_pos else None})
            spec.sort(key=lambda x:-x["spec"])
            self.specialized_neurons[nt] = spec
            print(f"  {nt}: {len(spec)} specialized")
            for s in spec[:3]: print(f"    {nt}_{s['idx']}: spec={s['spec']}, pos={s['pos']}")

    def run_ablation(self, dataloader, groups=None, max_batches=50):
        print(f"\n{'='*60}\nStep 6: Ablation Study\n{'='*60}")

        if groups is None:
            groups = {}

            # POS-based groups
            for pos in ["NOUN", "VERB", "DET", "ADJ", "ADP"]:
                if pos in self.pos_neuron_counts:
                    top = [n for n,_ in self.pos_neuron_counts[pos]["R"].most_common(5)]
                    if top: groups[f"POS_{pos}"] = [("R", top)]

            # Dependency-based groups
            for dep in ["nsubj", "ROOT", "dobj", "amod"]:
                if dep in self.dep_neuron_counts:
                    top = [n for n,_ in self.dep_neuron_counts[dep]["R"].most_common(5)]
                    if top: groups[f"DEP_{dep}"] = [("R", top)]

            # RANDOM CONTROL - Critical for proving specificity
            all_r_neurons = set()
            for pos_counts in self.pos_neuron_counts.values():
                all_r_neurons.update(pos_counts["R"].keys())
            if all_r_neurons:
                import random
                random.seed(42)
                random_neurons = random.sample(list(all_r_neurons), min(5, len(all_r_neurons)))
                groups["RANDOM_CONTROL"] = [("R", random_neurons)]

        if not groups:
            print("No groups")
            return {}

        print(f"Groups: {list(groups.keys())}")

        baseline = self._evaluate(dataloader, max_batches)
        self.ablation_results = {"baseline": baseline}
        print(f"Baseline: loss={baseline['loss']:.4f}, acc={baseline['accuracy']:.4f}")

        for name, neurons in groups.items():
            for nt, indices in neurons:
                self._ablate(nt, indices)

            ablated = self._evaluate(dataloader, max_batches)
            loss_pct = (ablated["loss"] - baseline["loss"]) / baseline["loss"] * 100
            acc_pct = (baseline["accuracy"] - ablated["accuracy"]) / baseline["accuracy"] * 100 if baseline["accuracy"] > 0 else 0

            self.ablation_results[name] = {"loss_increase_pct": round(loss_pct,2), "accuracy_drop_pct": round(acc_pct,2)}
            print(f"  {name}: loss +{loss_pct:.1f}%, acc -{acc_pct:.1f}%")
            self._restore()

        return self.ablation_results

    def _ablate(self, nt, indices):
        patterns = {"FR":["feature_r"],"FV":["feature_v"],"R":["relational"],"V":["value"]}
        for name, param in self.model.named_parameters():
            if any(p in name.lower() for p in patterns.get(nt,[])) and "norm" not in name.lower():
                if name not in self.original_weights:
                    self.original_weights[name] = param.data.clone()
                for i in indices:
                    if i < param.shape[0]: param.data[i] = 0

    def _restore(self):
        for name, orig in self.original_weights.items():
            for pn, p in self.model.named_parameters():
                if pn == name: p.data = orig.clone()
        self.original_weights = {}

    @torch.no_grad()
    def _evaluate(self, dataloader, max_batches):
        loss, correct, total = 0, 0, 0
        for i, batch in enumerate(dataloader):
            if i >= max_batches: break
            if isinstance(batch, dict): ids = batch['input_ids'].to(self.device)
            elif isinstance(batch, (list,tuple)): ids = batch[0].to(self.device)
            else: ids = batch.to(self.device)

            out = self.model(ids)
            # Handle tuple output from return_routing_info
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out.logits if hasattr(out, 'logits') else out
            sl = logits[...,:-1,:].contiguous()
            st = ids[...,1:].contiguous().long()  # .long() for cross_entropy

            loss += F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1), reduction='sum').item()
            correct += (sl.argmax(-1) == st).sum().item()
            total += st.numel()
        return {"loss": loss/total, "accuracy": correct/total, "perplexity": np.exp(loss/total)}

    def generate_viz(self, output_dir=None):
        print(f"\n{'='*60}\nStep 7: Visualizations\n{'='*60}")
        if output_dir: self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._plot_heatmap(self.pos_neuron_counts, "pos", "R", "POS")
        self._plot_heatmap(self.dep_neuron_counts, "dep", "R", "Dependency")
        self._plot_heatmap(self.entity_neuron_counts, "entity", "R", "Entity")
        self._plot_morph()
        self._plot_layers()
        self._plot_positions()
        self._plot_ablation()
        print(f"Saved to: {self.output_dir}")

    def _plot_heatmap(self, data, name, nt, title, top_k=30):
        labels = [l for l in data if sum(data[l][nt].values()) > 50]
        if not labels: return
        all_n = Counter()
        for l in labels: all_n.update(data[l][nt])
        neurons = [n for n,_ in all_n.most_common(top_k)]
        if not neurons: return

        mat = np.zeros((len(labels), len(neurons)))
        for i, l in enumerate(labels):
            total = sum(data[l][nt].values())
            for j, n in enumerate(neurons):
                mat[i,j] = data[l][nt].get(n,0) / total if total else 0

        fig, ax = plt.subplots(figsize=(14, max(6, len(labels)*0.4)))
        sns.heatmap(mat, xticklabels=neurons, yticklabels=labels, cmap="YlOrRd", ax=ax)
        ax.set_title(f"{title} → {nt} Neurons")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}_{nt}_heatmap.png"), dpi=150)
        plt.close()
        print(f"  Saved: {name}_{nt}_heatmap.png")

    def _plot_morph(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (name, data) in zip(axes, [("Tense", self.tense_neurons), ("Number", self.number_neurons), ("VerbForm", self.verb_form_neurons)]):
            if not data:
                ax.set_title(name)
                continue
            vals, counts = [], []
            for v, neurons in data.items():
                for n, c in neurons.most_common(5):
                    vals.append(f"{v}_{n}")
                    counts.append(c)
            if vals:
                ax.barh(vals[:15], counts[:15])
                ax.set_title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "morphology.png"), dpi=150)
        plt.close()
        print(f"  Saved: morphology.png")

    def _plot_layers(self):
        layers = sorted(self.layer_neuron_counts.keys())
        if not layers: return
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, nt in zip(axes.flatten(), ["FR","FV","R","V"]):
            counts = [len([c for c in self.layer_neuron_counts[l][nt].values() if c>0]) for l in layers]
            ax.bar(layers, counts)
            ax.set_title(nt)
            ax.set_xlabel("Layer")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_usage.png"), dpi=150)
        plt.close()
        print(f"  Saved: layer_usage.png")

    def _plot_positions(self):
        bins = ["first","start","middle","end"]
        avail = [b for b in bins if b in self.position_neuron_counts]
        if not avail: return
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, nt in zip(axes, ["FR","FV","R","V"]):
            counts = [len([c for c in self.position_neuron_counts[b][nt].values() if c>0]) for b in avail]
            ax.bar(avail, counts)
            ax.set_title(nt)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "position.png"), dpi=150)
        plt.close()
        print(f"  Saved: position.png")

    def _plot_ablation(self):
        groups = [k for k in self.ablation_results if k != "baseline"]
        if not groups: return
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(groups)*0.3)))

        loss = [self.ablation_results[g]["loss_increase_pct"] for g in groups]
        axes[0].barh(groups, loss, color=['coral' if x>0 else 'lightgreen' for x in loss])
        axes[0].axvline(0, color='black', linewidth=0.5)
        axes[0].set_title("Loss Increase %")

        acc = [self.ablation_results[g]["accuracy_drop_pct"] for g in groups]
        axes[1].barh(groups, acc, color=['coral' if x>0 else 'lightgreen' for x in acc])
        axes[1].axvline(0, color='black', linewidth=0.5)
        axes[1].set_title("Accuracy Drop %")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ablation.png"), dpi=150)
        plt.close()
        print(f"  Saved: ablation.png")

    def run_full_analysis(self, dataloader, max_batches=500, ablation_batches=50, output_dir="./interpretability_results", vectorized=True):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60 + "\nDAWN INTERPRETABILITY ANALYSIS\n" + "="*60)

        if vectorized:
            self.collect_data_vectorized(dataloader, max_batches)
        else:
            self.collect_data(dataloader, max_batches)
        self.analyze_linguistics()
        self.analyze_knowledge()
        self.analyze_behavior()
        self.analyze_specialization()
        self.run_ablation(dataloader, max_batches=ablation_batches)
        self.generate_viz()
        self.save_results()
        self.print_neuron_examples(nt="R", top_n=5)
        self.export_for_paper()

        print("\n" + "="*60 + "\nCOMPLETE\n" + "="*60)

    def save_results(self):
        results = {
            "summary": {"tokens": sum(self.total_token_counts.values()), "unique": len(self.token_neuron_map)},
            "specialized": self.specialized_neurons,
            "ablation": self.ablation_results,
        }
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved: results.json")

    def get_neuron_profile(self, nt, idx):
        nk = f"{nt}_{idx}"
        pos_counts = self.neuron_pos_counts[nk][nt]
        total = sum(pos_counts.values())
        probs = np.array(list(pos_counts.values())) / total if total else np.array([])
        ent = -np.sum(probs * np.log2(probs + 1e-10)) if len(probs) else 0
        max_ent = np.log2(len(pos_counts)) if pos_counts else 1
        spec = 1 - ent/max_ent if max_ent else 0

        return {
            "neuron": nk,
            "total": sum(self.neuron_token_map[nk]["tokens"].values()),
            "tokens": self.neuron_token_map[nk]["tokens"].most_common(15),
            "pos": list(pos_counts.most_common(5)),
            "dep": list(self.neuron_dep_counts[nk][nt].most_common(5)),
            "morph": list(self.neuron_morph_counts[nk][nt].most_common(5)),
            "entity": list(self.neuron_entity_counts[nk][nt].most_common(3)),
            "layers": dict(self.neuron_layer_distribution[nk]),
            "spec": round(spec, 3),
            "contexts": self.neuron_token_map[nk]["contexts"][:5],
        }

    def get_neurons_for_pos(self, pos, nt="R", k=10): return self.pos_neuron_counts[pos][nt].most_common(k)
    def get_neurons_for_dep(self, dep, nt="R", k=10): return self.dep_neuron_counts[dep][nt].most_common(k)
    def get_neurons_for_entity(self, et, nt="R", k=10): return self.entity_neuron_counts[et][nt].most_common(k)

    def print_neuron_examples(self, nt="R", top_n=5, examples_per_neuron=3):
        """Print neuron examples in paper-friendly format"""
        print(f"\n{'='*60}\nNeuron Examples ({nt})\n{'='*60}")

        # Get top specialized neurons
        spec_neurons = self.specialized_neurons.get(nt, [])[:top_n]

        for neuron_info in spec_neurons:
            idx = neuron_info['idx']
            nk = f"{nt}_{idx}"
            spec = neuron_info['spec']
            top_pos = neuron_info.get('pos', 'UNK')

            # Get contexts
            contexts = self.neuron_token_map[nk].get("contexts", [])[:examples_per_neuron]
            tokens = self.neuron_token_map[nk]["tokens"].most_common(5)

            print(f"\n{nk} (spec={spec:.3f}, dominant_pos={top_pos}):")
            print(f"  Top tokens: {[t for t,_ in tokens]}")
            for ctx in contexts:
                tok = ctx.get('token', '')
                context = ctx.get('context', '')
                layer = ctx.get('layer', 0)
                # Highlight the token in context
                highlighted = context.replace(tok, f"**{tok}**")
                print(f"  - L{layer}: \"{highlighted}\"")

    def export_for_paper(self, output_path=None):
        """Export key results in paper-friendly format"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "paper_results.json")

        results = {
            "specialized_neurons": self.specialized_neurons,
            "ablation_results": self.ablation_results,
            "top_pos_neurons": {},
            "top_dep_neurons": {},
        }

        # Top neurons per POS
        for pos in ["NOUN", "VERB", "DET", "ADJ", "ADP", "PUNCT"]:
            if pos in self.pos_neuron_counts:
                results["top_pos_neurons"][pos] = {
                    nt: [(n, c) for n, c in self.pos_neuron_counts[pos][nt].most_common(5)]
                    for nt in ["FR", "FV", "R", "V"]
                }

        # Top neurons per DEP
        for dep in ["nsubj", "ROOT", "dobj", "amod", "det", "prep"]:
            if dep in self.dep_neuron_counts:
                results["top_dep_neurons"][dep] = {
                    nt: [(n, c) for n, c in self.dep_neuron_counts[dep][nt].most_common(5)]
                    for nt in ["FR", "FV", "R", "V"]
                }

        # Layer distribution summary
        results["layer_usage"] = {}
        for layer in sorted(self.layer_neuron_counts.keys()):
            results["layer_usage"][int(layer)] = {
                nt: len([c for c in self.layer_neuron_counts[layer][nt].values() if c > 0])
                for nt in ["FR", "FV", "R", "V"]
            }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Paper results exported to: {output_path}")

        return results


def quick_analysis(model, tokenizer, dataloader, device="cuda", max_batches=500):
    interp = DAWNInterpreter(model, tokenizer, device)
    interp.run_full_analysis(dataloader, max_batches)
    return interp


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DAWN Interpretability Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file or directory")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data .pt file")
    parser.add_argument("--output_dir", type=str, default="./analysis", help="Output directory")
    parser.add_argument("--max_batches", type=int, default=500, help="Max batches for data collection")
    parser.add_argument("--ablation_batches", type=int, default=50, help="Max batches for ablation")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Find checkpoint file
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        # Find latest checkpoint in directory
        import glob
        ckpts = glob.glob(os.path.join(ckpt_path, "*.pt")) + glob.glob(os.path.join(ckpt_path, "**/*.pt"), recursive=True)
        if ckpts:
            ckpt_path = max(ckpts, key=os.path.getmtime)
            print(f"Using checkpoint: {ckpt_path}")
        else:
            print(f"No .pt files found in {args.checkpoint}")
            sys.exit(1)

    # Import model
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.model_v16 import DAWN
    except ImportError:
        try:
            from model_v16 import DAWN
        except ImportError:
            print("Error: Cannot import DAWN model. Make sure models/model_v16.py exists.")
            sys.exit(1)

    from transformers import BertTokenizer

    print(f"\n{'='*60}")
    print("DAWN Interpretability Analysis - CLI")
    print(f"{'='*60}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    config = checkpoint.get('config', {})
    print(f"Model config: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}")

    # Init model
    model = DAWN(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load data
    print(f"\nLoading data: {args.val_data}")
    val_data = torch.load(args.val_data)
    if isinstance(val_data, dict):
        input_ids = val_data.get('input_ids', val_data.get('tokens'))
    else:
        input_ids = val_data

    # Handle 1D data (flat tokens) - reshape to 2D
    if input_ids.dim() == 1:
        seq_len = config.get('max_seq_len', 128)
        n_tokens = input_ids.shape[0]
        n_seqs = n_tokens // seq_len
        input_ids = input_ids[:n_seqs * seq_len].view(n_seqs, seq_len)
        print(f"Reshaped 1D data to: {input_ids.shape}")
    else:
        print(f"Data shape: {input_ids.shape}")

    # Create dataloader
    dataset = torch.utils.data.TensorDataset(input_ids)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Run analysis
    interp = DAWNInterpreter(model, tokenizer, args.device)
    interp.run_full_analysis(
        dataloader,
        max_batches=args.max_batches,
        ablation_batches=args.ablation_batches,
        output_dir=args.output_dir
    )

    print(f"\nDone! Results saved to: {args.output_dir}")
