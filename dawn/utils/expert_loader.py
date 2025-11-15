"""
Expert Checkpoint Manager

Clean API for managing and loading expert checkpoints.
Supports:
- Checkpoint discovery and listing
- Expert initialization from checkpoints
- Weight transfer between models
- Feature extraction
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint"""
    path: Path
    expert_name: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    timestamp: Optional[datetime] = None
    model_config: Optional[Dict] = None

    @property
    def name(self) -> str:
        """Human-readable checkpoint name"""
        parts = [self.expert_name]
        if self.epoch is not None:
            parts.append(f"epoch{self.epoch}")
        if self.step is not None:
            parts.append(f"step{self.step}")
        if self.accuracy is not None:
            parts.append(f"acc{self.accuracy:.1%}")
        return "_".join(parts)

    def __str__(self) -> str:
        info = f"{self.expert_name}"
        if self.accuracy is not None:
            info += f" | Acc: {self.accuracy:.2%}"
        if self.step is not None:
            info += f" | Step: {self.step}"
        if self.path:
            info += f" | {self.path.name}"
        return info


class CheckpointRegistry:
    """
    Discover and manage expert checkpoints

    Usage:
        registry = CheckpointRegistry('/path/to/checkpoints')
        checkpoints = registry.list_checkpoints()
        best_ckpt = registry.get_best('lexical')
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Args:
            checkpoint_dir: Root directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self._cache: Dict[str, List[CheckpointInfo]] = {}

    def scan(self, expert_name: Optional[str] = None, force_refresh: bool = False) -> List[CheckpointInfo]:
        """
        Scan directory for checkpoints

        Args:
            expert_name: Filter by expert name (None = all)
            force_refresh: Re-scan even if cached

        Returns:
            List of checkpoint info
        """
        if not force_refresh and expert_name in self._cache:
            return self._cache[expert_name]

        checkpoints = []

        if not self.checkpoint_dir.exists():
            print(f"⚠️  Checkpoint directory not found: {self.checkpoint_dir}")
            return checkpoints

        # Find all .pt files
        for ckpt_path in self.checkpoint_dir.rglob("*.pt"):
            try:
                # Load checkpoint metadata
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

                # Extract metadata
                info = CheckpointInfo(
                    path=ckpt_path,
                    expert_name=ckpt.get('expert_name', self._infer_expert_name(ckpt_path)),
                    step=ckpt.get('step'),
                    epoch=ckpt.get('epoch'),
                    accuracy=ckpt.get('accuracy', ckpt.get('mlm_acc')),
                    loss=ckpt.get('loss'),
                    model_config=ckpt.get('model_config')
                )

                # Filter by expert name if specified
                if expert_name is None or info.expert_name == expert_name:
                    checkpoints.append(info)

            except Exception as e:
                print(f"⚠️  Failed to load {ckpt_path.name}: {e}")
                continue

        # Sort by accuracy (descending)
        checkpoints.sort(key=lambda x: x.accuracy or 0, reverse=True)

        # Cache results
        key = expert_name or 'all'
        self._cache[key] = checkpoints

        return checkpoints

    def list_checkpoints(self, expert_name: Optional[str] = None) -> None:
        """
        Print available checkpoints

        Args:
            expert_name: Filter by expert name
        """
        checkpoints = self.scan(expert_name)

        if not checkpoints:
            print("No checkpoints found.")
            return

        print(f"\n📦 Found {len(checkpoints)} checkpoint(s) in {self.checkpoint_dir}\n")

        # Group by expert
        by_expert: Dict[str, List[CheckpointInfo]] = {}
        for ckpt in checkpoints:
            if ckpt.expert_name not in by_expert:
                by_expert[ckpt.expert_name] = []
            by_expert[ckpt.expert_name].append(ckpt)

        # Print grouped
        for exp_name, ckpts in sorted(by_expert.items()):
            print(f"  {exp_name}:")
            for ckpt in ckpts:
                print(f"    • {ckpt}")
            print()

    def get_best(self, expert_name: str) -> Optional[CheckpointInfo]:
        """
        Get best checkpoint for expert (highest accuracy)

        Args:
            expert_name: Expert name

        Returns:
            Best checkpoint info or None
        """
        checkpoints = self.scan(expert_name)

        if not checkpoints:
            return None

        # Already sorted by accuracy
        return checkpoints[0]

    def get_latest(self, expert_name: str) -> Optional[CheckpointInfo]:
        """
        Get latest checkpoint for expert (by step/epoch)

        Args:
            expert_name: Expert name

        Returns:
            Latest checkpoint info or None
        """
        checkpoints = self.scan(expert_name)

        if not checkpoints:
            return None

        # Sort by step/epoch
        checkpoints.sort(
            key=lambda x: (x.epoch or 0, x.step or 0),
            reverse=True
        )

        return checkpoints[0]

    def get_checkpoint(
        self,
        expert_name: str,
        selector: str = 'best'
    ) -> Optional[CheckpointInfo]:
        """
        Get checkpoint by selector

        Args:
            expert_name: Expert name
            selector: 'best', 'latest', or checkpoint filename

        Returns:
            Checkpoint info or None
        """
        if selector == 'best':
            return self.get_best(expert_name)
        elif selector == 'latest':
            return self.get_latest(expert_name)
        else:
            # Search by filename
            checkpoints = self.scan(expert_name)
            for ckpt in checkpoints:
                if selector in ckpt.path.name:
                    return ckpt
            return None

    def _infer_expert_name(self, path: Path) -> str:
        """Infer expert name from path"""
        # Try to extract from filename
        name = path.stem.lower()

        for expert in ['lexical', 'syntactic', 'semantic', 'pragmatic']:
            if expert in name:
                return expert

        # Try parent directory
        if path.parent.name in ['lexical', 'syntactic', 'semantic', 'pragmatic']:
            return path.parent.name

        return 'unknown'


class ExpertLoader:
    """
    Load and initialize experts from checkpoints

    Usage:
        loader = ExpertLoader()
        expert = loader.load_expert('lexical', checkpoint_path)

        # Or auto-discover
        registry = CheckpointRegistry('/path/to/checkpoints')
        expert = loader.load_from_registry(registry, 'lexical', selector='best')
    """

    @staticmethod
    def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict:
        """
        Load checkpoint file

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        return checkpoint

    @staticmethod
    def load_expert(
        expert_name: str,
        checkpoint_path: Union[str, Path],
        config: Optional[Dict] = None,
        device: str = 'cuda',
        freeze_base: bool = False,
        strict: bool = False
    ) -> nn.Module:
        """
        Load expert from checkpoint

        Args:
            expert_name: Name of expert
            checkpoint_path: Path to checkpoint
            config: Model config (if None, use config from checkpoint)
            device: Device to load on
            freeze_base: Freeze base weights
            strict: Strict weight loading

        Returns:
            Loaded expert model
        """
        from models.delta_expert import DeltaExpert

        # Load checkpoint
        checkpoint = ExpertLoader.load_checkpoint(checkpoint_path)

        # Get config
        if config is None:
            config = checkpoint.get('model_config')
            if config is None:
                raise ValueError("No config provided and checkpoint doesn't contain model_config")

        # Create expert
        expert = DeltaExpert(config)

        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        try:
            expert.load_state_dict(state_dict, strict=strict)
            print(f"✓ Loaded {expert_name} from {Path(checkpoint_path).name}")
        except Exception as e:
            print(f"⚠️  Weight loading failed: {e}")
            if strict:
                raise

            # Try partial loading
            stats = ExpertLoader._load_partial(expert, state_dict, expert_name)
            print(f"  Loaded {stats['matched']}/{stats['total']} weights")

        # Freeze if requested
        if freeze_base:
            expert.freeze_base()
            print(f"  🔒 Frozen base weights")

        # Move to device
        expert = expert.to(device)

        return expert

    @staticmethod
    def load_from_registry(
        registry: CheckpointRegistry,
        expert_name: str,
        selector: str = 'best',
        config: Optional[Dict] = None,
        device: str = 'cuda',
        freeze_base: bool = False
    ) -> Optional[nn.Module]:
        """
        Load expert using checkpoint registry

        Args:
            registry: CheckpointRegistry instance
            expert_name: Expert name
            selector: 'best', 'latest', or filename
            config: Model config (optional)
            device: Device to load on
            freeze_base: Freeze base weights

        Returns:
            Loaded expert or None if not found
        """
        # Get checkpoint
        ckpt_info = registry.get_checkpoint(expert_name, selector)

        if ckpt_info is None:
            print(f"❌ No checkpoint found for {expert_name} (selector: {selector})")
            return None

        print(f"Loading {expert_name} from: {ckpt_info}")

        # Use checkpoint config if not provided
        if config is None and ckpt_info.model_config is not None:
            config = ckpt_info.model_config

        # Load expert
        expert = ExpertLoader.load_expert(
            expert_name=expert_name,
            checkpoint_path=ckpt_info.path,
            config=config,
            device=device,
            freeze_base=freeze_base
        )

        return expert

    @staticmethod
    def load_all_experts(
        registry: CheckpointRegistry,
        expert_names: List[str],
        selector: str = 'best',
        config: Optional[Dict] = None,
        device: str = 'cuda'
    ) -> Dict[str, nn.Module]:
        """
        Load multiple experts

        Args:
            registry: CheckpointRegistry instance
            expert_names: List of expert names
            selector: Checkpoint selector
            config: Shared config (optional)
            device: Device to load on

        Returns:
            Dictionary of expert_name -> expert model
        """
        experts = {}

        for expert_name in expert_names:
            expert = ExpertLoader.load_from_registry(
                registry=registry,
                expert_name=expert_name,
                selector=selector,
                config=config,
                device=device
            )

            if expert is not None:
                experts[expert_name] = expert

        print(f"\n✓ Loaded {len(experts)}/{len(expert_names)} experts")

        return experts

    @staticmethod
    def _load_partial(
        expert: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        expert_name: str
    ) -> Dict:
        """Load weights partially (skip incompatible)"""
        expert_state = expert.state_dict()

        matched = 0
        skipped = 0

        for key, param in expert_state.items():
            if key in state_dict:
                ckpt_param = state_dict[key]
                if ckpt_param.shape == param.shape:
                    expert_state[key] = ckpt_param
                    matched += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        expert.load_state_dict(expert_state)

        return {
            'matched': matched,
            'skipped': skipped,
            'total': len(expert_state)
        }


class FeatureExtractor:
    """
    Extract features from loaded expert

    Usage:
        extractor = FeatureExtractor(expert)
        features = extractor(input_ids, attention_mask)
    """

    def __init__(self, expert: nn.Module, device: str = 'cuda'):
        """
        Args:
            expert: Expert model
            device: Device to run on
        """
        self.expert = expert.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer: str = 'final'
    ) -> torch.Tensor:
        """
        Extract features

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            layer: 'final', 'all_steps', 'embedding'

        Returns:
            Features tensor or list of tensors
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if layer == 'all_steps':
            return self.expert(input_ids, attention_mask, return_all_steps=True)
        elif layer == 'embedding':
            return self.expert.get_embeddings(input_ids, attention_mask)
        else:
            return self.expert(input_ids, attention_mask)

    def get_sentence_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = 'mean'
    ) -> torch.Tensor:
        """
        Get sentence-level embeddings

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            pooling: 'mean', 'max', or 'cls'

        Returns:
            Sentence embeddings [batch, hidden]
        """
        features = self(input_ids, attention_mask)  # [B, L, D]

        if pooling == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).to(self.device)
            sum_embeddings = (features * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

        elif pooling == 'max':
            # Max pooling
            features[attention_mask == 0] = -1e9
            return features.max(dim=1)[0]

        elif pooling == 'cls':
            # CLS token (first token)
            return features[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling: {pooling}")


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_load(
    checkpoint_dir: Union[str, Path],
    expert_name: str,
    selector: str = 'best',
    device: str = 'cuda'
) -> nn.Module:
    """
    Quick load expert from checkpoint directory

    Args:
        checkpoint_dir: Checkpoint directory
        expert_name: Expert name
        selector: 'best', 'latest', or filename
        device: Device to load on

    Returns:
        Loaded expert

    Example:
        expert = quick_load('/path/to/checkpoints', 'lexical')
    """
    registry = CheckpointRegistry(checkpoint_dir)
    loader = ExpertLoader()

    expert = loader.load_from_registry(
        registry=registry,
        expert_name=expert_name,
        selector=selector,
        device=device
    )

    return expert


def load_phase1_experts(
    checkpoint_dir: Union[str, Path],
    expert_names: Optional[List[str]] = None,
    selector: str = 'best',
    device: str = 'cuda'
) -> Dict[str, nn.Module]:
    """
    Load all Phase 1 experts

    Args:
        checkpoint_dir: Phase 1 checkpoint directory
        expert_names: List of expert names (default: ['lexical', 'syntactic', 'semantic'])
        selector: Checkpoint selector
        device: Device to load on

    Returns:
        Dictionary of experts

    Example:
        experts = load_phase1_experts('/path/to/phase1_checkpoints')
        lexical = experts['lexical']
    """
    if expert_names is None:
        expert_names = ['lexical', 'syntactic', 'semantic']

    registry = CheckpointRegistry(checkpoint_dir)
    loader = ExpertLoader()

    experts = loader.load_all_experts(
        registry=registry,
        expert_names=expert_names,
        selector=selector,
        device=device
    )

    return experts


# Example usage
if __name__ == "__main__":
    """
    Example: Load experts from checkpoint directory
    """

    # Example 1: List all checkpoints
    registry = CheckpointRegistry('/content/drive/MyDrive/pnn/checkpoints')
    registry.list_checkpoints()

    # Example 2: Load single expert
    expert = quick_load(
        checkpoint_dir='/content/drive/MyDrive/pnn/checkpoints',
        expert_name='lexical',
        selector='best'
    )
    print(f"Loaded expert: {expert}")

    # Example 3: Load all Phase 1 experts
    experts = load_phase1_experts(
        checkpoint_dir='/content/drive/MyDrive/dawn/phase1_checkpoints',
        expert_names=['lexical', 'syntactic', 'semantic']
    )
    print(f"Loaded {len(experts)} experts")

    # Example 4: Feature extraction
    extractor = FeatureExtractor(experts['lexical'])

    dummy_input = torch.randint(0, 30522, (2, 128))
    dummy_mask = torch.ones(2, 128)

    # Token-level features
    features = extractor(dummy_input, dummy_mask)
    print(f"Token features: {features.shape}")

    # Sentence embeddings
    sentence_emb = extractor.get_sentence_embeddings(dummy_input, dummy_mask, pooling='mean')
    print(f"Sentence embeddings: {sentence_emb.shape}")
