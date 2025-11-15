"""
Hardware Configuration

Auto-detect hardware and adjust settings accordingly.
"""

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class HardwareConfig:
    """Hardware-specific configuration"""
    device: str
    gpu_name: Optional[str] = None
    vram_gb: Optional[float] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    mixed_precision: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def __str__(self) -> str:
        if self.device == "cpu":
            return f"Device: CPU, Batch: {self.batch_size}"
        else:
            return (
                f"Device: {self.gpu_name} ({self.vram_gb:.1f}GB), "
                f"Batch: {self.batch_size} x {self.gradient_accumulation_steps} = {self.effective_batch_size}, "
                f"Workers: {self.num_workers}, "
                f"Mixed Precision: {self.mixed_precision}"
            )


def detect_hardware() -> HardwareConfig:
    """
    Auto-detect GPU and return appropriate configuration
    
    Returns:
        HardwareConfig with optimal settings for detected hardware
    """
    
    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available, using CPU defaults")
        return HardwareConfig(
            device="cpu",
            batch_size=8,
            gradient_accumulation_steps=1,
            num_workers=0,
            mixed_precision=False,
        )
    
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected, using CPU")
        return HardwareConfig(
            device="cpu",
            batch_size=8,
            gradient_accumulation_steps=1,
            num_workers=0,
            mixed_precision=False,
        )
    
    # Get GPU info
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_memory / (1024**3)
    gpu_name = device_props.name
    
    print(f"✅ GPU detected: {gpu_name} ({vram_gb:.1f}GB VRAM)")
    
    # Configure based on VRAM
    # Note: With gradient accumulation, batch_size is what fits in GPU memory
    if vram_gb >= 70:  # A100 80GB, H100 80GB class
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=320,  # Balanced for stability and utilization
            gradient_accumulation_steps=1,  # No accumulation for faster training
            num_workers=8,
            mixed_precision=True,
        )
        print("🚀🚀 Ultra high-end GPU detected - DAWN optimized settings (80GB, batch=320x1=320)")

    elif vram_gb >= 38:  # A100 40GB class (39.6GB actual)
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=128,  # Balanced for stability and utilization
            gradient_accumulation_steps=1,  # No accumulation for faster training
            num_workers=8,
            mixed_precision=True,
        )
        print("🚀 High-end GPU detected - DAWN optimized settings (40GB, batch=128x1=128)")

    elif vram_gb >= 24:  # A10, RTX 3090/4090 class
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=12,  # DAWN uses more memory than PNN
            gradient_accumulation_steps=1,  # No accumulation (test stability)
            num_workers=8,
            mixed_precision=True,
        )
        print("💪 Mid-high GPU detected - DAWN optimized settings (24GB, batch=12x1=12)")

    elif vram_gb >= 16:  # T4, RTX 4060 Ti class
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=6,  # DAWN uses more memory than PNN
            gradient_accumulation_steps=1,  # No accumulation (test stability)
            num_workers=4,
            mixed_precision=True,
        )
        print("👍 Mid-range GPU detected - DAWN optimized settings (16GB, batch=6x1=6)")

    elif vram_gb >= 8:  # GTX 1080, RTX 2060 class
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=4,  # Increased batch size
            gradient_accumulation_steps=1,  # No accumulation (test stability)
            num_workers=2,
            mixed_precision=True,
        )
        print("⚠️  Low VRAM GPU - using conservative settings (DAWN optimized)")

    else:  # < 8GB
        config = HardwareConfig(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            batch_size=2,  # Increased batch size
            gradient_accumulation_steps=1,  # No accumulation (test stability)
            num_workers=2,
            mixed_precision=True,
        )
        print("⚠️  Very low VRAM - using minimal settings (DAWN optimized)")
    
    return config


def get_hardware_config(override_batch_size: Optional[int] = None) -> HardwareConfig:
    """
    Get hardware configuration with optional overrides
    
    Args:
        override_batch_size: Override auto-detected batch size
        
    Returns:
        HardwareConfig instance
    """
    config = detect_hardware()
    
    if override_batch_size is not None:
        print(f"🔧 Overriding batch size: {config.batch_size} -> {override_batch_size}")
        config.batch_size = override_batch_size
    
    print(f"\n📊 Final hardware config:\n{config}\n")
    
    return config


# Auto-detect on import (can be overridden)
DEFAULT_HARDWARE = detect_hardware()


if __name__ == "__main__":
    # Test hardware detection
    print("="*70)
    print("Hardware Detection Test")
    print("="*70)
    
    hw = get_hardware_config()
    
    print("\nHardware Configuration:")
    print(f"  Device: {hw.device}")
    if hw.gpu_name:
        print(f"  GPU: {hw.gpu_name}")
        print(f"  VRAM: {hw.vram_gb:.1f}GB")
    print(f"  Batch size: {hw.batch_size}")
    print(f"  Gradient accumulation: {hw.gradient_accumulation_steps}")
    print(f"  Effective batch size: {hw.effective_batch_size}")
    print(f"  Num workers: {hw.num_workers}")
    print(f"  Mixed precision: {hw.mixed_precision}")
    
    print("\n" + "="*70)
