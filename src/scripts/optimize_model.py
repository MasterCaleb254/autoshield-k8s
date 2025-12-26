# scripts/optimize_model.py
"""
Optimize model for <1ms inference latency.
"""
import torch
from torch.quantization import quantize_dynamic

def optimize_model_for_latency(model_path: str, output_path: str):
    """Apply optimizations for faster inference"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    from autoshield.detector.model import create_model
    model = create_model(
        model_type=checkpoint['model_type'],
        **checkpoint['model_config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 1. Dynamic quantization (reduces model size, increases speed)
    quantized_model = quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8
    )
    
    # 2. Script/Trace for optimization
    scripted_model = torch.jit.script(quantized_model)
    
    # Save optimized model
    torch.jit.save(scripted_model, output_path)
    print(f"Optimized model saved to {output_path}")