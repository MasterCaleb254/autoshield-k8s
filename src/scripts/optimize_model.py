# scripts/optimize_model.py
"""
Optimize model for <1ms inference latency.
"""
import torch
from torch.quantization import quantize_dynamic, quantize_fx
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig, QConfigMapping

def optimize_model_for_latency(model_path: str, output_path: str, sample_input: torch.Tensor):
    """Apply optimizations for faster inference"""
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create and load model
    from autoshield.detector.model import create_model
    model = create_model(
        model_type=checkpoint['model_type'],
        **checkpoint['model_config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 1. Fuse modules (if using CNN)
    if hasattr(model, 'features'):
        model = torch.quantization.fuse_modules(model, 
            [['features.0', 'features.1', 'features.2']],  # Adjust based on your model
            inplace=True)
    
    # 2. Dynamic quantization (more aggressive)
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.Conv1d},  # More layers
        dtype=torch.qint8
    )
    
    # 3. FX Graph Mode Quantization (more accurate)
    qconfig = get_default_qconfig('qnnpack')  # Better for mobile/CPU
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # Prepare model
    model_prepared = prepare_fx(quantized_model, qconfig_mapping, (sample_input,))
    # Calibrate (if needed)
    with torch.no_grad():
        model_prepared(sample_input)
    # Convert
    model_quantized = convert_fx(model_prepared)
    
    # 4. Script with optimizations
    model_scripted = torch.jit.optimize_for_inference(
        torch.jit.script(model_quantized)
    )
    
    # 5. Save with _save_for_lite_interpreter for mobile/edge
    model_scripted._save_for_lite_interpreter(output_path)
    
    print(f"Optimized model saved to {output_path}")
    return model_scripted