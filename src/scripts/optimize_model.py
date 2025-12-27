# scripts/optimize_model.py
"""
Optimize model for <1ms inference latency.
"""
import argparse
import os
import torch
from torch.quantization import quantize_dynamic

def optimize_model_for_latency(model_path: str, output_path: str, sample_input: torch.Tensor):
    """Export an optimized TorchScript artifact for low-latency inference.

    Notes:
    - Uses safe dynamic quantization for Linear/LSTM (Conv1d dynamic quantization is not supported).
    - Exports a regular TorchScript .pt that can be loaded via torch.jit.load.
    """
    checkpoint = torch.load(model_path, map_location='cpu')

    from autoshield.detector.model import create_model
    model = create_model(
        model_type=checkpoint['model_type'],
        **checkpoint['model_config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prefer qnnpack if available (good CPU perf on many builds)
    try:
        torch.backends.quantized.engine = 'qnnpack'
    except Exception:
        pass

    # Dynamic quantization (safe modules)
    try:
        model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8
        )
    except Exception:
        # If quantization fails, still export TorchScript
        pass

    # TorchScript export
    with torch.no_grad():
        try:
            scripted = torch.jit.script(model)
        except Exception:
            scripted = torch.jit.trace(model, sample_input)

    scripted = torch.jit.optimize_for_inference(scripted)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.jit.save(scripted, output_path)
    print(f"Optimized TorchScript model saved to {output_path}")
    return scripted


def _build_sample_input_from_checkpoint(model_path: str) -> torch.Tensor:
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg = checkpoint.get('model_config', {})
    seq_len = int(cfg.get('sequence_length', 20))
    in_features = int(cfg.get('input_features', 12))
    return torch.randn(1, seq_len, in_features)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export optimized TorchScript/quantized model")
    parser.add_argument("--model-path", required=True, help="Path to .pth checkpoint (best_model.pth/final_model.pth)")
    parser.add_argument("--output-path", required=True, help="Path to save TorchScript .pt")
    args = parser.parse_args()

    sample_input = _build_sample_input_from_checkpoint(args.model_path)
    optimize_model_for_latency(args.model_path, args.output_path, sample_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())