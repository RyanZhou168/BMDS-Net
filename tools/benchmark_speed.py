"""
Inference Speed Benchmark Tool.

Calculates Latency (ms) and FPS (Frames Per Second) for BMDS-Net and baselines.
Reproduces 'Table 2: Inference efficiency comparison' in the paper.
"""

import os
import time
import argparse
import yaml
import torch
import pandas as pd
from bmds_net.models import create_model

def measure_throughput(model, input_tensor, iterations=100, warmup=20):
    """
    Measure model throughput.
    """
    # 1. Warmup
    print("  Warmup...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # 2. Benchmark
    print(f"  Running {iterations} iterations...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000 # ms
    fps = iterations / total_time
    
    return avg_latency, fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to model config')
    parser.add_argument('--input_size', type=int, nargs=3, default=[128, 128, 128], help='Input ROI size')
    args = parser.parse_args()

    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = config['model']['name']
    
    print(f"==================================================")
    print(f" Benchmarking Model: {name}")
    print(f" Input Size: {args.input_size}")
    print(f" Device: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")

    # Create Model
    # Important: Disable gradient checkpointing for pure inference speed test
    # to measure the raw forward pass capability.
    model_args = config['model']
    model_args['use_checkpoint'] = False 
    
    try:
        model = create_model(name, **model_args)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[Error] Failed to create model {name}: {e}")
        return

    # Create dummy input: [Batch=1, C=4, H, W, D]
    input_tensor = torch.randn(1, 4, *args.input_size).to(device)
    
    # Run Benchmark
    latency, fps = measure_throughput(model, input_tensor)
    
    print(f"\n[Results] {name}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  FPS:     {fps:.2f}")
    
    # Save results
    os.makedirs('benchmark_results', exist_ok=True)
    df = pd.DataFrame([{
        'Model': name,
        'Input_Size': str(args.input_size),
        'Latency_ms': latency,
        'FPS': fps
    }])
    csv_path = os.path.join('benchmark_results', f'speed_{name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

if __name__ == '__main__':
    main()
