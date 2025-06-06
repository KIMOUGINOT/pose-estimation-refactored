import argparse
import time
import torch
import torch.nn as nn
from thop import profile, clever_format
from tabulate import tabulate

def benchmark_model(model_path, input_shape=(1, 3, 256, 192), warmup=10, runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_shape).to(device)

    with torch.no_grad():
        params = sum(p.numel() for p in model.parameters())
        params = f"{params/1e6:.2f} M"  

    for _ in range(warmup):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.perf_counter()
    mean_time = (end - start) / runs * 1000  # in ms

    # --- Print summary ---
    info_table = [
        ["Model", model_path],
        ["Device", device.type],
        ["Input shape", str(input_shape)],
        ["Params", params],
        ["Mean Inference Time", f"{mean_time:.2f} ms"],
        ["FPS (approx)", f"{1000/mean_time:.2f}"],
    ]

    print("\n" + tabulate(info_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a TorchScript model")
    parser.add_argument("--model", type=str, required=True, help="Path to the TorchScript .pt model")
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=192, help="Input width")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs to average inference time")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the sample")
    args = parser.parse_args()

    input_shape = (args.batch_size, 3, args.height, args.width)
    benchmark_model(args.model, input_shape=input_shape, runs=args.runs)
