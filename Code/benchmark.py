"""
benchmark.py — Inference speed benchmarking for the EuroSAT classification project.

Measures forward-pass latency and throughput for each trained model on both CPU and GPU.
This addresses the practical deployment question: could these models run in real time on
edge hardware? Reported numbers per model:
    - Latency (ms per image, mean ± std over N runs)
    - Throughput (images per second)
    - Parameter count and trainable parameter count

Methodology:
    - Use synthetic input tensors (correct shape and dtype) — no need to load real data
    - Run several warmup passes before timing (CUDA kernel compilation, BatchNorm warmup)
    - Time many forward passes per model and report mean/std
    - For GPU timing, synchronize before/after each pass (CUDA is asynchronous by default)

Outputs:
    - results/benchmark_summary.csv — full table of numbers
    - Console table summarizing the comparison

Usage:
    python benchmark.py \\
        --models baseline resnet_frozen resnet_finetuned \\
        --model-paths \\
            ../results/baseline_best.pt \\
            ../results/resnet_frozen_best.pt \\
            ../results/resnet_finetuned_best.pt
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from models import get_model


# --- Benchmark configuration ---
# Warmup passes are not timed; they exist to trigger lazy GPU kernel compilation
# and stabilize BatchNorm statistics. Without warmup, the first few forward passes
# can be 10-100x slower than steady-state.
WARMUP_PASSES = 20
TIMED_PASSES = 200
INPUT_SIZE = 224
BATCH_SIZE = 1  # Latency is measured per-image, not per-batch


def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    warmup_passes: int = WARMUP_PASSES,
    timed_passes: int = TIMED_PASSES,
    input_size: int = INPUT_SIZE,
) -> tuple[float, float, float]:
    """
    Time forward passes through the model and return latency statistics.

    For CUDA timing, we use torch.cuda.synchronize() to wait for kernel completion
    before stopping the timer. CUDA operations are asynchronous by default, so naive
    time.perf_counter() calls would just measure how fast the work is queued, not
    how long it actually took to execute.

    Args:
        model:          Model in eval mode, already on the target device.
        device:         CPU or CUDA device.
        warmup_passes:  Number of untimed forward passes before measurement.
        timed_passes:   Number of timed forward passes for the statistics.
        input_size:     Spatial size of the synthetic input (assumes square).

    Returns:
        (mean_ms_per_image, std_ms_per_image, throughput_imgs_per_sec)
    """
    model.eval()
    is_cuda = device.type == "cuda"

    # Synthetic input — same shape and normalization as real inference would use
    dummy_input = torch.randn(BATCH_SIZE, 3, input_size, input_size, device=device)

    # --- Warmup ---
    with torch.no_grad():
        for _ in range(warmup_passes):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

    # --- Timed passes ---
    times_ms = []
    with torch.no_grad():
        for _ in range(timed_passes):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    times_array = np.array(times_ms)
    mean_ms = float(times_array.mean())
    std_ms = float(times_array.std())
    throughput = 1000.0 / mean_ms  # images per second

    return mean_ms, std_ms, throughput


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Return (total_parameters, trainable_parameters) for the model.

    Total counts every parameter tensor's elements regardless of requires_grad.
    Trainable counts only parameters that would be updated during training.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_benchmarks(args: argparse.Namespace) -> None:
    """
    Main driver: load each model, benchmark on CPU and GPU, save results.

    Args:
        args: Parsed command-line arguments.
    """
    if len(args.models) != len(args.model_paths):
        raise ValueError(
            f"Number of --models ({len(args.models)}) must match number of "
            f"--model-paths ({len(args.model_paths)})."
        )

    os.makedirs(args.results_dir, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    devices = [torch.device("cpu")]
    if cuda_available:
        devices.append(torch.device("cuda"))
    print(f"Available devices: {[str(d) for d in devices]}")
    print(f"Configuration: warmup={WARMUP_PASSES}, timed={TIMED_PASSES}, input={INPUT_SIZE}x{INPUT_SIZE}")
    print()

    results: list[dict] = []

    for model_name, checkpoint_path in zip(args.models, args.model_paths):
        print(f"--- {model_name} ---")
        # Load on CPU first to get parameter counts independent of device
        model = get_model(model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        total_params, trainable_params = count_parameters(model)
        print(f"  Total parameters:     {total_params:>12,}")
        print(f"  Trainable parameters: {trainable_params:>12,}")

        for device in devices:
            model = model.to(device)
            mean_ms, std_ms, throughput = benchmark_model(model, device)
            print(
                f"  {device.type.upper():3s}: "
                f"latency = {mean_ms:6.2f} ± {std_ms:5.2f} ms  |  "
                f"throughput = {throughput:7.1f} imgs/s"
            )

            results.append({
                "model": model_name,
                "device": device.type,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "latency_mean_ms": round(mean_ms, 3),
                "latency_std_ms": round(std_ms, 3),
                "throughput_imgs_per_sec": round(throughput, 2),
            })
        print()

    # --- Save CSV ---
    csv_path = os.path.join(args.results_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

    # --- Final summary table ---
    print("\n" + "=" * 90)
    print(" SUMMARY")
    print("=" * 90)
    print(f"  {'Model':<22} {'Device':<6} {'Latency (ms)':<18} {'Throughput (img/s)':<18} {'Params':<14}")
    print("-" * 90)
    for r in results:
        latency = f"{r['latency_mean_ms']:.2f} ± {r['latency_std_ms']:.2f}"
        print(
            f"  {r['model']:<22} {r['device']:<6} {latency:<18} "
            f"{r['throughput_imgs_per_sec']:<18.1f} {r['total_params']:<14,}"
        )


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference speed for EuroSAT models on CPU and GPU."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        choices=["baseline", "resnet_frozen", "resnet_finetuned"],
        help="One or more models to benchmark (space-separated).",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to saved .pt checkpoints, in the same order as --models.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Directory for saving the CSV (default: ../results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmarks(args)