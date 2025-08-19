import argparse
import sys
import os

# Import the refactored functions from your other scripts
import batch_benchmark
import realtime_benchmark

def main():
    """
    Main function to parse arguments and launch the selected benchmark mode.
    """
    parser = argparse.ArgumentParser(
        description="Unified benchmarking tool for Llama.cpp models.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'realtime'],
        required=True,
        help="""Select the benchmark mode:
'batch':    Run a comprehensive benchmark from a config file and save to CSV.
'realtime': Run a live, interactive benchmark with a single model and prompt."""
    )

    # Arguments for 'batch' mode
    parser.add_argument(
        '--config',
        type=str,
        default="benchmark_config_grid.json",
        help="Path to the JSON config file for batch mode. (Default: benchmark_config_grid.json)"
    )

    # Arguments for 'realtime' mode
    parser.add_argument('--model-path', type=str, help="[realtime only] Path to the GGUF model file.")
    parser.add_argument('--prompt', type=str, default="Explain the theory of general relativity in one paragraph.", help="[realtime only] The prompt to use.")
    parser.add_argument('--n-gpu-layers', type=int, default=-1, help="[realtime only] Number of GPU layers to offload.")
    parser.add_argument('--n-ctx', type=int, default=4096, help="[realtime only] Context size for the model.")

    args = parser.parse_args()

    if args.mode == 'batch':
        print("🚀 Starting Batch Benchmark Mode...")
        # Call the main function from the batch script
        batch_benchmark.main(config_path=args.config)

    elif args.mode == 'realtime':
        if not args.model_path:
            print("❌ Error: --model-path is required for realtime mode.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.model_path):
            print(f"❌ Error: Model file not found at '{args.model_path}'", file=sys.stderr)
            sys.exit(1)

        print("🚀 Starting Realtime Benchmark Mode...")
        # Call the refactored function from the realtime script
        realtime_benchmark.run_realtime(
            model_path=args.model_path,
            prompt=args.prompt,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx
        )

if __name__ == "__main__":
    main()