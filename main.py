import sys
import os

def run_realtime_benchmark(args):
    """Execute the realtime_benchmark.py script with the provided arguments."""
    import realtime_benchmark
    # Prepare arguments for realtime_benchmark.main()
    sys.argv = [sys.argv[0]] + args  # Reset sys.argv to include only the script name and args
    realtime_benchmark.main()

def run_batch_benchmark(args):
    """Execute the batch_benchmark.py script with the provided arguments."""
    import batch_benchmark
    # Prepare arguments for batch_benchmark.main()
    sys.argv = [sys.argv[0]] + args  # Reset sys.argv to include only the script name and args
    batch_benchmark.main()

def print_usage():
    """Print usage instructions for the unified benchmark script."""
    print("""
Unified Ollama Benchmark Script
Usage: python main.py [realtime|batch] [options]

Modes:
  realtime   Run real-time monitoring interface (realtime_benchmark.py)
  batch      Run batch benchmarking suite (batch_benchmark.py)

Realtime Options:
  --mode {realtime,create-config}  Operation mode (default: realtime)
  --model-name STR                 Model name (e.g., qwen2.5:1.5b)
  --prompt STR                     Prompt for generation
  --host STR                       Ollama server host
  --max-tokens INT                 Maximum tokens to generate
  --temperature FLOAT              Temperature for generation
  --n-gpu-layers INT               Number of GPU layers
  --n-ctx INT                      Context length
  --top-k INT                      Top-k sampling parameter
  --top-p FLOAT                    Top-p sampling parameter
  --repeat-penalty FLOAT           Repeat penalty
  --num-thread INT                 Number of threads
  --config STR                     Path to JSON config file
  --force-size                     Skip terminal size checking
  --min-width INT                  Override minimum terminal width
  --min-height INT                 Override minimum terminal height

Batch Options:
  --config STR                     Path to configuration file (default: ollama_benchmark_config.json)

Examples:
  python main.py realtime --model-name qwen2.5:1.5b --prompt "Explain quantum computing"
  python main.py batch --config custom_benchmark_config.json
  python main.py realtime --mode create-config
""")

def main():
    # Check if at least one argument is provided
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Check for mode keywords in arguments
    args_lower = [arg.lower() for arg in sys.argv[1:]]
    mode = None
    remaining_args = sys.argv[1:]

    # Look for 'realtime' or 'batch' in arguments
    if "realtime" in args_lower:
        mode = "realtime"
        # Remove 'realtime' from arguments to avoid confusion with script's parser
        remaining_args = [arg for arg in sys.argv[1:] if arg.lower() != "realtime"]
    elif "batch" in args_lower:
        mode = "batch"
        # Remove 'batch' from arguments to avoid confusion with script's parser
        remaining_args = [arg for arg in sys.argv[1:] if arg.lower() != "batch"]
    else:
        print("Error: Please specify either 'realtime' or 'batch' mode.")
        print_usage()
        sys.exit(1)

    # Dispatch to the appropriate script
    if mode == "realtime":
        run_realtime_benchmark(remaining_args)
    elif mode == "batch":
        run_batch_benchmark(remaining_args)

if __name__ == "__main__":
    main()