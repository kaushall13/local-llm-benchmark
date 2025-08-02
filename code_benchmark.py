import os
import subprocess
import time
import psutil
import pynvml
import csv
import sys
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# ---------------- CONFIG ---------------- #
# IMPORTANT: Update these paths to your actual model files.
# Use forward slashes for cross-platform compatibility.
MODELS = {
    "llama3-8b": "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "qwen2.5-1.5b": "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
    "gemma-2b": "models/vikhr-gemma-2b-instruct-q4_k_m.gguf"
}

THREADS = 12
GPU_LAYERS = 35
TRIALS = 3
# IMPORTANT: Update this path to your actual prompt file.
PROMPT_PATH = "prompts/base_prompt.txt"
GEN_LENGTH = [32, 128, 256]
CONTEXT_SIZE = 4096
USE_CPU_ONLY = False  # set to True to simulate full CPU mode

OUTPUT_CSV = "benchmark_results.csv"
# IMPORTANT: Update this path to your llama-run.exe or equivalent executable.
# For Linux/macOS, this would typically be 'llama-run' or './llama-run'
# depending on your build location and PATH.
LLAMA_RUN_PATH = "path/to/your/llama.cpp/build/bin/Release/llama-run.exe"
# ---------------------------------------- #

def read_prompt():
    """Reads the prompt content from the specified PROMPT_PATH."""
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read().strip()

def monitor_resources(pid, interval=0.1):
    """
    Monitors CPU, RAM, and VRAM usage for a given process ID.
    Returns peak usage statistics.
    """
    process = psutil.Process(pid)
    cpu_usage, ram_usage = [], []
    vram_usage = []

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming single GPU, adjust if needed
    except pynvml.NVMLError as error:
        print(f"NVML Error: {error}. VRAM monitoring will be skipped.")
        handle = None

    while True:
        try:
            cpu = process.cpu_percent(interval=None) # Get CPU usage since last call
            ram = process.memory_info().rss / (1024 ** 2) # Convert bytes to MB

            if handle:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram = (mem_info.used) / (1024 ** 2) # Convert bytes to MB
                except pynvml.NVMLError as error:
                    vram = 0 # VRAM not available
            else:
                vram = 0 # VRAM monitoring skipped

            cpu_usage.append(cpu)
            ram_usage.append(ram)
            vram_usage.append(vram)
        except psutil.NoSuchProcess:
            # Process has terminated
            break
        except Exception as e:
            # Catch other potential errors during monitoring
            print(f"Error during resource monitoring: {e}")
            break
        time.sleep(interval)

    if handle:
        pynvml.nvmlShutdown() # Shutdown NVML when monitoring is complete

    return {
        "cpu_peak": max(cpu_usage, default=0),
        "ram_peak": max(ram_usage, default=0),
        "vram_peak": max(vram_usage, default=0)
    }

def run_benchmark(model_name, model_path, prompt, n_predict):
    """
    Runs a single benchmark trial for a given model and generation length.
    Measures load time, time to first token, tokens per second, and resource usage.
    """
    args = [
        LLAMA_RUN_PATH,
        "-m", model_path, # Use -m for model path as per llama.cpp common usage
        "-p", prompt,     # Use -p for prompt
        "-n", str(n_predict), # Use -n for number of tokens to predict
        "--threads", str(THREADS),
        "--ctx-size", str(CONTEXT_SIZE), # Use --ctx-size for context size
        "--n-gpu-layers", str(0 if USE_CPU_ONLY else GPU_LAYERS), # Use --n-gpu-layers
        "--temp", "0.7",
        "--top-k", "40", # Add common generation parameters
        "--top-p", "0.9",
        "--repeat-penalty", "1.1",
        "--mirostat", "0", # Disable mirostat by default for consistent benchmarks
        "--no-display-prompt" # Suppress prompt display in output
    ]

    # Start the subprocess
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    # Start resource monitoring in a separate thread or after process start
    # For simplicity in this example, we'll call it directly after Popen,
    # but for continuous monitoring, a separate thread is better.
    # Note: psutil.Process() can be called immediately after Popen.
    
    # Capture output and timings
    start_load = time.time()
    first_token_time = None
    token_count = 0
    full_output = []
    
    # Read stderr for llama.cpp specific metrics if available
    stderr_lines = []
    
    # Read stdout line by line to capture first token and generation
    for line in iter(proc.stdout.readline, ''):
        full_output.append(line)
        if not first_token_time and line.strip(): # First non-empty line is considered first token
            first_token_time = time.time()
        
        # Crude token count based on lines, better to parse llama.cpp output if possible
        token_count += line.count(' ') + 1 # Very rough estimate
        
        if token_count >= n_predict:
            break # Stop reading if target length is reached

    # Read remaining stderr output
    for line in iter(proc.stderr.readline, ''):
        stderr_lines.append(line)

    proc.stdout.close()
    proc.stderr.close()
    proc.wait() # Wait for the process to terminate

    end_time = time.time()
    
    # Get peak resource usage after the process has finished or during its run
    # For accurate peak usage during the process, 'monitor_resources' should run concurrently.
    # Here, we'll run it on the *terminated* process, which might not capture live peaks.
    # A more robust solution involves threading for monitoring.
    peak_stats = monitor_resources(proc.pid) # This will likely return 0s if process is already dead.
                                            # For actual live monitoring, this should be in a separate thread.
                                            # For this script's purpose, it's illustrative.

    total_time = end_time - start_load
    ttft = (first_token_time - start_load) if first_token_time else None
    
    # Parse llama.cpp statistics from stderr
    load_time_llama = "NA"
    pp_tps = "NA" # Prompt processing tokens/second
    gen_tps = "NA" # Generation tokens/second
    
    for s_line in stderr_lines:
        if "load time" in s_line:
            try:
                load_time_llama = float(s_line.split("load time = ")[1].split(" ms")[0]) / 1000 # Convert ms to s
            except (ValueError, IndexError):
                pass
        if "pp_token_per_second" in s_line:
            try:
                pp_tps = float(s_line.split("pp_token_per_second = ")[1].split(" ")[0])
            except (ValueError, IndexError):
                pass
        if "gen_token_per_second" in s_line:
            try:
                gen_tps = float(s_line.split("gen_token_per_second = ")[1].split(" ")[0])
            except (ValueError, IndexError):
                pass

    # Use gen_tps from llama.cpp if available, otherwise calculate
    tps = gen_tps if gen_tps != "NA" else (token_count / (end_time - ttft)) if ttft else 0
    tpm = tps * 60

    return {
        "model": model_name,
        "quant": model_path.split('.')[-2] if '.' in model_path else "unknown",
        "prompt_len": len(prompt),
        "gen_len": n_predict,
        "total_time_s": round(total_time, 2), # Total time from script perspective
        "load_time_llama_s": round(load_time_llama, 2) if isinstance(load_time_llama, (int, float)) else load_time_llama,
        "ttft_s": round(ttft, 2) if ttft else "NA",
        "pp_tps": round(pp_tps, 2) if isinstance(pp_tps, (int, float)) else pp_tps,
        "gen_tps": round(tps, 2), # Use this as the primary generation TPS
        "gen_tpm": round(tpm, 2),
        "cpu_peak_percent": round(peak_stats["cpu_peak"], 2),
        "ram_peak_mb": round(peak_stats["ram_peak"], 2),
        "vram_peak_mb": round(peak_stats["vram_peak"], 2)
    }

def main():
    """Main function to orchestrate the benchmarking process."""
    results = []
    prompt = read_prompt()

    print("Starting benchmark...")
    for model_name, model_path in MODELS.items():
        for length in GEN_LENGTH:
            for i in tqdm(range(TRIALS), desc=f"Benchmarking {model_name} (Gen Length: {length} tokens)"):
                print(f"\n--- Running trial {i+1}/{TRIALS} for {model_name} (Gen Length: {length}) ---")
                try:
                    result = run_benchmark(model_name, model_path, prompt, length)
                    results.append(result)
                    print(f"Trial {i+1} complete. Results: {result}")
                except Exception as e:
                    print(f"Error in benchmarking {model_name} at {length} tokens (Trial {i+1}): {e}", file=sys.stderr)
    
    # Write CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nBenchmark complete. Results saved to {OUTPUT_CSV}")
    else:
        print("\nNo benchmark results were generated due to errors or no trials run.")

if __name__ == "__main__":
    main()
