import os
import time
import psutil
import pynvml
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import re
import threading
import sys
from llama_cpp import Llama
import contextlib
import itertools
from operator import itemgetter

# ---------------- ANSI Color Codes for Terminal Output ---------------- #
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# --------------------------- CONFIGURATION ---------------------------- #
OUTPUT_CSV_PREFIX = "benchmark_results"
ENV_SNAPSHOT = "environment_snapshot.json"
THREADS = psutil.cpu_count(logical=False) or 4 # Default to physical cores
TRIALS = 2 # Number of times to run each benchmark for averaging
# ---------------------------------------------------------------------- #

def print_message(text, color=bcolors.ENDC):
    """Prints a colored message to the console."""
    print(f"{color}{text}{bcolors.ENDC}")

def read_prompt(prompt_path):
    """Reads the prompt from the specified file."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def snapshot_env():
    """Saves a snapshot of the hardware environment to a JSON file."""
    print_message("🔬 Taking environment snapshot...", bcolors.OKCYAN)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        gpu_name = name.decode('utf-8') if isinstance(name, bytes) else name
        vram_gb = round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3), 2)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        gpu_name, vram_gb = "N/A", 0
        print_message("⚠️ NVML not available. GPU info will be skipped.", bcolors.WARNING)

    env = {
        "timestamp": datetime.now().isoformat(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu_name,
        "vram_gb": vram_gb
    }
    with open(ENV_SNAPSHOT, 'w') as f:
        json.dump(env, f, indent=4)
    print_message("✅ Environment snapshot saved.", bcolors.OKGREEN)

def monitor_resources(stop_event, result_dict):
    """Monitors the current process's resource usage in a thread."""
    pid = os.getpid()
    process = psutil.Process(pid)
    process.cpu_percent(interval=None) # Prime the pump for cpu_percent

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_gpu = True
    except pynvml.NVMLError:
        has_gpu = False

    cpu_usage, ram_usage, vram_usage = [], [], []

    while not stop_event.is_set():
        try:
            cpu_usage.append(process.cpu_percent(interval=None) / THREADS)
            ram_usage.append(process.memory_info().rss / (1024 ** 2))
            if has_gpu:
                vram_usage.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(0.2)

    result_dict['cpu_peak_percent'] = max(cpu_usage, default=0)
    result_dict['ram_peak_mb'] = max(ram_usage, default=0)
    result_dict['vram_peak_mb'] = max(vram_usage, default=0)

    if has_gpu:
        pynvml.nvmlShutdown()

def get_quant_from_filename(filename):
    """Extracts quantization type from a GGUF filename."""
    base_name = os.path.basename(filename)
    match = re.search(r'(Q[2-8]_[A-Z0-9_]+)', base_name, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "unknown"

def run_inference(llm_instance, config, prompt):
    """Runs inference on an already loaded Llama instance."""
    result = {**config, "error": None}
    resource_stats = {}

    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring, resource_stats))
    monitor_thread.start()

    try:
        with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
            start_gen_time = time.time()
            output = llm_instance(
                prompt,
                max_tokens=config['gen_length'],
                temperature=0.7,
            )
            manual_gen_time_s = time.time() - start_gen_time

        usage = output['usage']
        prompt_eval_time_s = usage.get('prompt_eval_time', 0) / 1000
        result['ttft_s'] = prompt_eval_time_s # TTFT for a loaded model is just prompt eval time

        completion_time_s = usage.get('completion_time', 0) / 1000
        completion_tokens = usage.get('completion_tokens', 0)

        tps = 0
        if completion_time_s > 0 and completion_tokens > 0:
            tps = completion_tokens / completion_time_s
        elif manual_gen_time_s > 0 and completion_tokens > 0:
            tps = completion_tokens / manual_gen_time_s

        result['tps'] = tps
        result['tpm'] = tps * 60

    except Exception as e:
        result['error'] = str(e)
        print_message(f"\n--- INFERENCE FAILED for {config['model_name']}: {e} ---", bcolors.FAIL)

    finally:
        stop_monitoring.set()
        monitor_thread.join()
        result.update(resource_stats)

    return result

def generate_test_configs(grid_config):
    """Generates a list of individual test configurations from a grid."""
    test_configs = []
    models = grid_config.get('models', [])
    params = grid_config.get('parameters', {})

    param_keys = params.keys()
    param_values = params.values()
    param_combinations = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

    for model in models:
        for param_set in param_combinations:
            config = {
                "model_name": model["name"],
                "model_path": os.path.abspath(model["path"]),
                **param_set
            }
            test_configs.append(config)

    return test_configs

def print_summary(df):
    """Prints a formatted summary of the benchmark results to the console."""
    print_message("\n\n--- Benchmark Summary ---", bcolors.HEADER)

    # Group by model, then by test parameters
    for model_name, model_group in df.groupby('model_name'):
        print_message(f"\nModel: {bcolors.BOLD}{model_name}{bcolors.ENDC}", bcolors.OKBLUE)

        param_groups = model_group.groupby(['context_size', 'gen_length'])
        for params, group in param_groups:
            context_size, gen_length = params
            print(f"  Test Scenario (context: {int(context_size)}, generation: {int(gen_length)}):")

            cpu_run = group[group['ngl'] == 0].iloc[0] if not group[group['ngl'] == 0].empty else None
            gpu_run = group[group['ngl'] != 0].iloc[0] if not group[group['ngl'] != 0].empty else None

            if cpu_run is not None:
                print(f"    - CPU: TTFT: {cpu_run['ttft_s']:.2f}s | TPS: {bcolors.OKGREEN}{cpu_run['tps']:.2f}{bcolors.ENDC} | RAM Peak: {cpu_run['ram_peak_mb']:.0f} MB")

            if gpu_run is not None:
                print(f"    - GPU: TTFT: {gpu_run['ttft_s']:.2f}s | TPS: {bcolors.OKGREEN}{gpu_run['tps']:.2f}{bcolors.ENDC} | VRAM Peak: {gpu_run['vram_peak_mb']:.0f} MB")

            if cpu_run is not None and gpu_run is not None and cpu_run['tps'] > 0:
                performance_increase = gpu_run['tps'] / cpu_run['tps']
                print(f"      {bcolors.OKCYAN}Performance Gain: GPU is {performance_increase:.2f}x faster{bcolors.ENDC}")

def main(config_path="benchmark_config_grid.json"):
    """Main function to orchestrate the benchmarking suite."""
    if not os.path.exists(config_path):
        print_message(f"FATAL: `{config_path}` not found.", bcolors.FAIL)
        return

    prompt_path = "prompts/base_prompt.txt"
    if not os.path.exists(prompt_path):
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f: f.write("Once upon a time,")

    with open(config_path, 'r') as f:
        grid_config = json.load(f)

    configs = generate_test_configs(grid_config)

    snapshot_env()
    prompt = read_prompt(prompt_path)
    all_results = []

    configs.sort(key=itemgetter('model_path', 'context_size', 'ngl'))
    grouped_for_loading = itertools.groupby(configs, key=itemgetter('model_path', 'context_size', 'ngl'))

    total_scenarios = len(configs)

    try:
        with tqdm(total=total_scenarios, desc="Benchmarking Scenarios", file=sys.stdout) as pbar:
            for (model_path, context_size, ngl), group_iterator in grouped_for_loading:
                group_configs = list(group_iterator)
                model_name = group_configs[0]['model_name']
                pbar.set_description(f"Loading {model_name[:20]} (ctx: {context_size}, ngl: {ngl})")

                load_time = 0
                llm = None
                try:
                    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
                        start_load_time = time.time()
                        llm = Llama(model_path=model_path, n_ctx=context_size, n_gpu_layers=ngl, n_threads=THREADS, verbose=False)
                        load_time = time.time() - start_load_time
                except Exception as e:
                    print_message(f"\nFATAL: Failed to load model {model_name}. Error: {e}", bcolors.FAIL)
                    for cfg in group_configs:
                        cfg['error'] = str(e)
                        all_results.append(cfg)
                        pbar.update(1)
                    continue

                for cfg in group_configs:
                    pbar.set_description(f"Running {cfg['model_name'][:20]} (gen: {cfg['gen_length']})")

                    trial_results = [run_inference(llm, cfg, prompt) for _ in range(TRIALS)]

                    successful_runs = [r for r in trial_results if not r.get("error")]
                    if successful_runs:
                        avg_result = pd.DataFrame(successful_runs).mean(numeric_only=True).to_dict()
                        for key, value in successful_runs[0].items():
                            if key not in avg_result: avg_result[key] = value
                        avg_result['load_time_s'] = load_time
                        avg_result['ttft_s'] += load_time
                        all_results.append(avg_result)
                    elif trial_results:
                        all_results.append(trial_results[0])

                    pbar.update(1)

                del llm

    except KeyboardInterrupt:
        print_message("\nBenchmark interrupted by user. Saving partial results...", bcolors.WARNING)

    if not all_results:
        print_message("\nNo benchmarks were successfully completed.", bcolors.FAIL)
        return

    df = pd.DataFrame(all_results)
    df['quant'] = df['model_path'].apply(get_quant_from_filename)

    cols_order = ['model_name', 'quant', 'context_size', 'gen_length', 'ngl',
                  'load_time_s', 'ttft_s', 'tps', 'tpm', 'cpu_peak_percent',
                  'ram_peak_mb', 'vram_peak_mb', 'error']
    df = df.reindex(columns=[col for col in cols_order if col in df.columns])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_CSV_PREFIX}.csv"
    df.to_csv(output_filename, index=False, float_format='%.2f')
    print_message(f"\n✅ Benchmarking complete. Results saved to {output_filename}", bcolors.OKGREEN)

    print_summary(df)

if __name__ == "__main__":
    main()
