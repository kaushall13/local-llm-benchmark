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
import contextlib
import itertools
from operator import itemgetter
import requests
import ollama
import logging
from typing import *
from dataclasses import dataclass

# --- LOGGER SETUP ---
log_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.FileHandler('ollama_benchmark_log.txt')
log_handler.setFormatter(log_formatter)
performance_logger = logging.getLogger('ollama_benchmark_logger')
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(log_handler)

# ----------------- ANSI Color Codes for Terminal Output ----------------- #
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
OUTPUT_CSV_PREFIX = "ollama_benchmark_results"
ENV_SNAPSHOT = "environment_snapshot.json"
THREADS = psutil.cpu_count(logical=False) or 4  # Default to physical cores
TRIALS = 2  # Number of times to run each benchmark for averaging
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama server

# ---------------------------------------------------------------------- #

@dataclass
class ModelInfo:
    """Data class for model information"""
    name: str
    size: str = "Unknown"
    family: str = "Unknown"
    format: str = "Unknown"
    context_length: int = 2048
    modified_at: str = ""
    digest: str = ""

class OllamaModelManager:
    """Enhanced model management with smart model discovery and caching"""
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self._model_cache: Dict[str, ModelInfo] = {}
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300  # 5 minutes cache
        
    def is_server_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def _refresh_model_cache(self) -> bool:
        """Refresh the model cache from server"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._model_cache.clear()
                
                for model_data in data.get('models', []):
                    name = model_data.get('name', '')
                    if name:
                        self._model_cache[name] = ModelInfo(
                            name=name,
                            size=self._format_bytes(model_data.get('size', 0)),
                            digest=model_data.get('digest', ''),
                            modified_at=model_data.get('modified_at', '')
                        )
                
                self._cache_timestamp = time.time()
                return True
        except requests.exceptions.RequestException as e:
            print_message(f"Failed to refresh model cache: {e}", bcolors.FAIL)
        return False

    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of available models with caching"""
        if force_refresh or (time.time() - self._cache_timestamp) > self._cache_ttl:
            self._refresh_model_cache()
        
        return list(self._model_cache.keys())

    def find_model(self, model_query: str) -> Optional[str]:
        """
        Smart model finding with fuzzy matching and suggestions
        Handles cases like:
        - Exact match: "llama3.2:3b" 
        - Partial match: "llama3" -> finds "llama3.2:latest"
        - Version agnostic: "gemma" -> finds "gemma3:2b"
        """
        available_models = self.get_available_models()
        
        # Exact match first
        if model_query in available_models:
            return model_query
            
        # Add :latest if no tag specified
        if ':' not in model_query:
            latest_variant = f"{model_query}:latest"
            if latest_variant in available_models:
                return latest_variant
        
        # Fuzzy matching - find models containing the query
        matches = []
        query_lower = model_query.lower().replace(':', '').replace('-', '')
        
        for model in available_models:
            model_clean = model.lower().replace(':', '').replace('-', '')
            if query_lower in model_clean or model_clean.startswith(query_lower):
                matches.append(model)
        
        if matches:
            # Prefer models with :latest or smaller versions first
            matches.sort(key=lambda x: (
                ':latest' not in x,  # :latest models first
                len(x),  # shorter names first
                x  # alphabetical
            ))
            return matches[0]
        
        return None

    def get_model_suggestions(self, model_query: str, limit: int = 5) -> List[str]:
        """Get model suggestions based on query"""
        available_models = self.get_available_models()
        query_lower = model_query.lower()
        
        suggestions = []
        for model in available_models:
            if query_lower in model.lower():
                suggestions.append(model)
        
        return suggestions[:limit]

    def pull_model(self, model_name: str, show_progress: bool = True) -> bool:
        """Pull a model with enhanced progress display"""
        if not model_name:
            return False
            
        try:
            print_message(f"📥 Pulling model: {model_name}", bcolors.OKCYAN)
            
            response = requests.post(
                f"{self.host}/api/pull", 
                json={"name": model_name}, 
                stream=True, 
                timeout=600  # 10 minutes timeout for large models
            )
            
            if response.status_code != 200:
                print_message(f"❌ Failed to initiate pull: {response.status_code}", bcolors.FAIL)
                return False
            
            total_size = 0
            completed_size = 0
            last_update = 0
            
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    status = data.get('status', '')
                    
                    if show_progress and time.time() - last_update > 1.0:  # Update every second
                        if 'pulling' in status.lower():
                            total = data.get('total', 0)
                            completed = data.get('completed', 0)
                            if total > 0:
                                progress = (completed / total) * 100
                                print_message(f"Progress: {progress:.1f}% ({self._format_bytes(completed)}/{self._format_bytes(total)})", bcolors.OKBLUE)
                        else:
                            print_message(f"{status}", bcolors.OKBLUE)
                        last_update = time.time()
                    
                    if data.get('status') == 'success' or 'successfully' in status.lower():
                        print_message(f"✅ Model {model_name} pulled successfully!", bcolors.OKGREEN)
                        self._refresh_model_cache()  # Refresh cache after successful pull
                        return True
                        
                except json.JSONDecodeError:
                    continue
                    
            return False
            
        except requests.exceptions.RequestException as e:
            print_message(f"❌ Error pulling model {model_name}: {e}", bcolors.FAIL)
            return False

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model"""
        if model_name in self._model_cache:
            cached_info = self._model_cache[model_name]
        else:
            cached_info = ModelInfo(name=model_name)
        
        # Get additional details from show API
        try:
            response = requests.post(f"{self.host}/api/show", 
                json={"name": model_name}, timeout=10)
                
            if response.status_code == 200:
                data = response.json()
                details = data.get('details', {})
                
                cached_info.family = details.get('family', 'Unknown')
                cached_info.format = details.get('format', 'Unknown')
                
                # Parse context length from modelfile
                modelfile = data.get('modelfile', '')
                if 'num_ctx' in modelfile:
                    for line in modelfile.split('\n'):
                        if 'num_ctx' in line.lower():
                            try:
                                cached_info.context_length = int(line.split()[-1])
                                break
                            except (ValueError, IndexError):
                                pass
                
                # Update cache
                self._model_cache[model_name] = cached_info
                
        except requests.exceptions.RequestException:
            pass  # Use cached info if API call fails
            
        return cached_info

    def ensure_model_available(self, model_query: str) -> Optional[str]:
        """
        Comprehensive model availability check with smart resolution
        Returns the actual model name if available, None if failed
        """
        if not self.is_server_running():
            print_message(f"❌ Ollama server is not running at {self.host}", bcolors.FAIL)
            print_message("Please start Ollama with: ollama serve", bcolors.WARNING)
            return None
        
        # Step 1: Try to find existing model
        found_model = self.find_model(model_query)
        if found_model:
            print_message(f"✅ Model found: {found_model}", bcolors.OKGREEN)
            return found_model
        
        # Step 2: Try to pull the exact model name
        print_message(f"🔍 Model '{model_query}' not found locally. Attempting to pull...", bcolors.OKCYAN)
        
        if self.pull_model(model_query):
            return model_query
        
        # Step 3: Show suggestions
        suggestions = self.get_model_suggestions(model_query)
        if suggestions:
            print_message(f"💡 Did you mean one of these?", bcolors.OKCYAN)
            for i, suggestion in enumerate(suggestions, 1):
                print_message(f"  {i}. {suggestion}", bcolors.OKBLUE)
        else:
            # Show some popular models if no suggestions
            available_models = self.get_available_models()[:10]
            if available_models:
                print_message("Available models:", bcolors.OKCYAN)
                for model in available_models:
                    print_message(f"  - {model}", bcolors.OKBLUE)
        
        return None

    @staticmethod
    def _format_bytes(b):
        """Format bytes into human readable format."""
        if b is None or b == 0: 
            return "0 B"
        b = float(b)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

def print_message(text, color=bcolors.ENDC):
    """Prints a colored message to the console."""
    print(f"{color}{text}{bcolors.ENDC}")

def read_prompt(prompt_path):
    """Reads the prompt from the specified file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print_message(f"❌ Prompt file not found: {prompt_path}", bcolors.FAIL)
        raise

def check_ollama_running():
    """Checks if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def snapshot_env():
    """Saves a snapshot of the hardware environment to a JSON file."""
    print_message("🔬 Taking environment snapshot...", bcolors.OKCYAN)
    
    # Get GPU info
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        gpu_name = name if isinstance(name, str) else name.decode('utf-8')
        vram_gb = round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 3), 2)
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        gpu_name, vram_gb = "N/A", 0
        print_message("⚠️ NVML not available. GPU info will be skipped.", bcolors.WARNING)
    
    # Get Ollama version
    ollama_version = "Unknown"
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/version", timeout=5)
        if response.status_code == 200:
            ollama_version = response.json().get('version', 'Unknown')
    except requests.exceptions.RequestException:
        pass

    env = {
        "timestamp": datetime.now().isoformat(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "ollama_version": ollama_version,
        "ollama_host": OLLAMA_HOST
    }
    
    with open(ENV_SNAPSHOT, 'w') as f:
        json.dump(env, f, indent=4)
    print_message("✅ Environment snapshot saved.", bcolors.OKGREEN)

def monitor_resources(stop_event, result_dict):
    """Monitors the current process's resource usage in a thread."""
    # Monitor Ollama process instead of current process
    ollama_processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ollama' in proc.info['name'].lower():
                ollama_processes.append(psutil.Process(proc.info['pid']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Initialize GPU monitoring
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_gpu = True
    except pynvml.NVMLError:
        has_gpu = False

    cpu_usage, ram_usage, vram_usage = [], [], []
    
    while not stop_event.is_set():
        try:
            # Monitor Ollama processes
            total_cpu = 0
            total_ram = 0
            for proc in ollama_processes:
                try:
                    total_cpu += proc.cpu_percent(interval=None)
                    total_ram += proc.memory_info().rss / (1024 ** 2)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            cpu_usage.append(total_cpu)
            ram_usage.append(total_ram)
            
            if has_gpu:
                vram_usage.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2))
        except Exception as e:
            performance_logger.error(f"Resource monitoring error: {e}")
        
        time.sleep(0.2)
    
    result_dict['cpu_peak_percent'] = max(cpu_usage, default=0)
    result_dict['ram_peak_mb'] = max(ram_usage, default=0)
    result_dict['vram_peak_mb'] = max(vram_usage, default=0)
    
    if has_gpu:
        pynvml.nvmlShutdown()

def run_inference(config, prompt):
    """Runs inference using Ollama."""
    result = {**config, "error": None}
    resource_stats = {}
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_monitoring, resource_stats))
    monitor_thread.start()
    
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        
        # Generate options based on config - now includes all supported parameters
        options = {}
        if 'temperature' in config:
            options['temperature'] = config['temperature']
        if 'num_gpu' in config:
            options['num_gpu'] = config['num_gpu']
        if 'num_thread' in config:
            options['num_thread'] = config['num_thread']
        if 'num_ctx' in config:
            options['num_ctx'] = config['num_ctx']
        if 'num_predict' in config:  # Added support for num_predict
            options['num_predict'] = config['num_predict']
        if 'top_k' in config:
            options['top_k'] = config['top_k']
        if 'top_p' in config:
            options['top_p'] = config['top_p']
        if 'repeat_penalty' in config:
            options['repeat_penalty'] = config['repeat_penalty']
        
        start_time = time.time()
        
        # Use the generate API for more detailed timing information
        response = client.generate(
            model=config['model_name'],
            prompt=prompt,
            options=options,
            stream=False
        )
        
        total_time = time.time() - start_time
        
        # Extract timing information
        prompt_eval_count = response.get('prompt_eval_count', 0)
        prompt_eval_duration = response.get('prompt_eval_duration', 0) / 1e9  # Convert to seconds
        eval_count = response.get('eval_count', 0)
        eval_duration = response.get('eval_duration', 0) / 1e9  # Convert to seconds
        
        # Calculate metrics
        result['ttft_s'] = prompt_eval_duration  # Time to first token
        result['total_time_s'] = total_time
        result['prompt_tokens'] = prompt_eval_count
        result['completion_tokens'] = eval_count
        
        if eval_duration > 0 and eval_count > 0:
            result['tps'] = eval_count / eval_duration
        else:
            result['tps'] = 0
        
        result['tpm'] = result['tps'] * 60
        
        # Store response for analysis if needed
        result['response_length'] = len(response.get('response', ''))
        
        performance_logger.info(f"Successful inference for {config['model_name']}: TPS={result['tps']:.2f}, TPM={result['tpm']:.2f}")
        
    except requests.exceptions.RequestException as e:
        result['error'] = str(e)
        print_message(f"\n--- INFERENCE FAILED for {config['model_name']}: {e} ---", bcolors.FAIL)
        performance_logger.error(f"Inference failed: {e}")
    except Exception as e:
        result['error'] = str(e)
        print_message(f"\n--- INFERENCE FAILED for {config['model_name']}: {e} ---", bcolors.FAIL)
        performance_logger.error(f"Inference failed: {e}")
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
                **param_set
            }
            test_configs.append(config)
    
    return test_configs

def create_sample_config(config_path):
    """Creates a comprehensive sample configuration file."""
    sample_config = {
        "models": [
            {"name": "gemma3:270m"},
            {"name": "qwen2.5:1.5b"}
        ],
        "parameters": {
            "num_ctx": [2048],
            "num_predict": [128, 512],
            "num_gpu": [0, -1],  # 0 = CPU only, -1 = all GPU layers
            "temperature": [0.7],
            "num_thread": [THREADS],
            "top_k": [40],
            "top_p": [0.9],
            "repeat_penalty": [1.1]
        },
        "benchmark_settings": {
            "trials": TRIALS,
            "output_prefix": OUTPUT_CSV_PREFIX,
            "ollama_host": OLLAMA_HOST
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    return sample_config

def load_config(config_path):
    """Load and validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required sections
        if 'models' not in config:
            raise ValueError("Config must contain 'models' section")
        if 'parameters' not in config:
            raise ValueError("Config must contain 'parameters' section")
        
        # Apply benchmark settings if present
        if 'benchmark_settings' in config:
            settings = config['benchmark_settings']
            global TRIALS, OUTPUT_CSV_PREFIX, OLLAMA_HOST
            TRIALS = settings.get('trials', TRIALS)
            OUTPUT_CSV_PREFIX = settings.get('output_prefix', OUTPUT_CSV_PREFIX)
            OLLAMA_HOST = settings.get('ollama_host', OLLAMA_HOST)
        
        return config
        
    except json.JSONDecodeError as e:
        print_message(f"❌ Invalid JSON in config file: {e}", bcolors.FAIL)
        return None
    except FileNotFoundError:
        print_message(f"❌ Config file not found: {config_path}", bcolors.FAIL)
        return None
    except Exception as e:
        print_message(f"❌ Error loading config: {e}", bcolors.FAIL)
        return None

def print_summary(df):
    """Prints a formatted summary of the benchmark results to the console."""
    print_message("\n\n--- Ollama Benchmark Summary ---", bcolors.HEADER)
    
    # Group by model, then by test parameters
    for model_name, model_group in df.groupby('model_name'):
        print_message(f"\nModel: {bcolors.BOLD}{model_name}{bcolors.ENDC}", bcolors.OKBLUE)
        
        # Create grouping key based on available parameters
        group_cols = ['num_ctx', 'temperature']
        if 'num_predict' in df.columns:
            group_cols.append('num_predict')
        
        param_groups = model_group.groupby(group_cols)
        for params, group in param_groups:
            if len(group_cols) == 2:
                num_ctx, temperature = params
                print(f"  Test Scenario (context: {int(num_ctx)}, temp: {temperature}):")
            else:
                num_ctx, temperature, num_predict = params
                print(f"  Test Scenario (context: {int(num_ctx)}, temp: {temperature}, predict: {int(num_predict)}):")
            
            cpu_run = group[group['num_gpu'] == 0].iloc[0] if not group[group['num_gpu'] == 0].empty else None
            gpu_run = group[group['num_gpu'] != 0].iloc[0] if not group[group['num_gpu'] != 0].empty else None
            
            if cpu_run is not None:
                print(f"    - CPU: TTFT: {cpu_run['ttft_s']:.2f}s | TPS: {bcolors.OKGREEN}{cpu_run['tps']:.2f}{bcolors.ENDC} | RAM Peak: {cpu_run['ram_peak_mb']:.0f} MB")
            if gpu_run is not None:
                print(f"    - GPU: TTFT: {gpu_run['ttft_s']:.2f}s | TPS: {bcolors.OKGREEN}{gpu_run['tps']:.2f}{bcolors.ENDC} | VRAM Peak: {gpu_run['vram_peak_mb']:.0f} MB")
            
            if cpu_run is not None and gpu_run is not None and cpu_run['tps'] > 0:
                performance_increase = gpu_run['tps'] / cpu_run['tps']
                print(f"      {bcolors.OKCYAN}Performance Gain: GPU is {performance_increase:.2f}x faster{bcolors.ENDC}")

def setup_batch_benchmark_models(model_manager):
    """
    Setup function for batch benchmark script.
    Add this to the beginning of batch_benchmark.main()
    """
    # Ensure your specific model is available
    if not model_manager.ensure_model_available("qwen2.5:1.5b"):
        print_message("Cannot proceed without the required model.", bcolors.FAIL)
        return False
    
    return True

def main(config_path="ollama_benchmark_config.json"):
    """Main function to orchestrate the Ollama benchmarking suite."""
    model_manager = OllamaModelManager(host=OLLAMA_HOST)
    
    # Load or create configuration
    if not os.path.exists(config_path):
        print_message(f"📄 Creating sample config file: {config_path}", bcolors.OKCYAN)
        grid_config = create_sample_config(config_path)
        print_message("Please edit the config file with your desired models and parameters.", bcolors.WARNING)
        print_message("The config file supports the following parameters:", bcolors.OKCYAN)
        print_message("  - num_ctx: Context length", bcolors.OKBLUE)
        print_message("  - num_predict: Max tokens to generate", bcolors.OKBLUE)
        print_message("  - num_gpu: GPU layers (0=CPU, -1=all GPU)", bcolors.OKBLUE)
        print_message("  - temperature: Sampling temperature", bcolors.OKBLUE)
        print_message("  - num_thread: CPU threads to use", bcolors.OKBLUE)
        print_message("  - top_k, top_p, repeat_penalty: Additional sampling params", bcolors.OKBLUE)
        return
    else:
        grid_config = load_config(config_path)
        if grid_config is None:
            return
    
    if not setup_batch_benchmark_models(model_manager):
        return
    
    # Check if Ollama is running
    if not model_manager.is_server_running():
        print_message("❌ Ollama server is not running. Please start Ollama first.", bcolors.FAIL)
        print_message("   Run 'ollama serve' in a separate terminal or start Ollama Desktop.", bcolors.WARNING)
        return
    
    print_message("✅ Ollama server is running", bcolors.OKGREEN)
    
    # Create prompt file if it doesn't exist
    prompt_path = "prompts/base_prompt.txt"
    if not os.path.exists(prompt_path):
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, "w") as f:
            f.write("Write a short story about artificial intelligence.")
    
    # Ensure all models are available
    print_message("🔍 Checking model availability...", bcolors.OKCYAN)
    available_models = []
    for model in grid_config.get('models', []):
        model_name = model['name']
        actual_model = model_manager.ensure_model_available(model_name)
        if actual_model:
            model['name'] = actual_model  # Update to resolved name if fuzzy matched
            available_models.append(model)
        else:
            print_message(f"⚠️ Skipping unavailable model: {model_name}", bcolors.WARNING)
    
    if not available_models:
        print_message("❌ No models available for testing", bcolors.FAIL)
        return
    
    # Update config with available models
    grid_config['models'] = available_models
    configs = generate_test_configs(grid_config)
    
    snapshot_env()
    prompt = read_prompt(prompt_path)
    all_results = []
    
    total_scenarios = len(configs)
    
    print_message(f"🚀 Starting benchmark with {total_scenarios} test scenarios", bcolors.OKGREEN)
    
    try:
        with tqdm(total=total_scenarios, desc="Benchmarking Scenarios", file=sys.stdout) as pbar:
            for i, cfg in enumerate(configs):
                model_name = cfg['model_name']
                # Create a more descriptive progress description
                desc_parts = [f"{model_name}"]
                if 'num_ctx' in cfg:
                    desc_parts.append(f"ctx:{cfg['num_ctx']}")
                if 'num_predict' in cfg:
                    desc_parts.append(f"pred:{cfg['num_predict']}")
                if 'num_gpu' in cfg:
                    desc_parts.append(f"gpu:{cfg['num_gpu']}")
                
                pbar.set_description(f"Running {' '.join(desc_parts)}")
                
                # Run multiple trials for averaging
                trial_results = []
                for trial in range(TRIALS):
                    result = run_inference(cfg, prompt)
                    trial_results.append(result)
                
                # Average successful runs
                successful_runs = [r for r in trial_results if not r.get("error")]
                if successful_runs:
                    avg_result = pd.DataFrame(successful_runs).mean(numeric_only=True).to_dict()
                    # Copy non-numeric fields from first successful run
                    for key, value in successful_runs[0].items():
                        if key not in avg_result:
                            avg_result[key] = value
                    all_results.append(avg_result)
                elif trial_results:
                    # If all failed, keep the first failure
                    all_results.append(trial_results[0])
                
                pbar.update(1)
                
    except KeyboardInterrupt:
        print_message("\nBenchmark interrupted by user. Saving partial results...", bcolors.WARNING)
    
    if not all_results:
        print_message("\nNo benchmarks were successfully completed.", bcolors.FAIL)
        return
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability - now includes num_predict
    cols_order = ['model_name', 'num_ctx', 'num_predict', 'temperature', 'num_gpu', 'num_thread',
                  'ttft_s', 'total_time_s', 'tps', 'tpm', 'prompt_tokens', 'completion_tokens',
                  'cpu_peak_percent', 'ram_peak_mb', 'vram_peak_mb', 'response_length', 'error',
                  'top_k', 'top_p', 'repeat_penalty']  # Include additional parameters if present
    df = df.reindex(columns=[col for col in cols_order if col in df.columns])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_CSV_PREFIX}_{timestamp}.csv"
    df.to_csv(output_filename, index=False, float_format='%.3f')
    
    print_message(f"\n✅ Benchmarking complete. Results saved to {output_filename}", bcolors.OKGREEN)
    print_summary(df)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama Benchmark Suite")
    parser.add_argument("--config", "-c", default="ollama_benchmark_config.json",
                       help="Path to configuration file (default: ollama_benchmark_config.json)")
    
    args = parser.parse_args()
    
    main(args.config)