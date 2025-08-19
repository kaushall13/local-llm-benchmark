# Local LLM Benchmarking Suite

**A modular toolkit for benchmarking local Large Language Models (LLMs) using Ollama.**  
Evaluate performance metrics like throughput, latency, and resource usage across models, configurations, and hardware setups. Ideal for developers, researchers, and AI enthusiasts optimizing local AI deployments.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9d4c4105-b239-448f-b525-73b9cf5503c8" alt="Live Monitoring Example" width="48%"/>
  <img src="https://github.com/user-attachments/assets/96370c1e-a1f7-4ce6-8766-03c9bd35de8c" alt="Benchmark Table" width="48%"/>
</p>

---

## 🚀 Why Use This Toolkit?

Running LLMs locally offers privacy, cost savings, and low-latency inference—but performance depends heavily on your hardware, model choice, and settings. This toolkit helps you:
- **Measure real-world viability**: Test if a model meets your speed and resource requirements.
- **Compare configurations**: Evaluate trade-offs in context size, temperature, GPU layers, and more.
- **Monitor in real-time**: Visualize generation progress with live metrics and resource graphs.
- **Batch benchmark**: Run automated suites across multiple models and params, exporting results to CSV.

Built with Ollama for simplicity—no complex setups required. Focus on insights, not infrastructure.

---

## ✨ Key Features

- **Real-Time Mode**: Interactive dashboard for monitoring generation, with tokens/sec, TPM (tokens per minute), CPU/GPU/RAM usage, and live text output.
- **Batch Mode**: Automated benchmarking across model grids, averaging multiple trials for reliable results.
- **Detailed Metrics**: Time-to-first-token (TTFT), tokens per second (TPS), peak resource usage, and more.
- **Config-Driven**: JSON-based configs for easy customization of models, prompts, sampling params, and hardware settings.
- **Smart Model Management**: Automatic model pulling, fuzzy matching, and suggestions if a model isn't found locally.
- **Logging & Export**: Detailed logs and CSV outputs for analysis and sharing.
- **Hardware Snapshot**: Captures your system's CPU, RAM, GPU, and Ollama version for reproducible benchmarks.

---

## 📦 Installation

1. **Install Ollama**: Download and set up Ollama from [ollama.com](https://ollama.com). Start the server with `ollama serve`.

2. **Clone the Repository**:
   ```
   git clone https://github.com/kaushall13/local-llm-benchmark
   cd local-llm-benchmark
   ```

3. **Install Python Dependencies**:
   Ensure Python 3.8+ is installed, then:
   ```
   pip install -r requirements.txt
   ```
   (Requirements include `ollama`, `psutil`, `pynvml`, `rich`, `pandas`, `tqdm`, and more for monitoring and analysis.)

---

## 🔧 Usage

The toolkit uses a unified entry point (`main.py`) to run either **real-time** or **batch** modes.

### General Command
```
python main.py [realtime|batch] [options]
```

#### Real-Time Mode
Monitor a single generation in real-time with a rich terminal UI.

- **Basic Example**:
  ```
  python main.py realtime --model-name qwen2.5:1.5b --prompt "Explain quantum computing in simple terms"
  ```

- **With Custom Config**:
  First, create a default config:
  ```
  python main.py realtime --mode create-config
  ```
  Edit `config.json`, then run:
  ```
  python main.py realtime --config config.json
  ```

- **Options**:
  - `--model-name STR`: Model (e.g., `qwen2.5:1.5b`). Fuzzy matching supported.
  - `--prompt STR`: Input prompt for generation.
  - `--host STR`: Ollama server URL (default: `http://localhost:11434`).
  - `--max-tokens INT`: Max tokens to generate (default: 2000).
  - `--temperature FLOAT`: Sampling temperature (default: 0.7).
  - `--n-gpu-layers INT`: GPU layers (-1 for all, default: -1).
  - `--n-ctx INT`: Context length (default: 2048).
  - `--top-k INT`: Top-k sampling (default: 40).
  - `--top-p FLOAT`: Top-p sampling (default: 0.9).
  - `--repeat-penalty FLOAT`: Repeat penalty (default: 1.1).
  - `--num-thread INT`: CPU threads (default: 0, auto-detect).
  - `--config STR`: Path to JSON config file.
  - `--force-size`: Skip terminal size check (use cautiously).
  - `--min-width INT` / `--min-height INT`: Override min terminal dimensions.

  The UI requires a terminal of at least 80x30 (adjustable). Press Ctrl+C to stop.

#### Batch Mode
Run automated benchmarks across multiple configs, averaging trials.

- **Basic Example** (creates sample config if missing):
  ```
  python main.py batch --config ollama_benchmark_config.json
  ```

- **Options**:
  - `--config STR`: Path to config file (default: `ollama_benchmark_config.json`).

If the config doesn't exist, a sample is created with models like `gemma3:270m` and `qwen2.5:1.5b`. Edit it to specify:
```json
{
  "models": [
    {"name": "qwen2.5:1.5b"},
    {"name": "llama3.1:8b"}
  ],
  "parameters": {
    "num_ctx": [2048, 4096],
    "num_predict": [128, 512],
    "num_gpu": [0, -1],
    "temperature": [0.7],
    "num_thread": [4],
    "top_k": [40],
    "top_p": [0.9],
    "repeat_penalty": [1.1]
  },
  "benchmark_settings": {
    "trials": 3,
    "output_prefix": "ollama_benchmark_results",
    "ollama_host": "http://localhost:11434"
  }
}
```
Results are saved as CSV (e.g., `ollama_benchmark_results_YYYYMMDD_HHMMSS.csv`) with a console summary.

---

## 📊 Example Outputs

### Real-Time Mode
Live dashboard showing model info, resource usage, TPS/TPM, and generated text.

### Batch Mode Summary
```
--- Ollama Benchmark Summary ---

Model: qwen2.5:1.5b
  Test Scenario (context: 2048, temp: 0.7, predict: 128):
    - CPU: TTFT: 0.45s | TPS: 45.20 | RAM Peak: 2100 MB
    - GPU: TTFT: 0.12s | TPS: 120.50 | VRAM Peak: 1500 MB
      Performance Gain: GPU is 2.67x faster
```

Full CSV includes TTFT, TPS, TPM, tokens, resource peaks, and errors.

---

## 📌 Use Cases

- **Model Selection**: Compare Llama, Qwen, Gemma, etc., for your hardware.
- **Optimization**: Tune GPU layers, context size, and sampling for best performance.
- **Hardware Testing**: Profile CPU vs. GPU inference on different setups.
- **Deployment Prep**: Ensure configs meet latency/throughput SLAs for apps.

---

## 🤝 Contributing

Contributions welcome! Help add features like multi-GPU support or visualizations.
- Fork and create a feature branch.
- Submit a PR with clear descriptions and tests.
- Share your benchmark results across hardware.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
