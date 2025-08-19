# local-llm-benchmark  

**A modular benchmarking toolkit to evaluate local LLMs with Ollama.**  
Measure throughput, latency, and system-level metrics across models, quantizations, and prompt lengths—helping you determine deployment viability for your hardware.  

---

## 🚀 Introduction  

Running large language models locally can be fast, private, and cost-effective—but performance varies dramatically depending on the **model**, **quantization**, and **hardware setup**.  

**local-llm-benchmark** provides:  
- A **simple Ollama-based benchmarking flow** for quick evaluations.  
- Detailed metrics on **throughput, latency, and resource usage**.  
- Flexible configuration so you can tailor benchmarks to your environment.  

> ✅ By default, this project uses **Ollama** for a frictionless setup.  
> ⚙️ Advanced users can optionally use the **llama.cpp backend** (see [`setup.md`](setup.md) for detailed installation).  

---

## ✨ Key Features  

- 🔹 **Ollama-first design** – start benchmarking in minutes.  
- 🔹 **Latency & throughput metrics** – tokens/sec, tokens/min, time to first token.  
- 🔹 **Resource monitoring** – track CPU, RAM, and GPU usage.  
- 🔹 **Config-driven** – define models, quantizations, prompt sizes in JSON/YAML.  
- 🔹 **Extensible** – add new models, backends, or metrics with minimal changes.  

---

## 📦 Getting Started  

### 1. Install Dependencies  
Make sure [Ollama](https://ollama.ai/) is installed and running on your system. Then clone and install requirements:  

```bash
git clone https://github.com/kaushall13/local-llm-benchmark
cd local-llm-benchmark
pip install -r requirements.txt
```

### 2. Configure Your Benchmark  
Edit `config.json` (or your own config file) to specify models, quantizations, and prompt lengths.  

Example:  
```json
{
  "models": [
    { "name": "llama3.1:8b", "quant": "Q4_K_M", "prompt_length": 512, "gen_length": 512 }
  ],
  "runs": 3
}
```

### 3. Run the Benchmark  
```bash
python benchmark.py --config config.json
```

Results will include:  
- Average **tokens per minute (TPM)**  
- **Latency breakdown** (time-to-first-token, tokens/sec)  
- Peak **CPU/RAM/GPU usage**  

---

## 📊 Example Output  

```
--------------------------------------------------------------------------------
Model                     | Quant    | Avg TPM    | Load (ms)  | Peak VRAM (MB)  | Peak RAM (MB)
--------------------------------------------------------------------------------
llama3.1:8b               | Q4_K_M   | 15,930     | 78         | 4560            | 8221
qwen2.5:1.5b              | Q4_K_M   | 8,450      | 65         | 2100            | 4120
gemma:2b                  | Q4_K_M   | 10,120     | 72         | 2350            | 4300
```

---

## ⚙️ Advanced Mode: llama.cpp  

For users who want more control (e.g., CUDA layers, context sizes, fine-grained performance tuning), we also provide a **llama.cpp backend**.  

- Requires additional setup (CUDA toolkit, VS Build Tools on Windows, etc.).  
- See [`setup.md`](setup.md) for detailed installation and build instructions.  
- Once installed, you can benchmark llama.cpp models using the same config format.  

---

## 📌 Use Cases  

- **Model selection** – compare LLaMA, Qwen, Gemma, and more.  
- **Quantization trade-offs** – balance memory savings vs. speed.  
- **Hardware profiling** – see how different CPUs/GPUs perform.  
- **Deployment readiness** – identify configs that meet latency & throughput goals.  

---

## 🤝 Contributing  

We welcome contributions!  
- Add new models or quantization formats.  
- Improve monitoring & visualization.  
- Share benchmarking results across devices.  

Fork, branch, and open a PR 🚀  

---

## 📄 License  
This project is licensed under the **MIT License**.  
