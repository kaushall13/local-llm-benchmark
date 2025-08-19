### Goal

Run GGUF models locally on Windows 11 with NVIDIA GPU acceleration via **llama-cpp-python** (cuBLAS). This guide uses prebuilt CUDA wheels (fast path) and also documents a from-source path.

---

### 1) Hardware & OS checklist

- Windows 11 x64.
- NVIDIA GPU with recent drivers (Turing/RTX 20‑series or newer recommended).
- Sufficient VRAM for your chosen quant (e.g., Q4\_K\_M for 7B ≈ \~5 GB file size; leave headroom).
- Admin access to install drivers/tooling.

---

### 2) Install NVIDIA CUDA (Toolkit + Driver)

**Why:** The prebuilt CUDA wheels for llama‑cpp‑python expect CUDA 12.x runtime. Install a CUDA 12.x Toolkit that matches the wheel you will choose (12.1–12.5 recommended).

**Visit:** NVIDIA CUDA Downloads → [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

**Quick install (PowerShell as Admin):**

```powershell
winget install -e --id Nvidia.CUDA
```

**Verify after install:**

```powershell
nvidia-smi   # shows driver status
nvcc --version  # shows toolkit version (e.g., release 12.5)
```

> Note: `nvidia-smi` shows the **driver-supported** CUDA level; `nvcc --version` shows the **Toolkit** actually installed. Match your wheel to the Toolkit (12.1 ⇢ cu121, 12.2 ⇢ cu122, 12.3 ⇢ cu123, 12.4 ⇢ cu124, 12.5 ⇢ cu125).

---

### 3) Install Python 3.10–3.12

llama‑cpp‑python CUDA wheels support Python **3.10–3.12**.

**Option A (winget):**

```powershell
winget install -e --id Python.Python.3.12
```

**Option B (python.org):** [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

**Verify:**

```powershell
python --version
pip --version
```

---

### 4) Create a clean virtual environment

```powershell
# In a working folder, e.g., C:\llama
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel setuptools
```

---

### 5) Install llama-cpp-python (CUDA, prebuilt wheel) — *Fast path*

Pick the extra index that matches your installed CUDA **Toolkit** (from step 2):

```powershell
# Choose ONE of these based on nvcc version
pip install --upgrade llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
# pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
# pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
# pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

**Confirm installation:**

```powershell
python -c "import llama_cpp; print('llama-cpp-python OK')"
```

> If a CPU wheel is installed by mistake, reinstall with the correct `--extra-index-url` and ensure your Python version is 3.10–3.12.

---

### 6) Get a GGUF model

You can use **Hugging Face** models in GGUF format.

**Option A: Pull automatically in code (recommended)**

```powershell
pip install -U huggingface_hub
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",  # example repo
    filename="*Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=True
)

print(llm("Q: 2+2? A:", max_tokens=8))
```

**Option B: Download explicitly via CLI**

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF `
  --include "*Q4_K_M.gguf" --local-dir .\models\mistral7b
```

Then reference the file path with `model_path` in code:

```python
from llama_cpp import Llama
llm = Llama(
    model_path=r"C:\\llama\\models\\mistral7b\\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=True,
)
print(llm("Q: 2+2? A:", max_tokens=8))
```

---

### 7) Local LLM Performance & Benchmarking Suite

After verifying GPU inference works with llama-cpp-python, you can use the **Local LLM Performance & Benchmarking Suite** to systematically benchmark and visualize model performance.

**Repo:** [https://github.com/kaushall13/local-llm-benchmark](https://github.com/kaushall13/local-llm-benchmark)

#### 📂 Updated Project Structure

```
local-llm-benchmark/
├── llama-cpp-python/
│   ├── batch_benchmark.py
│   ├── benchmark_config_grid.json
│   ├── interface.py
│   ├── main.py
│   ├── realtime_benchmark.py
│   └── requirements.txt
├── prompts/
│   └── ...
├── sample_dashboard_images/
├── README.md
├── approach.md
├── batch_benchmark.py
├── config.json
├── interface.py
├── main.py
├── my_benchmark.json
├── realtime_benchmark.py
└── requirements.txt
```

`llama-cpp-python/` contains the benchmarking scripts tailored for llama-cpp-python. Root-level files include configs, prompts, and docs.

#### Setup & Installation

```bash
git clone https://github.com/kaushall13/local-llm-benchmark.git
cd local-llm-benchmark/llama-cpp-python
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Download GGUF models and place them under `models/`.

#### Usage

**Batch Benchmarking:**

```bash
python batch_benchmark.py --config benchmark_config_grid.json
```

**Real-time Monitoring:**

```bash
python realtime_benchmark.py --model-path "models/mymodel.gguf"
```

**Visualization:**

```bash
streamlit run interface.py
```

#### Configuration

Config files include:

- `benchmark_config_grid.json`: grid-based configs.
- `config.json` or `my_benchmark.json`: custom setups.

#### Output

Generates `benchmark_results.csv`, `log.txt`, and `environment_snapshot.json`.

---

### 8) Performance & tuning cheatsheet

- **n\_gpu\_layers**: -1 = offload all possible layers.
- **Quant**: Q4\_K\_M is a good starting point.
- **n\_ctx**: Increase for longer prompts.
- **n\_threads**: Match CPU cores.
- **Batching**: Tune `n_batch` and `n_ubatch`.
- **MMAP**: Enable to save RAM.

---

### 9) Common problems & fixes

- Missing DLL: Ensure CUDA toolkit installed.
- CPU wheel installed: Reinstall with correct cu12x wheel.
- Build fails: Install VS Build Tools + CMake.
- Server slow: Ensure GPU layers offloaded.
- Hugging Face 403: Accept license + login.

---

### 10) Build from source (optional)

Install VS Build Tools + CMake.

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install --upgrade --force-reinstall --no-cache-dir --verbose llama-cpp-python
```

---

### 11) Housekeeping

```powershell
pip install -U --force-reinstall --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

---

### 12) Quick checklist

1. Install CUDA 12.x.
2. Install Python 3.10–3.12.
3. Create venv & install matching cu12x wheel.
4. Download GGUF model.
5. Run benchmark suite (`batch_benchmark.py` or `realtime_benchmark.py`).
6. Analyze results with Streamlit (`interface.py`).

---

### Appendix: Picking a model & quant

- Start with 7B instruct models.
- Choose `Q4_K_M` for balance.
- Higher quant levels (Q5, Q6) need more VRAM.
- Only one `.gguf` file is needed per model.

