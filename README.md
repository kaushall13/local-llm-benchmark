# **Local LLM Performance & Benchmarking Suite**

A comprehensive toolkit to benchmark, monitor, and visualize the performance of local GGUF models. This suite provides the tools necessary to determine the viability and performance characteristics of running open-source language models on your local hardware.

## **📋 Table of Contents**

* [Features](#features)  
* [Project Structure](#project-structure)  
* [Setup & Installation](#setup--installation)  
* [Usage](#usage)  
  * [1. Batch Benchmarking](#1-batch-benchmarking)  
  * [2. Real-time Monitoring](#2-real-time-monitoring)  
  * [3. Visualizing Results](#3-visualizing-results)  
* [Configuration](#configuration)  
* [Output](#output)  
* [Future Scope](#future-scope)  
* [Development Approach](#development-approach)

## **Features**

* **📊 Comprehensive Batch Benchmarking**: Systematically run tests on multiple models with varying parameters (context size, generation length, GPU layers) from a single configuration file.  
* **🚀 Real-time Performance Monitoring**: Launch a live dashboard in your terminal to monitor a model's performance metrics (TPS, TPM, resource usage) as it generates text.  
* **📈 Interactive Visualization Dashboard**: A user-friendly Streamlit web interface to upload and analyze your benchmark results, featuring interactive charts and filters.  
* **💻 Hardware Resource Tracking**: Monitors and records peak CPU, RAM, and VRAM usage for each benchmark run to understand hardware limitations.  
* **⚙️ Unified & Modular Design**: A single, clean command-line interface (main.py) controls the different modes, with logic cleanly separated into distinct modules.

## **Project Structure**

```
.  
├── main.py                   # Main entry point to run the benchmark suite  
├── batch_benchmark.py        # Core logic for running batch tests from a config file  
├── realtime_benchmark.py     # Core logic for the live terminal monitoring dashboard  
├── interface.py              # Streamlit web application for visualizing results  
├── requirements.txt          # Python dependencies for the project  
├── benchmark_config_grid.json # Example configuration file for batch mode  
└── prompts/  
    └── base_prompt.txt       # A sample prompt used for batch benchmarking
```

Additionally, add a models folder in main directory to store models.

## **Setup & Installation**

Follow these steps to get the project running on your local machine.

### **1. Clone the Repository**

```bash
git clone https://github.com/kaushall13/local-llm-benchmark.git  
cd local-llm-benchmark
```

### **2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv  
source venv/bin/activate
# On Windows, use venv\Scripts\activate
```

### **3. Install Dependencies**

Install all the necessary Python packages from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### **4. Download GGUF Models**

This tool requires GGUF-formatted models to run. Download the models you wish to test (e.g., from Hugging Face) and place them in a known directory (e.g., a models/ folder).

## **Usage**

The benchmarking suite is controlled via main.py and offers two primary modes.

### **1. Batch Benchmarking**

This mode is ideal for running a comprehensive set of tests overnight or for a long period. It reads its configuration from benchmark_config_grid.json.

**To run the batch benchmark:**

```bash
python main.py --mode batch
```

* The script will automatically find benchmark_config_grid.json.  
* It will run through all model and parameter combinations defined in the file.  
* A benchmark_results.csv file will be created or overwritten with the new results.

To use a different configuration file:

```bash
python main.py --mode batch --config my_custom_config.json
```

### **2. Real-time Monitoring**

This mode is perfect for quick, interactive tests of a single model. It launches a live dashboard directly in your terminal.

> To run the real-time monitor:  
> You must provide the path to a model. 

```bash
python main.py --mode realtime --model-path "path/to/your/model.gguf"
```

> Customizing the real-time session:  
> You can specify the prompt, context size, and number of GPU layers. 

```bash
python main.py --mode realtime \
  --model-path "path/to/your/model.gguf" \
  --prompt "Write a long, detailed story about a robot who discovers music." \
  --n-gpu-layers 35 \
  --n-ctx 4096
```

### **3. Visualizing Results**

After running a batch benchmark, you can analyze the generated benchmark_results.csv using the interactive Streamlit dashboard.

**To launch the dashboard:**

```bash
streamlit run interface.py
```

* This will open a new tab in your web browser.  
* Click the "Browse files" button to upload your benchmark_results.csv file.  
* The dashboard will update instantly, allowing you to filter and explore your results.

## **Configuration**

The batch benchmark mode is controlled by benchmark_config_grid.json. This file defines the models to test and the parameters to test them with.

**Example benchmark_config_grid.json:**

```json
{  
  "models": [  
    {  
      "name": "Llama-3.1-8B-Instruct-Q4_K_M",  
      "path": "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"  
    },  
    {  
      "name": "Qwen-2.5-7B-Instruct-Q4_K_M",  
      "path": "models/Qwen-2.5-7B-Instruct-Q4_K_M.gguf"  
    }  
  ],  
  "parameters": {  
    "context_size": [2048, 4096],  
    "gen_length": [256, 512, 1024],  
    "ngl": [0, 99]  
  }  
}
```

* **models**: A list of model objects, each with a display name and the path to the GGUF file.  
* **parameters**: A dictionary where each key is a parameter (context_size, gen_length, ngl) and the value is a list of settings to test. The script will generate a test for every possible combination.  
  * ngl: 0 will test CPU-only inference.  
  * ngl: 99 will offload all possible layers to the GPU.

## **Output**

The primary output of the batch benchmark is the benchmark_results.csv file, which contains the following columns:

| Column | Description |
| :---- | :---- |
| model_name | The display name of the model. |
| quant | The quantization type (e.g., Q4_K_M), extracted from the filename. |
| context_size | The context size (n_ctx) used for the run. |
| gen_length | The number of tokens generated (max_tokens). |
| ngl | The number of GPU layers offloaded. |
| load_time_s | Time taken to load the model into memory (in seconds). |
| ttft_s | Time to First Token (in seconds). |
| tps | Tokens per Second during generation. |
| tpm | Tokens per Minute (tps * 60). |
| cpu_peak_percent | Peak CPU utilization during inference. |
| ram_peak_mb | Peak RAM usage (in Megabytes). |
| vram_peak_mb | Peak VRAM usage (in Megabytes). |
| error | Any error message, if the run failed. |

The tool also generates a log.txt for the real-time monitor and an environment_snapshot.json to record the hardware used for the tests.

## **Future Scope**

This project provides a strong foundation for local LLM evaluation. Future enhancements could include:

* **Quality Benchmarking**: Integrate metrics like perplexity or run standardized evaluations (e.g., MMLU, HellaSwag) to measure model output quality in addition to performance.  
* **Historical Comparison**: Enhance the Streamlit dashboard to compare two different benchmark_results.csv files, allowing for easy A/B testing of hardware or model changes.  
* **Expanded Model Support**: Add support for other model libraries beyond llama-cpp-python, such as Ollama or vLLM, to broaden the scope of testing.  
* **Automated Reporting**: Add a feature to automatically generate a PDF or HTML summary report from the benchmark results, including key charts and system information.  
* **Power Usage Monitoring**: Integrate power consumption tracking (e.g., via pyJoules) to analyze the energy efficiency (performance-per-watt) of different models and hardware setups.

## **Development Approach**

For detailed information about the development approach and methodology, see [APPROACH.md](./APPROACH.md).
