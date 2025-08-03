# **Local LLM Performance & Benchmarking Suite**

A comprehensive toolkit to benchmark, monitor, and visualize the performance of local GGUF models. This suite provides the tools necessary to determine the viability and performance characteristics of running open-source language models on your local hardware.

## **📋 Table of Contents**

* [Features](https://www.google.com/search?q=%23-features)  
* [Project Structure](https://www.google.com/search?q=%23-project-structure)  
* [Setup & Installation](https://www.google.com/search?q=%23-setup--installation)  
* [Usage](https://www.google.com/search?q=%23-usage)  
  * [1\. Batch Benchmarking](https://www.google.com/search?q=%231-batch-benchmarking)  
  * [2\. Real-time Monitoring](https://www.google.com/search?q=%232-real-time-monitoring)  
  * [3\. Visualizing Results](https://www.google.com/search?q=%233-visualizing-results)  
* [Configuration](https://www.google.com/search?q=%23-configuration)  
* [Output](https://www.google.com/search?q=%23-output)  
* [Future Scope](https://www.google.com/search?q=%23-future-scope)  
* [Development Approach](https://www.google.com/search?q=./APPROACH.md)

## **✨ Features**

* **📊 Comprehensive Batch Benchmarking**: Systematically run tests on multiple models with varying parameters (context size, generation length, GPU layers) from a single configuration file.  
* **🚀 Real-time Performance Monitoring**: Launch a live dashboard in your terminal to monitor a model's performance metrics (TPS, TPM, resource usage) as it generates text.  
* **📈 Interactive Visualization Dashboard**: A user-friendly Streamlit web interface to upload and analyze your benchmark results, featuring interactive charts and filters.  
* **💻 Hardware Resource Tracking**: Monitors and records peak CPU, RAM, and VRAM usage for each benchmark run to understand hardware limitations.  
* **⚙️ Unified & Modular Design**: A single, clean command-line interface (main.py) controls the different modes, with logic cleanly separated into distinct modules.

## **📂 Project Structure**

.  
├── main.py                   \# Main entry point to run the benchmark suite  
├── batch\_benchmark.py        \# Core logic for running batch tests from a config file  
├── realtime\_benchmark.py     \# Core logic for the live terminal monitoring dashboard  
├── interface.py              \# Streamlit web application for visualizing results  
├── requirements.txt          \# Python dependencies for the project  
├── benchmark\_config\_grid.json \# Example configuration file for batch mode  
└── prompts/  
    └── base\_prompt.txt       \# A sample prompt used for batch benchmarking
Additionally, add a models folder in main directory to store models.
## **🛠️ Setup & Installation**

Follow these steps to get the project running on your local machine.

### **1\. Clone the Repository**

git clone \[https://github.com/your-username/your-repository-name.git\](https://github.com/your-username/your-repository-name.git)  
cd your-repository-name

### **2\. Create a Virtual Environment (Recommended)**

python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

### **3\. Install Dependencies**

Install all the necessary Python packages from the requirements.txt file.

pip install \-r requirements.txt

### **4\. Download GGUF Models**

This tool requires GGUF-formatted models to run. Download the models you wish to test (e.g., from Hugging Face) and place them in a known directory (e.g., a models/ folder).

## **🚀 Usage**

The benchmarking suite is controlled via main.py and offers two primary modes.

### **1\. Batch Benchmarking**

This mode is ideal for running a comprehensive set of tests overnight or for a long period. It reads its configuration from benchmark\_config\_grid.json.

**To run the batch benchmark:**

python main.py \--mode batch

* The script will automatically find benchmark\_config\_grid.json.  
* It will run through all model and parameter combinations defined in the file.  
* A benchmark\_results.csv file will be created or overwritten with the new results.

To use a different configuration file:

python main.py \--mode batch \--config my\_custom\_config.json

### **2\. Real-time Monitoring**

This mode is perfect for quick, interactive tests of a single model. It launches a live dashboard directly in your terminal.

To run the real-time monitor:  
You must provide the path to a model.  
python main.py \--mode realtime \--model-path "path/to/your/model.gguf"

Customizing the real-time session:  
You can specify the prompt, context size, and number of GPU layers.  
python main.py \--mode realtime \\  
  \--model-path "path/to/your/model.gguf" \\  
  \--prompt "Write a long, detailed story about a robot who discovers music." \\  
  \--n-gpu-layers 35 \\  
  \--n-ctx 4096

### **3\. Visualizing Results**

After running a batch benchmark, you can analyze the generated benchmark\_results.csv using the interactive Streamlit dashboard.

**To launch the dashboard:**

streamlit run interface.py

* This will open a new tab in your web browser.  
* Click the "Browse files" button to upload your benchmark\_results.csv file.  
* The dashboard will update instantly, allowing you to filter and explore your results.

## **⚙️ Configuration**

The batch benchmark mode is controlled by benchmark\_config\_grid.json. This file defines the models to test and the parameters to test them with.

**Example benchmark\_config\_grid.json:**

{  
  "models": \[  
    {  
      "name": "Llama-3.1-8B-Instruct-Q4\_K\_M",  
      "path": "models/Meta-Llama-3.1-8B-Instruct-Q4\_K\_M.gguf"  
    },  
    {  
      "name": "Qwen-2.5-7B-Instruct-Q4\_K\_M",  
      "path": "models/Qwen-2.5-7B-Instruct-Q4\_K\_M.gguf"  
    }  
  \],  
  "parameters": {  
    "context\_size": \[2048, 4096\],  
    "gen\_length": \[256, 512, 1024\],  
    "ngl": \[0, 99\]  
  }  
}

* **models**: A list of model objects, each with a display name and the path to the GGUF file.  
* **parameters**: A dictionary where each key is a parameter (context\_size, gen\_length, ngl) and the value is a list of settings to test. The script will generate a test for every possible combination.  
  * ngl: 0 will test CPU-only inference.  
  * ngl: 99 will offload all possible layers to the GPU.

## **📊 Output**

The primary output of the batch benchmark is the benchmark\_results.csv file, which contains the following columns:

| Column | Description |
| :---- | :---- |
| model\_name | The display name of the model. |
| quant | The quantization type (e.g., Q4\_K\_M), extracted from the filename. |
| context\_size | The context size (n\_ctx) used for the run. |
| gen\_length | The number of tokens generated (max\_tokens). |
| ngl | The number of GPU layers offloaded. |
| load\_time\_s | Time taken to load the model into memory (in seconds). |
| ttft\_s | Time to First Token (in seconds). |
| tps | Tokens per Second during generation. |
| tpm | Tokens per Minute (tps \* 60). |
| cpu\_peak\_percent | Peak CPU utilization during inference. |
| ram\_peak\_mb | Peak RAM usage (in Megabytes). |
| vram\_peak\_mb | Peak VRAM usage (in Megabytes). |
| error | Any error message, if the run failed. |

The tool also generates a log.txt for the real-time monitor and an environment\_snapshot.json to record the hardware used for the tests.

## **🔮 Future Scope**

This project provides a strong foundation for local LLM evaluation. Future enhancements could include:

* **Quality Benchmarking**: Integrate metrics like perplexity or run standardized evaluations (e.g., MMLU, HellaSwag) to measure model output quality in addition to performance.  
* **Historical Comparison**: Enhance the Streamlit dashboard to compare two different benchmark\_results.csv files, allowing for easy A/B testing of hardware or model changes.  
* **Expanded Model Support**: Add support for other model libraries beyond llama-cpp-python, such as Ollama or vLLM, to broaden the scope of testing.  
* **Automated Reporting**: Add a feature to automatically generate a PDF or HTML summary report from the benchmark results, including key charts and system information.  
* **Power Usage Monitoring**: Integrate power consumption tracking (e.g., via pyJoules) to analyze the energy efficiency (performance-per-watt) of different models and hardware setups.
