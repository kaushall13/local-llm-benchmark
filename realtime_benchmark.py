import time
import threading
import psutil
import subprocess
import os
import textwrap
import logging
from datetime import datetime
from dataclasses import dataclass
from llama_cpp import Llama, llama_cpp
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.console import Console
from rich.align import Align

# --- SETUP THE LOGGER ---
log_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.FileHandler('log.txt')
log_handler.setFormatter(log_formatter)
performance_logger = logging.getLogger('performance_logger')
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(log_handler)

console = Console()

@dataclass
class ModelMetrics:
    gpu_layers_used: int = 0
    total_layers: int = 0
    kv_cache_size: float = 0.0
    kv_cache_tokens: int = 0
    context_length: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_speed: float = 0.0
    generation_speed: float = 0.0
    tpm: float = 0.0  # <-- 1. ADDED TPM to dataclass
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    gpu_util: float = 0.0
    gpu_kv_reserved: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0
    model_load_time: float = 0.0
    model_path: str = "N/A"

class LlamaCppMonitor:
    def __init__(self, model=None, update_interval=0.1, log_interval=5):
        self.model = model
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.metrics = ModelMetrics()
        self.is_monitoring = False
        self.monitor_thread = None
        self.logging_thread = None
        self._lock = threading.Lock()

        self.raw_generated_text = ""
        self.max_raw_chars = 4000
        self.output_width = 100
        self.output_rows = 15

        self.previous_time = 0
        self.previous_tokens = 0

    def _logging_loop(self):
        """A separate loop that writes logs periodically."""
        while self.is_monitoring:
            time.sleep(self.log_interval)
            self.log_performance_metrics()

    def log_performance_metrics(self):
        """Formats and writes the current metrics to the log file."""
        with self._lock:
            m = self.metrics
            # --- 4. ADDED TPM to log message ---
            log_message = (
                f"Tokens: {m.output_tokens}, "
                f"Speed: {m.generation_speed:.1f} t/s ({m.tpm:.0f} TPM), "
                f"CPU: {m.cpu_usage:.1f}%, "
                f"RAM: {m.ram_usage:.1f}%, "
                f"VRAM: {m.gpu_memory_used}MB ({m.gpu_util:.1f}%)"
            )
            performance_logger.info(log_message)

    def set_model(self, model):
        self.model = model
        self._update_static_metrics()

    def _update_static_metrics(self):
        if not self.model: return
        try:
            ctx = self.model._ctx.ctx
            model = self.model._model.model
            self.metrics.total_layers = llama_cpp.llama_n_layer(model)
            self.metrics.context_length = llama_cpp.llama_n_ctx(ctx)
            self.metrics.gpu_layers_used = getattr(self.model, 'n_gpu_layers', 0)
            self.metrics.model_path = getattr(self.model, 'model_path', 'Unknown')
        except Exception as e:
            console.print(f"[red]Error updating static metrics: {e}[/red]")

    def _get_gpu_stats(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=1
            )
            used, total, util = map(float, result.stdout.strip().split(', '))
            return int(used), int(total), util
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return 0, 0, 0.0

    def _get_kv_cache_info(self):
        if not self.model: return 0.0, 0.0, 0
        try:
            ctx = self.model._ctx.ctx
            model = self.model._model.model
            kv_tokens = llama_cpp.llama_kv_self_n_tokens(ctx)
            n_embd = llama_cpp.llama_n_embd(model)
            n_layer = llama_cpp.llama_n_layer(model)
            head_size = n_embd * 2
            reserved = 2 * n_layer * head_size * self.metrics.context_length
            actual = 2 * n_layer * head_size * kv_tokens
            return reserved, actual, kv_tokens
        except:
            return 0.0, 0.0, 0

    def _get_system_stats(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        gpu_used, gpu_total, gpu_util = self._get_gpu_stats()
        return cpu_percent, memory.percent, memory.used, memory.total, gpu_used, gpu_total, gpu_util

    def update_metrics(self, input_tokens=0, output_tokens=0, prompt_time=0.0):
        with self._lock:
            m = self.metrics
            m.input_tokens = input_tokens
            m.output_tokens = output_tokens
            if prompt_time > 0 and input_tokens > 0:
                m.prompt_speed = input_tokens / prompt_time

            (m.cpu_usage, m.ram_usage, m.ram_used, m.ram_total,
             m.gpu_memory_used, m.gpu_memory_total, m.gpu_util) = self._get_system_stats()

            m.gpu_kv_reserved, m.kv_cache_size, m.kv_cache_tokens = self._get_kv_cache_info()

            current_time = time.time()
            if self.previous_time > 0:
                time_diff = current_time - self.previous_time
                token_diff = output_tokens - self.previous_tokens
                if time_diff > 0:
                    tps = token_diff / time_diff
                    m.generation_speed = tps
                    m.tpm = tps * 60  # <-- 2. CALCULATED TPM
                else:
                    m.generation_speed = 0
                    m.tpm = 0
            self.previous_time = current_time
            self.previous_tokens = output_tokens

    def update_generated_text(self, token: str):
        with self._lock:
            self.raw_generated_text += token
            if len(self.raw_generated_text) > self.max_raw_chars:
                self.raw_generated_text = self.raw_generated_text[-self.max_raw_chars:]

    def _format_bytes(self, b):
        if b is None: return "0 B"
        b = float(b)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

    def display_metrics_rich(self):
        m = self.metrics
        layout = Layout()

        layout.split(
            Layout(ratio=1, name="main"),
            Layout(size=self.output_rows + 2, name="output"),
            Layout(size=1, name="footer")
        )
        layout["main"].split_row(Layout(name="stats"), Layout(name="resources"))

        # Left Panel: Model Info
        t1 = Table(title="💹 Model Info", title_style="", border_style="dim")
        t1.add_column("Key", style="bold")
        t1.add_column("Value")
        t1.add_row("Model", os.path.basename(m.model_path))
        t1.add_row("Total Layers", str(m.total_layers))
        t1.add_row("Context", str(m.context_length))
        t1.add_row("Model Load Time", f"{m.model_load_time:.2f} sec")
        t1.add_row("In/Out", f"{m.input_tokens} / {m.output_tokens}")
        t1.add_row("Tokens", str(m.input_tokens + m.output_tokens))
        layout["stats"].update(Panel(t1, style="dim"))

        # Right Panel: System & Cache
        t2 = Table(title="💻 System & Cache", title_style="", border_style="dim")
        t2.add_column("Metric", style="bold")
        t2.add_column("Usage")
        t2.add_row("CPU", f"{m.cpu_usage:.1f}%")
        t2.add_row("RAM", f"{m.ram_usage:.1f}% ({self._format_bytes(m.ram_used)}/{self._format_bytes(m.ram_total)})")
        t2.add_row("VRAM", f"{m.gpu_memory_used} / {m.gpu_memory_total} MB")
        t2.add_row("GPU UTIL", f"{m.gpu_util}%")
        t2.add_row("KV Reserved", f"{self._format_bytes(m.gpu_kv_reserved)}")
        t2.add_row("KV Used", f"{self._format_bytes(m.kv_cache_size)}")
        t2.add_row("KV Tokens", str(m.kv_cache_tokens))
        t2.add_row("Gen TPS", f"{m.generation_speed:.1f}")
        t2.add_row("TPM", f"{m.tpm:.0f}") # <-- 3. ADDED TPM to display
        layout["resources"].update(Panel(t2, style="dim"))

        wrapped_lines = textwrap.wrap(self.raw_generated_text, width=self.output_width)
        display_text = "\n".join(wrapped_lines[-self.output_rows:])

        output_panel = Panel(
            Text(display_text, justify="left"),
            title="🤖 Model Output",
            border_style="green"
        )
        layout["output"].update(output_panel)

        layout["footer"].update(Align.center(Text("Press Ctrl+C to stop", style="dim")))
        return layout

    def _monitor_loop(self):
        with Live(self.display_metrics_rich(), refresh_per_second=int(1/self.update_interval), screen=True, transient=True) as live:
            while self.is_monitoring:
                live.update(self.display_metrics_rich())
                time.sleep(self.update_interval)

    def start_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            self.logging_thread.start()

    def stop_monitoring(self):
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1)

class MonitoredLlama(Llama):
    def __init__(self, *args, **kwargs):
        self.monitor = LlamaCppMonitor()

        start_time = time.time()
        super().__init__(*args, **kwargs)
        end_time = time.time()

        self.monitor.metrics.model_load_time = end_time - start_time
        self.monitor.set_model(self)

    def __call__(self, *args, **kwargs):
        if not self.monitor.is_monitoring:
            self.monitor.start_monitoring()

        prompt = args[0] if args else kwargs.get('prompt', '')
        input_tokens = len(self.tokenize(prompt.encode() if isinstance(prompt, str) else prompt))

        kwargs['stream'] = True
        output_tokens = 0
        prompt_time = 0.0
        prompt_processed = False

        self.monitor.previous_time = 0

        generation_start_time = time.time()
        for token_obj in super().__call__(*args, **kwargs):
            if not prompt_processed:
                prompt_time = time.time() - generation_start_time
                prompt_processed = True
                self.monitor.previous_time = time.time()
                self.monitor.previous_tokens = 0

            token_text = token_obj['choices'][0]['text']
            output_tokens += 1

            self.monitor.update_generated_text(token_text)

            if output_tokens % 2 == 0:
                self.monitor.update_metrics(input_tokens, output_tokens, prompt_time)

        self.monitor.update_metrics(input_tokens, output_tokens, prompt_time)
        self.monitor.log_performance_metrics()
        return {"choices": [{"text": self.monitor.raw_generated_text}]}

    def stop_monitoring(self):
        self.monitor.stop_monitoring()


if __name__ == "__main__":
    MIN_WIDTH = 120
    MIN_HEIGHT = 35

    if console.width < MIN_WIDTH or console.height < MIN_HEIGHT:
        warning_message = (
            f"⚠️ [bold yellow]Terminal Size Warning[/bold yellow]\n\n"
            f"Your current terminal size is {console.width}x{console.height}. "
            f"The recommended minimum is {MIN_WIDTH}x{MIN_HEIGHT}.\n\n"
            f"Please expand the window for optimal display.\n"
        )
        console.print(Panel(warning_message, title="[yellow]Alert[/yellow]", border_style="yellow"))
        console.print("Continuing in 2 seconds...")
        time.sleep(2)

    # --- CONFIGURE YOUR MODEL HERE ---
    model = MonitoredLlama(
        model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=4096,
        offload_kqv=True,
        verbose=False
    )

    try:
        performance_logger.info("--- Starting new generation task ---")
        prompt = "Explain the theory of general relativity in three concise paragraphs."
        response = model(prompt, max_tokens=4096, temperature=0.7)
        performance_logger.info("--- Generation task finished ---")

        print("Generation complete. Shutting down in 10 seconds...")
        time.sleep(10)

    except KeyboardInterrupt:
        performance_logger.warning("--- Generation cancelled by user ---")
        print("\n\n[Generation cancelled by user.]")
    finally:
        model.stop_monitoring()