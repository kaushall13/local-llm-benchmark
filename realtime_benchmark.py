import time
import threading
import psutil
import subprocess
import os
import textwrap
import logging
import requests
import json
import argparse
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.console import Console
from rich.align import Align

import ollama  # Import ollama for generation

# --- LOGGER SETUP ---
log_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = logging.FileHandler('ollama_log.txt')
log_handler.setFormatter(log_formatter)
performance_logger = logging.getLogger('ollama_performance_logger')
performance_logger.setLevel(logging.INFO)
performance_logger.addHandler(log_handler)

class TerminalChecker:
    """Terminal dimension checker and handler"""
    
    MIN_WIDTH = 120
    MIN_HEIGHT = 30
    
    @classmethod
    def get_terminal_size(cls) -> Tuple[int, int]:
        """Get current terminal dimensions"""
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24  # Default fallback
    
    @classmethod
    def check_terminal_size(cls) -> bool:
        """Check if terminal size is adequate"""
        width, height = cls.get_terminal_size()
        return width >= cls.MIN_WIDTH and height >= cls.MIN_HEIGHT
    
    @classmethod
    def wait_for_adequate_size(cls) -> None:
        """Wait for user to resize terminal to adequate dimensions"""
        if cls.check_terminal_size():
            return
        
        console.print("[yellow]⚠️  Terminal Size Check[/yellow]")
        console.print()
        
        while not cls.check_terminal_size():
            width, height = cls.get_terminal_size()
            console.clear()
            
            # Create a visual warning
            warning_panel = Panel(
                f"""[bold red]❌ Terminal Too Small[/bold red]

[yellow]Current size:[/yellow] {width} × {height}
[yellow]Minimum required:[/yellow] {cls.MIN_WIDTH} × {cls.MIN_HEIGHT}

[cyan]Please resize your terminal window to be larger.[/cyan]

[dim]The monitoring interface requires adequate space to display:
• Model information and metrics tables
• Real-time system resource usage  
• Generated text output window
• Performance graphs and statistics[/dim]

[bold green]Resize your terminal and this will automatically continue...[/bold green]""",
                title="🖥️  Terminal Size Warning",
                border_style="red",
                padding=(1, 2)
            )
            
            console.print(Align.center(warning_panel))
            console.print()
            console.print(Align.center(f"[dim]Checking every 2 seconds... Current: {width}×{height} | Need: {cls.MIN_WIDTH}×{cls.MIN_HEIGHT}[/dim]"))
            
            time.sleep(2)
        
        # Terminal is now adequate size
        console.clear()
        success_panel = Panel(
            "[bold green]✅ Terminal size is now adequate![/bold green]\n\n[cyan]Starting Ollama monitor...[/cyan]",
            title="🎉 Ready to Start",
            border_style="green"
        )
        console.print(Align.center(success_panel))
        time.sleep(1.5)
console = Console()

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

@dataclass
class ModelMetrics:
    """Cleaned up metrics class with only used variables"""
    model_name: str = "N/A"
    model_size: str = "N/A"
    model_family: str = "N/A"
    model_format: str = "N/A"
    context_length: int = 0
    output_tokens: int = 0
    tps_realtime: float = 0.0  # Real-time tokens per second
    tpm: float = 0.0
    gpu_memory_used: int = 0
    gpu_memory_total: int = 0
    gpu_util: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    ram_used: float = 0.0
    ram_total: float = 0.0
    total_duration: float = 0.0
    load_duration: float = 0.0
    generation_status: str = "Idle"

@dataclass
class Config:
    """Configuration class for model parameters"""
    model_name: str = "qwen2.5:1.5b"
    prompt: str = "Write a detailed explanation of machine learning and its real-world applications."
    host: str = "http://localhost:11434"
    max_tokens: int = 2000
    temperature: float = 0.7
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_thread: int = 0
    update_interval: float = 0.5
    log_interval: int = 10

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load config from {json_path}: {e}[/yellow]")
            console.print("[yellow]Using default configuration[/yellow]")
            return cls()

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
        }
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        console.print(f"[green]Configuration saved to {json_path}[/green]")

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
        except Exception as e:
            console.print(f"[red]Failed to refresh model cache: {e}[/red]")
        return False

    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Get list of available models with caching"""
        if force_refresh or (time.time() - self._cache_timestamp) > self._cache_ttl:
            self._refresh_model_cache()
        
        return list(self._model_cache.keys())

    def find_model(self, model_query: str) -> Optional[str]:
        """
        Smart model finding with fuzzy matching and suggestions
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
            console.print(f"[yellow]📥 Pulling model: {model_name}[/yellow]")
            
            response = requests.post(
                f"{self.host}/api/pull", 
                json={"name": model_name}, 
                stream=True, 
                timeout=600  # 10 minutes timeout for large models
            )
            
            if response.status_code != 200:
                console.print(f"[red]❌ Failed to initiate pull: {response.status_code}[/red]")
                return False
            
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
                                console.print(f"[dim]Progress: {progress:.1f}% ({self._format_bytes(completed)}/{self._format_bytes(total)})[/dim]")
                        else:
                            console.print(f"[dim]{status}[/dim]")
                        last_update = time.time()
                    
                    if data.get('status') == 'success' or 'successfully' in status.lower():
                        console.print(f"[green]✅ Model {model_name} pulled successfully![/green]")
                        self._refresh_model_cache()  # Refresh cache after successful pull
                        return True
                        
                except json.JSONDecodeError:
                    continue
                    
            return False
            
        except Exception as e:
            console.print(f"[red]❌ Error pulling model {model_name}: {e}[/red]")
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
                
        except Exception:
            pass  # Use cached info if API call fails
            
        return cached_info

    def ensure_model_available(self, model_query: str) -> Optional[str]:
        """
        Comprehensive model availability check with smart resolution
        Returns the actual model name if available, None if failed
        """
        if not self.is_server_running():
            console.print(f"[red]❌ Ollama server is not running at {self.host}[/red]")
            console.print("[yellow]Please start Ollama with: ollama serve[/yellow]")
            return None
        
        # Step 1: Try to find existing model
        found_model = self.find_model(model_query)
        if found_model:
            console.print(f"[green]✅ Model found: {found_model}[/green]")
            return found_model
        
        # Step 2: Try to pull the exact model name
        console.print(f"[yellow]🔍 Model '{model_query}' not found locally. Attempting to pull...[/yellow]")
        
        if self.pull_model(model_query):
            return model_query
        
        # Step 3: Show suggestions
        suggestions = self.get_model_suggestions(model_query)
        if suggestions:
            console.print(f"[yellow]💡 Did you mean one of these?[/yellow]")
            for i, suggestion in enumerate(suggestions, 1):
                console.print(f"  {i}. {suggestion}")
        else:
            # Show some popular models if no suggestions
            available_models = self.get_available_models()[:10]
            if available_models:
                console.print("[yellow]Available models:[/yellow]")
                for model in available_models:
                    console.print(f"  - {model}")
        
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

class OllamaMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.metrics = ModelMetrics()
        self.is_monitoring = False
        self.monitor_thread = None
        self.logging_thread = None
        self.size_monitor_thread = None
        self._lock = threading.RLock()
        self._terminal_size_ok = True

        self.raw_generated_text = ""
        self.max_raw_chars = 2000
        self.output_width = 100
        self.output_rows = 15

        self.generation_start_time = 0
        self.first_token_time = 0
        self.last_update_time = 0
        self.tokens_generated = 0
        
        # Performance optimization: Cache system stats
        self._last_system_update = 0
        self._system_update_interval = 2.0
        self._cached_system_stats = None

    def _monitor_terminal_size(self):
        """Monitor terminal size during execution"""
        while self.is_monitoring:
            if not TerminalChecker.check_terminal_size():
                self._terminal_size_ok = False
            else:
                self._terminal_size_ok = True
            time.sleep(2)  # Check every 2 seconds

    def _logging_loop(self):
        """A separate loop that writes logs periodically."""
        while self.is_monitoring:
            time.sleep(self.config.log_interval)
            if self.is_monitoring:
                self.log_performance_metrics()

    def log_performance_metrics(self):
        """Formats and writes the current metrics to the log file."""
        try:
            with self._lock:
                m = self.metrics
                log_message = (
                    f"Model: {m.model_name}, "
                    f"Tokens: {m.output_tokens}, "
                    f"Speed: {m.tps_realtime:.1f} t/s ({m.tpm:.0f} TPM), "
                    f"CPU: {m.cpu_usage:.1f}%, "
                    f"RAM: {m.ram_usage:.1f}%, "
                    f"VRAM: {m.gpu_memory_used}MB ({m.gpu_util:.1f}%), "
                    f"Status: {m.generation_status}"
                )
                performance_logger.info(log_message)
        except Exception:
            pass

    def set_model(self, model_name, model_manager: OllamaModelManager):
        """Set model and load its information"""
        self.metrics.model_name = model_name
        
        model_info = model_manager.get_model_info(model_name)
        if model_info:
            self.metrics.model_family = model_info.family
            self.metrics.model_format = model_info.format
            self.metrics.model_size = model_info.size
            self.metrics.context_length = model_info.context_length

    def _get_gpu_stats(self):
        """Get GPU statistics using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=1
            )
            used, total, util = map(float, result.stdout.strip().split(', '))
            return int(used), int(total), util
        except:
            return 0, 0, 0.0

    def _get_system_stats(self):
        """Get comprehensive system statistics with caching"""
        current_time = time.time()
        
        if (self._cached_system_stats and 
            current_time - self._last_system_update < self._system_update_interval):
            return self._cached_system_stats
        
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            gpu_used, gpu_total, gpu_util = self._get_gpu_stats()
            
            stats = (cpu_percent, memory.percent, memory.used, memory.total, 
                    gpu_used, gpu_total, gpu_util)
            
            self._cached_system_stats = stats
            self._last_system_update = current_time
            
            return stats
        except:
            return (0.0, 0.0, 0, 0, 0, 0, 0.0)

    def update_metrics(self, response_data=None):
        """Update metrics with response data from Ollama"""
        if not self.is_monitoring:
            return
            
        try:
            with self._lock:
                m = self.metrics
                
                # Update system stats
                (m.cpu_usage, m.ram_usage, m.ram_used, m.ram_total,
                 m.gpu_memory_used, m.gpu_memory_total, m.gpu_util) = self._get_system_stats()
                
                # Update from response data if provided
                if response_data:
                    # Convert nanoseconds to seconds
                    m.total_duration = response_data.get('total_duration', 0) / 1e9
                    m.load_duration = response_data.get('load_duration', 0) / 1e9
                    
                    # Token counts
                    eval_count = response_data.get('eval_count', 0)
                    
                    if eval_count > 0:
                        m.output_tokens = eval_count
        except:
            pass

    def update_generated_text(self, text: str, num_tokens: int):
        """Update the generated text display and calculate real-time speed"""
        current_time = time.time()
        
        try:
            self.raw_generated_text += text
            if len(self.raw_generated_text) > self.max_raw_chars:
                self.raw_generated_text = self.raw_generated_text[-self.max_raw_chars:]
            
            # Update real-time generation speed
            if self.first_token_time == 0:
                self.first_token_time = current_time
                self.metrics.generation_status = "Generating"
            
            self.tokens_generated += num_tokens
            
            # Calculate real-time speed
            if (current_time - self.last_update_time) > 1.0:
                time_elapsed = current_time - self.first_token_time
                if time_elapsed > 0:
                    self.metrics.tps_realtime = self.tokens_generated / time_elapsed
                    self.metrics.tpm = self.metrics.tps_realtime * 60
                    self.metrics.output_tokens = self.tokens_generated
                self.last_update_time = current_time
        except:
            pass

    def _format_bytes(self, b):
        """Format bytes into human readable format."""
        if b is None or b == 0: 
            return "0 B"
        b = float(b)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} PB"

    def display_metrics_rich(self):
        """Create the Rich layout for displaying metrics"""
        try:
            with self._lock:
                # Check if terminal size is adequate
                if not self._terminal_size_ok:
                    width, height = TerminalChecker.get_terminal_size()
                    return Panel(
                        f"[bold red]❌ Terminal too small![/bold red]\n\n"
                        f"Current: {width}×{height}\n"
                        f"Required: {TerminalChecker.MIN_WIDTH}×{TerminalChecker.MIN_HEIGHT}\n\n"
                        f"[yellow]Please resize your terminal window[/yellow]",
                        title="⚠️  Resize Terminal",
                        border_style="red"
                    )
                
                m = self.metrics
                layout = Layout()

                layout.split(
                    Layout(ratio=1, name="main"),
                    Layout(size=self.output_rows + 2, name="output"),
                    Layout(size=1, name="footer")
                )
                layout["main"].split_row(Layout(name="stats"), Layout(name="resources"))

                # Model Info Table
                t1 = Table(title="🤖 Ollama Model Info", title_style="bold cyan", border_style="dim")
                t1.add_column("Key", style="bold")
                t1.add_column("Value")
                t1.add_row("Model", str(m.model_name))
                t1.add_row("Family", str(m.model_family))
                t1.add_row("Size", str(m.model_size))
                t1.add_row("Context Length", str(m.context_length))
                t1.add_row("Output Tokens", str(m.output_tokens))
                t1.add_row("Status", f"[bold green]{m.generation_status}[/bold green]")
                layout["stats"].update(Panel(t1, style="dim"))

                # System Resources Table
                t2 = Table(title="💻 System & Performance", title_style="bold cyan", border_style="dim")
                t2.add_column("Metric", style="bold")
                t2.add_column("Value")
                t2.add_row("CPU Usage", f"{m.cpu_usage:.1f}%")
                t2.add_row("RAM Usage", f"{m.ram_usage:.1f}% ({self._format_bytes(m.ram_used)}/{self._format_bytes(m.ram_total)})")
                t2.add_row("GPU Memory", f"{m.gpu_memory_used} / {m.gpu_memory_total} MB")
                t2.add_row("GPU Utilization", f"{m.gpu_util:.1f}%")
                t2.add_row("Real-time TPS", f"[bold green]{m.tps_realtime:.1f}[/bold green] t/s")
                t2.add_row("TPM", f"[bold green]{m.tpm:.0f}[/bold green]")
                t2.add_row("Load Duration", f"{m.load_duration:.2f}s")
                layout["resources"].update(Panel(t2, style="dim"))

                # Generated Text Output
                wrapped_lines = textwrap.wrap(self.raw_generated_text, width=self.output_width)
                display_text = "\n".join(wrapped_lines[-self.output_rows:])

                output_panel = Panel(
                    Text(display_text, justify="left"),
                    title=f"🤖 Generated Output ({len(self.raw_generated_text)} chars)",
                    border_style="green"
                )
                layout["output"].update(output_panel)

                # Footer with terminal size info
                width, height = TerminalChecker.get_terminal_size()
                footer_text = f"Press Ctrl+C to stop | Terminal: {width}×{height}"
                layout["footer"].update(Align.center(Text(footer_text, style="dim")))
                return layout
        except:
            return Layout()

    def _monitor_loop(self):
        """Main monitoring loop with Rich display"""
        try:
            refresh_rate = max(1, int(1/self.config.update_interval))
            with Live(self.display_metrics_rich(), refresh_per_second=refresh_rate, 
                      screen=True, transient=True) as live:
                while self.is_monitoring:
                    try:
                        live.update(self.display_metrics_rich())
                        time.sleep(self.config.update_interval)
                    except Exception:
                        time.sleep(self.config.update_interval)
                        if not self.is_monitoring:
                            break
        except Exception as e:
            console.print(f"[red]Monitor loop error: {e}[/red]")

    def start_monitoring(self):
        """Start the monitoring threads."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logging_thread = threading.Thread(target=self._logging_loop, daemon=True)
            self.logging_thread.start()
            self.size_monitor_thread = threading.Thread(target=self._monitor_terminal_size, daemon=True)
            self.size_monitor_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring threads."""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            if self.size_monitor_thread:
                self.size_monitor_thread.join(timeout=1)

class MonitoredOllamaAPI:
    """Enhanced wrapper class with smart model management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = OllamaModelManager(config.host)
        self.monitor = OllamaMonitor(config)

    def generate(self, model_name: str):
        """Generate text with real-time monitoring"""
        # Set up model with enhanced info
        self.monitor.set_model(model_name, self.model_manager)
        
        if not self.monitor.is_monitoring:
            self.monitor.start_monitoring()
            time.sleep(0.1)

        self.monitor.generation_start_time = time.time()
        self.monitor.first_token_time = 0
        self.monitor.tokens_generated = 0
        self.monitor.raw_generated_text = ""
        self.monitor.metrics.generation_status = "Starting"

        try:
            # Build options from config
            options = {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "num_ctx": self.config.n_ctx,
                "num_thread": self.config.num_thread,
                "repeat_penalty": self.config.repeat_penalty,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            }
            
            # Add GPU layers if specified
            if self.config.n_gpu_layers > 0:
                options["num_gpu"] = self.config.n_gpu_layers

            # Use ollama library for streaming
            stream = ollama.generate(
                model=model_name,
                prompt=self.config.prompt,
                stream=True,
                options=options
            )

            full_response = ""
            final_response_data = None
            token_count = 0
            batch_text = ""
            batch_tokens = 0

            # Optimized streaming processing
            for chunk in stream:
                if 'response' in chunk and chunk['response']:
                    token = chunk['response']
                    full_response += token
                    token_count += 1
                    batch_text += token
                    batch_tokens += 1
                    
                    # Batch monitor updates
                    if batch_tokens >= 10 or chunk.get('done', False):
                        self.monitor.update_generated_text(batch_text, batch_tokens)
                        batch_text = ""
                        batch_tokens = 0
                
                # Update metrics periodically
                if token_count % 50 == 0 or chunk.get('done', False):
                    self.monitor.update_metrics(chunk)
                
                if chunk.get('done', False):
                    final_response_data = chunk
                    self.monitor.metrics.generation_status = "Complete"
                    break

            # Final update
            if final_response_data:
                self.monitor.update_metrics(final_response_data)

            return {
                "response": full_response,
                "metrics": final_response_data,
                "token_count": token_count
            }

        except Exception as e:
            self.monitor.metrics.generation_status = "Error"
            return {"response": "", "error": str(e)}

    def stop_monitoring(self):
        """Stop the monitoring display."""
        self.monitor.stop_monitoring()

def create_default_config():
    """Create default configuration file"""
    config = Config()
    config.to_json("config.json")
    console.print("[green]Default configuration file 'config.json' created![/green]")

def run_realtime(config: Config):
    """
    Enhanced real-time Ollama monitor with smart model management
    """
    # Check terminal size before starting
    console.print("[cyan]🔍 Checking terminal dimensions...[/cyan]")
    TerminalChecker.wait_for_adequate_size()
    
    model_manager = OllamaModelManager(config.host)
    
    # Smart model resolution
    actual_model = model_manager.ensure_model_available(config.model_name)
    if not actual_model:
        console.print(f"[red]❌ Could not resolve model: {config.model_name}[/red]")
        return

    # Create monitored Ollama instance
    console.print(f"[cyan]🚀 Initializing monitored Ollama with model: {actual_model}[/cyan]")
    
    # Update config with actual model name
    config.model_name = actual_model
    model = MonitoredOllamaAPI(config)

    try:
        performance_logger.info(f"--- Starting generation with model {actual_model} ---")
        
        # Run the generation
        response = model.generate(actual_model)
        
        if response.get("error"):
            console.print(f"[red]❌ Generation failed: {response['error']}[/red]")
        else:
            token_count = response.get("token_count", 0)
            response_length = len(response.get("response", ""))
            console.print(f"[green]✅ Generation completed successfully![/green]")
            console.print(f"[green]   Generated {token_count} tokens ({response_length} characters)[/green]")
        
        performance_logger.info("--- Generation task finished ---")
        console.print("\n[green]Generation complete. Showing results for 5 seconds...[/green]")
        time.sleep(5)

    except KeyboardInterrupt:
        performance_logger.warning("--- Generation cancelled by user ---")
        console.print("\n\n[yellow]Generation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {e}[/red]")
    finally:
        model.stop_monitoring()
        console.print("[cyan]👋 Monitoring stopped. Goodbye![/cyan]")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Ollama Monitor with CLI")
    parser.add_argument("--mode", choices=["realtime", "create-config"], default="realtime",
                       help="Operation mode")
    parser.add_argument("--model-name", type=str, help="Model name to use")
    parser.add_argument("--prompt", type=str, help="Prompt to generate text from")
    parser.add_argument("--host", type=str, help="Ollama server host")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")
    parser.add_argument("--n-gpu-layers", type=int, help="Number of GPU layers")
    parser.add_argument("--n-ctx", type=int, help="Context length")
    parser.add_argument("--top-k", type=int, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, help="Repeat penalty")
    parser.add_argument("--num-thread", type=int, help="Number of threads")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    if args.mode == "create-config":
        create_default_config()
        return
    
    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
        console.print(f"[green]✅ Loaded configuration from {args.config}[/green]")
    else:
        config = Config()
        console.print("[yellow]Using default configuration[/yellow]")
    
    # Override config with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.prompt:
        config.prompt = args.prompt
    if args.host:
        config.host = args.host
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.n_gpu_layers is not None:
        config.n_gpu_layers = args.n_gpu_layers
    if args.n_ctx:
        config.n_ctx = args.n_ctx
    if args.top_k:
        config.top_k = args.top_k
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.repeat_penalty is not None:
        config.repeat_penalty = args.repeat_penalty
    if args.num_thread is not None:
        config.num_thread = args.num_thread
    
    # Display current configuration
    console.print("[cyan]🔧 Current Configuration:[/cyan]")
    console.print(f"  Model: {config.model_name}")
    console.print(f"  Host: {config.host}")
    console.print(f"  Max Tokens: {config.max_tokens}")
    console.print(f"  Temperature: {config.temperature}")
    console.print(f"  Context Length: {config.n_ctx}")
    console.print(f"  GPU Layers: {config.n_gpu_layers}")
    console.print(f"  Prompt: {config.prompt[:50]}{'...' if len(config.prompt) > 50 else ''}")
    console.print()
    
    if args.mode == "realtime":
        run_realtime(config)

if __name__ == "__main__":
    main()