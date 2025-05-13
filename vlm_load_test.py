import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import json
import threading
import datetime

from openai import OpenAI
from tqdm import tqdm

# Import GPU monitoring libraries
try:
    import pynvml
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: pynvml not installed. GPU metrics will not be available.")
    print("Install with: pip install pynvml")

@dataclass
class GPUMetrics:
    timestamp: float
    gpu_id: int
    utilization: int  # GPU utilization percentage
    memory_used: int  # Memory used in MB
    memory_total: int  # Total memory in MB
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_human": datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
            "gpu_id": self.gpu_id,
            "utilization": self.utilization,
            "memory_used_mb": self.memory_used,
            "memory_total_mb": self.memory_total,
            "memory_used_percent": (self.memory_used / self.memory_total * 100) if self.memory_total > 0 else 0
        }

@dataclass
class RequestMetrics:
    start_time: float
    end_time: float
    ttft: float  # Time to first token
    total_tokens: int
    success: bool
    error_message: str = None

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_seconds": self.total_time,
            "ttft_seconds": self.ttft,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "error_message": self.error_message
        }

class GPUMonitor:
    def __init__(self, polling_interval: float = 0.5):
        """
        Initialize GPU monitor.
        
        Args:
            polling_interval: Time in seconds between GPU metric collections
        """
        self.polling_interval = polling_interval
        self.should_stop = threading.Event()
        self.metrics: List[GPUMetrics] = []
        self.gpu_count = 0
        self.thread: Optional[threading.Thread] = None
        
        if HAS_GPU:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                print(f"Found {self.gpu_count} GPUs")
            except Exception as e:
                print(f"Error initializing NVML: {e}")
                self.gpu_count = 0
    
    def _collect_gpu_metrics(self) -> List[GPUMetrics]:
        """Collect metrics from all available GPUs."""
        current_metrics = []
        
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu  # GPU utilization percentage
                
                # Get memory information
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = memory.used // 1024 // 1024  # Convert to MB
                mem_total = memory.total // 1024 // 1024  # Convert to MB
                
                current_metrics.append(GPUMetrics(
                    timestamp=time.time(),
                    gpu_id=i,
                    utilization=gpu_util,
                    memory_used=mem_used,
                    memory_total=mem_total
                ))
            except Exception as e:
                print(f"Error collecting metrics for GPU {i}: {e}")
        
        return current_metrics
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while not self.should_stop.is_set():
            try:
                current_metrics = self._collect_gpu_metrics()
                self.metrics.extend(current_metrics)
            except Exception as e:
                print(f"Error in GPU monitoring loop: {e}")
            
            time.sleep(self.polling_interval)
    
    def start(self):
        """Start GPU monitoring in a separate thread."""
        if not HAS_GPU or self.gpu_count == 0:
            print("No GPUs available for monitoring")
            return
        
        self.should_stop.clear()
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        print(f"GPU monitoring started with polling interval of {self.polling_interval}s")
    
    def stop(self):
        """Stop GPU monitoring."""
        if self.thread and self.thread.is_alive():
            self.should_stop.set()
            self.thread.join(timeout=2.0)
            print("GPU monitoring stopped")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for GPU metrics."""
        if not self.metrics:
            return {"error": "No GPU metrics collected"}
        
        summary = {}
        for gpu_id in range(self.gpu_count):
            gpu_metrics = [m for m in self.metrics if m.gpu_id == gpu_id]
            if not gpu_metrics:
                continue
            
            utilization = [m.utilization for m in gpu_metrics]
            memory_used_percent = [(m.memory_used / m.memory_total * 100) if m.memory_total > 0 else 0 for m in gpu_metrics]
            
            summary[f"gpu_{gpu_id}"] = {
                "utilization_percent": {
                    "min": min(utilization),
                    "max": max(utilization),
                    "mean": statistics.mean(utilization),
                    "median": statistics.median(utilization),
                    "p95": sorted(utilization)[int(len(utilization)*0.95)] if len(utilization) > 20 else max(utilization)
                },
                "memory_used_percent": {
                    "min": min(memory_used_percent),
                    "max": max(memory_used_percent),
                    "mean": statistics.mean(memory_used_percent),
                    "median": statistics.median(memory_used_percent),
                    "p95": sorted(memory_used_percent)[int(len(memory_used_percent)*0.95)] if len(memory_used_percent) > 20 else max(memory_used_percent)
                }
            }
        
        return summary

class LLMLoadTester:
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model_name: str = None,
        concurrency: int = 5,
        total_requests: int = 50,
        image_url: str = "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg",
        prompt: str = "describe this image",
        monitor_gpu: bool = True,
        gpu_polling_interval: float = 0.5
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.image_url = image_url
        self.prompt = prompt
        self.monitor_gpu = monitor_gpu and HAS_GPU
        
        # Initialize client and get model name if not provided
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        if model_name is None:
            try:
                self.model_name = self.client.models.list().data[0].id
            except Exception as e:
                print(f"Error getting model list: {e}")
                self.model_name = "default_model"
        else:
            self.model_name = model_name
            
        self.results: List[RequestMetrics] = []
        
        # Initialize GPU monitor if requested
        self.gpu_monitor = None
        if self.monitor_gpu:
            self.gpu_monitor = GPUMonitor(polling_interval=gpu_polling_interval)
    
    def _make_single_request(self) -> RequestMetrics:
        """Make a single request to the LLM and return metrics."""
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        try:
            start_time = time.time()
            ttft_recorded = False
            ttft = None
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': self.prompt,
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url': self.image_url,
                        },
                    }],
                }],
                temperature=0.8,
                top_p=0.8,
                stream=True  # Enable streaming to measure TTFT
            )
            
            # Process the stream to measure TTFT
            content = ""
            for chunk in response:
                if not ttft_recorded and chunk.choices and chunk.choices[0].delta.content:
                    ttft = time.time() - start_time
                    ttft_recorded = True
                
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            end_time = time.time()
            
            # Estimate token count (simple approximation)
            total_tokens = len(content.split()) * 1.3  # Rough estimate
            
            return RequestMetrics(
                start_time=start_time,
                end_time=end_time,
                ttft=ttft if ttft else (end_time - start_time),  # Fallback if TTFT wasn't captured
                total_tokens=int(total_tokens),
                success=True
            )
            
        except Exception as e:
            end_time = time.time()
            return RequestMetrics(
                start_time=start_time,
                end_time=end_time,
                ttft=0.0,
                total_tokens=0,
                success=False,
                error_message=str(e)
            )
    
    def run_load_test(self):
        """Run the load test with specified concurrency."""
        print(f"Starting load test with {self.concurrency} concurrent requests...")
        print(f"Total requests to send: {self.total_requests}")
        print(f"Model: {self.model_name}")
        print(f"Base URL: {self.base_url}")
        
        # Start GPU monitoring if enabled
        if self.gpu_monitor:
            self.gpu_monitor.start()
        
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for _ in range(self.total_requests):
                futures.append(executor.submit(self._make_single_request))
            
            # Track progress
            for future in tqdm(futures, total=self.total_requests, desc="Processing requests"):
                self.results.append(future.result())
        
        # Stop GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
        
        print("Load test completed. Analyzing results...")
        self._analyze_results()
    
    def _analyze_results(self):
        """Analyze and print the results of the load test."""
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]
        
        if not successful_requests:
            print("No successful requests to analyze!")
            return
        
        # Calculate metrics
        total_time = max(r.end_time for r in self.results) - min(r.start_time for r in self.results)
        requests_per_second = len(successful_requests) / total_time if total_time > 0 else 0
        
        latencies = [r.total_time for r in successful_requests]
        ttfts = [r.ttft for r in successful_requests]
        
        # Print summary
        print("\n==== LOAD TEST RESULTS ====")
        print(f"Total requests sent: {len(self.results)}")
        print(f"Successful requests: {len(successful_requests)} ({len(successful_requests)/len(self.results)*100:.2f}%)")
        print(f"Failed requests: {len(failed_requests)} ({len(failed_requests)/len(self.results)*100:.2f}%)")
        print(f"Total test duration: {total_time:.2f} seconds")
        print(f"Requests per second: {requests_per_second:.2f}")
        
        if latencies:
            print("\n==== LATENCY (seconds) ====")
            print(f"Min: {min(latencies):.4f}")
            print(f"Max: {max(latencies):.4f}")
            print(f"Mean: {statistics.mean(latencies):.4f}")
            print(f"Median: {statistics.median(latencies):.4f}")
            print(f"P95: {sorted(latencies)[int(len(latencies)*0.95)]:.4f}")
            print(f"P99: {sorted(latencies)[int(len(latencies)*0.99)]:.4f}")
        
        if ttfts:
            print("\n==== TIME TO FIRST TOKEN (seconds) ====")
            print(f"Min: {min(ttfts):.4f}")
            print(f"Max: {max(ttfts):.4f}")
            print(f"Mean: {statistics.mean(ttfts):.4f}")
            print(f"Median: {statistics.median(ttfts):.4f}")
            print(f"P95: {sorted(ttfts)[int(len(ttfts)*0.95)]:.4f}")
            print(f"P99: {sorted(ttfts)[int(len(ttfts)*0.99)]:.4f}")
        
        # Print GPU metrics if available
        if self.gpu_monitor and self.gpu_monitor.metrics:
            print("\n==== GPU METRICS ====")
            gpu_summary = self.gpu_monitor.get_summary()
            
            for gpu_id, metrics in gpu_summary.items():
                print(f"\nGPU {gpu_id.split('_')[1]}:")
                
                print("  Utilization (%):")
                print(f"    Min: {metrics['utilization_percent']['min']}")
                print(f"    Max: {metrics['utilization_percent']['max']}")
                print(f"    Mean: {metrics['utilization_percent']['mean']:.2f}")
                print(f"    P95: {metrics['utilization_percent']['p95']}")
                
                print("  Memory Usage (%):")
                print(f"    Min: {metrics['memory_used_percent']['min']:.2f}")
                print(f"    Max: {metrics['memory_used_percent']['max']:.2f}")
                print(f"    Mean: {metrics['memory_used_percent']['mean']:.2f}")
                print(f"    P95: {metrics['memory_used_percent']['p95']:.2f}")
        
        if failed_requests:
            print("\n==== ERROR SUMMARY ====")
            error_counts = {}
            for req in failed_requests:
                error_counts[req.error_message] = error_counts.get(req.error_message, 0) + 1
            
            for error, count in error_counts.items():
                print(f"[{count}x] {error[:100]}...")
    
    def save_results(self, filepath: str = "llm_load_test_results.json"):
        """Save test results to a JSON file."""
        output = {
            "test_config": {
                "model": self.model_name,
                "base_url": self.base_url,
                "concurrency": self.concurrency,
                "total_requests": self.total_requests,
                "image_url": self.image_url,
                "prompt": self.prompt,
                "monitor_gpu": self.monitor_gpu
            },
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_requests": len(self.results),
                "successful_requests": len([r for r in self.results if r.success]),
                "failed_requests": len([r for r in self.results if not r.success]),
                "requests_per_second": len([r for r in self.results if r.success]) / 
                    (max(r.end_time for r in self.results) - min(r.start_time for r in self.results)) 
                    if self.results else 0
            }
        }
        
        # Add GPU metrics if available
        if self.gpu_monitor and self.gpu_monitor.metrics:
            output["gpu_metrics"] = {
                "summary": self.gpu_monitor.get_summary(),
                "detailed": [m.to_dict() for m in self.gpu_monitor.metrics]
            }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
            
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    load_tester = LLMLoadTester(
        api_key="thieu",
        base_url="http://0.0.0.0:8000/v1",
        concurrency=5,  # Number of concurrent requests
        total_requests=50,  # Total number of requests to send
        image_url="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg",
        prompt="describe this image",
        monitor_gpu=True,  # Enable GPU monitoring
        gpu_polling_interval=0.5  # Collect GPU metrics every 0.5 seconds
    )
    
    load_tester.run_load_test()
    load_tester.save_results()
