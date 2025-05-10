import requests
import time
import json
import concurrent.futures
import numpy as np
import argparse
import subprocess
import csv
import os
import signal
import sys
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process, Event

class GPUMonitor:
    """Monitor GPU memory usage"""
    
    def __init__(self, interval=0.5, output_file="gpu_monitor.csv"):
        self.interval = interval
        self.output_file = output_file
        self.stop_event = Event()
        self.process = None
        
    def get_gpu_memory_usage(self):
        """Get GPU memory usage using nvidia-smi"""
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']
            ).decode('utf-8')
            
            gpus = []
            for line in output.strip().split('\n'):
                index, mem_used, mem_total, util = line.split(',')
                gpus.append({
                    'index': int(index),
                    'memory_used_mb': float(mem_used),
                    'memory_total_mb': float(mem_total),
                    'utilization_percent': float(util)
                })
            return gpus
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return []
    
    def monitor_loop(self, stop_event, interval, output_file):
        """Background monitoring process"""
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'gpu_index', 'memory_used_mb', 'memory_total_mb', 'utilization_percent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            while not stop_event.is_set():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                gpus = self.get_gpu_memory_usage()
                
                for gpu in gpus:
                    writer.writerow({
                        'timestamp': timestamp,
                        'gpu_index': gpu['index'],
                        'memory_used_mb': gpu['memory_used_mb'],
                        'memory_total_mb': gpu['memory_total_mb'],
                        'utilization_percent': gpu['utilization_percent']
                    })
                
                # Flush to ensure data is written immediately
                csvfile.flush()
                time.sleep(interval)
    
    def start(self):
        """Start GPU monitoring in a separate process"""
        if self.process is not None and self.process.is_alive():
            print("GPU monitoring is already running")
            return
            
        self.stop_event.clear()
        self.process = Process(
            target=self.monitor_loop,
            args=(self.stop_event, self.interval, self.output_file)
        )
        self.process.start()
        print(f"Started GPU monitoring, logging to {self.output_file}")
        
    def stop(self):
        """Stop GPU monitoring"""
        if self.process is None or not self.process.is_alive():
            return
            
        self.stop_event.set()
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
        self.process = None
        print(f"Stopped GPU monitoring, data saved to {self.output_file}")
        
    def plot_results(self):
        """Plot GPU monitoring results"""
        if not os.path.exists(self.output_file):
            print(f"Error: Monitoring file {self.output_file} not found")
            return
            
        # Read the CSV file
        timestamps = []
        gpu_data = {}
        
        with open(self.output_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp = datetime.strptime(row['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                gpu_index = int(row['gpu_index'])
                
                if len(timestamps) == 0 or timestamp != timestamps[-1]:
                    timestamps.append(timestamp)
                
                if gpu_index not in gpu_data:
                    gpu_data[gpu_index] = {
                        'memory_used': [],
                        'utilization': []
                    }
                    
                gpu_data[gpu_index]['memory_used'].append(float(row['memory_used_mb']))
                gpu_data[gpu_index]['utilization'].append(float(row['utilization_percent']))
        
        # Convert timestamps to relative seconds
        if not timestamps:
            print("No data found in the monitoring file")
            return
            
        start_time = timestamps[0]
        rel_times = [(t - start_time).total_seconds() for t in timestamps]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        for gpu_idx, data in gpu_data.items():
            ax1.plot(rel_times, data['memory_used'], label=f'GPU {gpu_idx}')
            ax2.plot(rel_times, data['utilization'], label=f'GPU {gpu_idx}')
        
        ax1.set_title('GPU Memory Usage')
        ax1.set_ylabel('Memory Used (MB)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_title('GPU Utilization')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Utilization (%)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_file = self.output_file.replace('.csv', '.png')
        plt.savefig(plot_file)
        print(f"GPU monitoring plot saved to {plot_file}")


class LLMLoadTester:
    """Load test an LLM API"""
    
    def __init__(self, endpoint, prompt, model, max_tokens=700, temperature=0):
        self.endpoint = endpoint
        self.prompt = prompt
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results = []

    def send_request(self):
        """Send a single request to the API and measure time"""
        start_time = time.time()
        
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            
            response_time = time.time() - start_time
            status_code = response.status_code
            
            if status_code == 200:
                # Try to extract token count from response if possible
                response_json = response.json()
                tokens_generated = 0
                
                if "choices" in response_json and response_json["choices"]:
                    if "text" in response_json["choices"][0]:
                        tokens_generated = len(response_json["choices"][0]["text"].split())
                
                return {
                    "success": True,
                    "latency": response_time,
                    "tokens_generated": tokens_generated,
                    "status_code": status_code
                }
            else:
                return {
                    "success": False,
                    "latency": response_time,
                    "status_code": status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "latency": time.time() - start_time,
                "error": str(e)
            }

    def run_load_test(self, num_requests, concurrency, output_file=None):
        """Run the load test with specified concurrency"""
        print(f"\nStarting load test with {num_requests} requests, concurrency={concurrency}")
        print(f"Model: {self.model}")
        print(f"Endpoint: {self.endpoint}")
        print(f"Prompt: '{self.prompt}'")
        
        start_time = time.time()
        self.results = []
        
        # Initialize GPU monitor if available
        gpu_monitor = None
        if output_file:
            gpu_output = output_file.replace('.json', '_gpu.csv')
            gpu_monitor = GPUMonitor(interval=0.5, output_file=gpu_output)
            try:
                gpu_monitor.start()
            except Exception as e:
                print(f"Warning: Could not start GPU monitoring: {e}")
                gpu_monitor = None
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(self.send_request) for _ in range(num_requests)]
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=num_requests):
                    self.results.append(future.result())
            
            total_time = time.time() - start_time
            self.analyze_results(total_time, output_file)
            
        finally:
            # Stop GPU monitoring if running
            if gpu_monitor:
                gpu_monitor.stop()
                try:
                    gpu_monitor.plot_results()
                except Exception as e:
                    print(f"Warning: Could not plot GPU results: {e}")

    def analyze_results(self, total_time, output_file=None):
        """Analyze test results"""
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        # Calculate metrics
        success_rate = len(successful_requests) / len(self.results) * 100 if self.results else 0
        rps = len(self.results) / total_time
        
        if successful_requests:
            latencies = [r["latency"] for r in successful_requests]
            p50 = np.percentile(latencies, 50)
            p90 = np.percentile(latencies, 90)
            p99 = np.percentile(latencies, 99)
            avg_latency = np.mean(latencies)
            
            if "tokens_generated" in successful_requests[0]:
                tokens = [r["tokens_generated"] for r in successful_requests]
                avg_tokens = np.mean(tokens)
                tokens_per_second = sum(tokens) / total_time
            else:
                avg_tokens = "N/A"
                tokens_per_second = "N/A"
        else:
            p50, p90, p99, avg_latency = 0, 0, 0, 0
            avg_tokens, tokens_per_second = "N/A", "N/A"
        
        # Print report
        print("\n" + "="*50)
        print("LOAD TEST RESULTS")
        print("="*50)
        print(f"Total requests: {len(self.results)}")
        print(f"Successful requests: {len(successful_requests)} ({success_rate:.2f}%)")
        print(f"Failed requests: {len(failed_requests)}")
        print(f"Total test time: {total_time:.2f} seconds")
        print(f"Requests per second (RPS): {rps:.2f}")
        print(f"Average tokens per response: {avg_tokens}")
        print(f"Tokens per second (throughput): {tokens_per_second}")
        print("\nLatency statistics (milliseconds):")
        print(f"  Average: {avg_latency*1000:.2f} ms")
        print(f"  P50: {p50*1000:.2f} ms")
        print(f"  P90: {p90*1000:.2f} ms")
        print(f"  P99: {p99*1000:.2f} ms")
        
        if failed_requests:
            error_counts = {}
            for req in failed_requests:
                error = req.get("error", "Unknown error")
                if isinstance(error, str) and len(error) > 100:
                    error = error[:100] + "..."
                error_counts[error] = error_counts.get(error, 0) + 1
            
            print("\nError distribution:")
            for error, count in error_counts.items():
                print(f"  - {error}: {count} occurrences")
        
        print("="*50)
        
        # Save results to file if specified
        if output_file:
            result_data = {
                "test_info": {
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "prompt": self.prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "concurrency": len(self.results) // total_time,
                    "total_requests": len(self.results),
                    "test_duration_seconds": total_time
                },
                "metrics": {
                    "success_rate": success_rate,
                    "requests_per_second": rps,
                    "avg_tokens_per_response": avg_tokens,
                    "tokens_per_second": tokens_per_second,
                    "latency_ms": {
                        "average": avg_latency * 1000,
                        "p50": p50 * 1000,
                        "p90": p90 * 1000,
                        "p99": p99 * 1000
                    }
                },
                "raw_results": self.results
            }
            
            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"Results saved to {output_file}")
            
            # Generate plots
            self.plot_results(output_file)
    
    def plot_results(self, output_file):
        """Generate plots for load test results"""
        try:
            # Extract latencies from successful requests
            successful = [r for r in self.results if r["success"]]
            if not successful:
                print("No successful requests to plot")
                return
                
            latencies = [r["latency"] * 1000 for r in successful]  # Convert to ms
            
            # Create distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(latencies, bins=30, alpha=0.7, color='blue')
            plt.axvline(np.mean(latencies), color='red', linestyle='dashed', linewidth=1, 
                     label=f'Mean: {np.mean(latencies):.2f} ms')
            plt.axvline(np.percentile(latencies, 90), color='green', linestyle='dashed', linewidth=1,
                     label=f'p90: {np.percentile(latencies, 90):.2f} ms')
            plt.axvline(np.percentile(latencies, 99), color='orange', linestyle='dashed', linewidth=1,
                     label=f'p99: {np.percentile(latencies, 99):.2f} ms')
            
            plt.title('Response Latency Distribution')
            plt.xlabel('Latency (ms)')
            plt.ylabel('Number of Requests')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = output_file.replace('.json', '_latency.png')
            plt.savefig(plot_file)
            print(f"Latency plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load test an LLM API endpoint")
    
    # Test configuration
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions", 
                        help="API endpoint URL")
    parser.add_argument("--model", type=str, 
                        default="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
                        help="Model name")
    parser.add_argument("--prompt", type=str, default="San Francisco is a",
                        help="Prompt to use for testing")
    parser.add_argument("--num_requests", type=int, default=100,
                        help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent requests")
    parser.add_argument("--max_tokens", type=int, default=700,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature")
    parser.add_argument("--output", type=str, default="load_test_results.json",
                        help="Output file for results")
    
    # GPU monitoring mode
    parser.add_argument("--monitor_gpu", action="store_true",
                        help="Run in GPU monitoring mode only")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="GPU monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # GPU monitoring only mode
    if args.monitor_gpu:
        monitor = GPUMonitor(interval=args.interval, output_file="gpu_monitor.csv")
        print("Starting GPU monitoring. Press Ctrl+C to stop.")
        
        try:
            monitor.start()
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        finally:
            monitor.stop()
            monitor.plot_results()
        return
    
    # Load testing mode
    tester = LLMLoadTester(
        endpoint=args.endpoint,
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    tester.run_load_test(args.num_requests, args.concurrency, args.output)

if __name__ == "__main__":
    main()
