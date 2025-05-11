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
import random

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
    """Load test an LLM API with multiple prompts"""
    
    def __init__(self, endpoint, prompts, model, max_tokens=700, temperature=0):
        self.endpoint = endpoint
        self.prompts = prompts if isinstance(prompts, list) else [prompts]
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.results = []

    @staticmethod
    def load_prompts_from_file(filename):
        """Load prompts from a file, one per line"""
        prompts = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {filename}")
        except Exception as e:
            print(f"Error loading prompts from file: {e}")
            sys.exit(1)
        return prompts

    def send_request(self, prompt):
        """Send a single request to the API and measure time including TTFT"""
        start_time = time.time()
        first_byte_time = None
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True  # Enable streaming to measure TTFT
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            # For TTFT measurement we need streaming mode
            with requests.post(
                self.endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
                stream=True
            ) as response:
                status_code = response.status_code
                
                if status_code == 200:
                    content = ""
                    for chunk in response.iter_content(chunk_size=1):
                        # Record time of first byte
                        if first_byte_time is None and chunk:
                            first_byte_time = time.time()
                        content += chunk.decode('utf-8', errors='ignore')
                    
                    # If streaming didn't work, fall back to normal request
                    if first_byte_time is None:
                        first_byte_time = time.time()
                    
                    # Calculate metrics
                    ttft = first_byte_time - start_time
                    response_time = time.time() - start_time
                    
                    # Parse the response content
                    try:
                        response_json = json.loads(content)
                    except json.JSONDecodeError:
                        # If streaming response is not valid JSON, try again with non-streaming
                        payload["stream"] = False
                        response = requests.post(
                            self.endpoint,
                            headers=headers,
                            data=json.dumps(payload),
                            timeout=60
                        )
                        response_json = response.json()
                    
                    # Extract token count
                    tokens_generated = 0
                    if "choices" in response_json and response_json["choices"]:
                        if "text" in response_json["choices"][0]:
                            tokens_generated = len(response_json["choices"][0]["text"].split())
                    
                    # Calculate latency per token
                    latency_per_token = 0
                    if tokens_generated > 0:
                        latency_per_token = (response_time - ttft) / tokens_generated
                    
                    return {
                        "success": True,
                        "prompt": prompt,  # Include the prompt used for this request
                        "latency": response_time,
                        "ttft": ttft,
                        "tokens_generated": tokens_generated,
                        "latency_per_token": latency_per_token,
                        "status_code": status_code
                    }
                else:
                    return {
                        "success": False,
                        "prompt": prompt,  # Include the prompt used for this request
                        "latency": time.time() - start_time,
                        "ttft": None,
                        "status_code": status_code,
                        "error": response.text
                    }
        except Exception as e:
            return {
                "success": False,
                "prompt": prompt,  # Include the prompt used for this request
                "latency": time.time() - start_time,
                "ttft": None,
                "error": str(e)
            }

    def run_load_test(self, num_requests, concurrency, output_file=None, use_random_prompts=True):
        """Run the load test with specified concurrency using multiple prompts"""
        print(f"\nStarting load test with {num_requests} requests, concurrency={concurrency}")
        print(f"Model: {self.model}")
        print(f"Endpoint: {self.endpoint}")
        print(f"Using {len(self.prompts)} different prompts")
        print(f"Prompt selection: {'Random' if use_random_prompts else 'Sequential'}")
        
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
                # Distribute prompts across requests
                if use_random_prompts:
                    # Randomly select prompts for each request
                    selected_prompts = [random.choice(self.prompts) for _ in range(num_requests)]
                else:
                    # Use prompts sequentially, cycling through the list
                    selected_prompts = [self.prompts[i % len(self.prompts)] for i in range(num_requests)]
                
                # Submit tasks with their specific prompts
                futures = [executor.submit(self.send_request, prompt) for prompt in selected_prompts]
                
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
            
            # Time to First Token metrics
            if "ttft" in successful_requests[0] and successful_requests[0]["ttft"]:
                ttfts = [r["ttft"] for r in successful_requests if r.get("ttft")]
                avg_ttft = np.mean(ttfts)
                p50_ttft = np.percentile(ttfts, 50)
                p90_ttft = np.percentile(ttfts, 90)
                p99_ttft = np.percentile(ttfts, 99)
            else:
                avg_ttft = p50_ttft = p90_ttft = p99_ttft = "N/A"
            
            # Per token latency metrics
            if "latency_per_token" in successful_requests[0]:
                per_token_latencies = [r["latency_per_token"] for r in successful_requests 
                                    if r.get("latency_per_token") and r.get("tokens_generated", 0) > 0]
                if per_token_latencies:
                    avg_latency_per_token = np.mean(per_token_latencies)
                    p50_latency_per_token = np.percentile(per_token_latencies, 50)
                    p90_latency_per_token = np.percentile(per_token_latencies, 90)
                else:
                    avg_latency_per_token = p50_latency_per_token = p90_latency_per_token = "N/A"
            else:
                avg_latency_per_token = p50_latency_per_token = p90_latency_per_token = "N/A"
            
            if "tokens_generated" in successful_requests[0]:
                tokens = [r["tokens_generated"] for r in successful_requests]
                avg_tokens = np.mean(tokens)
                tokens_per_second = sum(tokens) / total_time
            else:
                avg_tokens = "N/A"
                tokens_per_second = "N/A"
        else:
            p50 = p90 = p99 = avg_latency = 0
            avg_ttft = p50_ttft = p90_ttft = p99_ttft = "N/A"
            avg_latency_per_token = p50_latency_per_token = p90_latency_per_token = "N/A"
            avg_tokens = tokens_per_second = "N/A"
        
        # Count prompts used
        unique_prompts = len(set(r.get("prompt", "") for r in self.results))
        
        # Print report
        print("\n" + "="*70)
        print("LOAD TEST RESULTS")
        print("="*70)
        print(f"Total requests: {len(self.results)}")
        print(f"Unique prompts used: {unique_prompts}")
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
        
        print("\nTime To First Token (TTFT) statistics (milliseconds):")
        if isinstance(avg_ttft, float):
            print(f"  Average: {avg_ttft*1000:.2f} ms")
            print(f"  P50: {p50_ttft*1000:.2f} ms")
            print(f"  P90: {p90_ttft*1000:.2f} ms")
            print(f"  P99: {p99_ttft*1000:.2f} ms")
        else:
            print(f"  TTFT metrics not available")
        
        print("\nLatency per token statistics (milliseconds/token):")
        if isinstance(avg_latency_per_token, float):
            print(f"  Average: {avg_latency_per_token*1000:.2f} ms/token")
            print(f"  P50: {p50_latency_per_token*1000:.2f} ms/token")
            print(f"  P90: {p90_latency_per_token*1000:.2f} ms/token")
        else:
            print(f"  Latency per token metrics not available")
        
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
        
        print("="*70)
        
        # Save results to file if specified
        if output_file:
            result_data = {
                "test_info": {
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "unique_prompts_used": unique_prompts,
                    "total_prompts_available": len(self.prompts),
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
                        "average": avg_latency * 1000 if isinstance(avg_latency, float) else avg_latency,
                        "p50": p50 * 1000 if isinstance(p50, float) else p50,
                        "p90": p90 * 1000 if isinstance(p90, float) else p90,
                        "p99": p99 * 1000 if isinstance(p99, float) else p99
                    },
                    "ttft_ms": {
                        "average": avg_ttft * 1000 if isinstance(avg_ttft, float) else avg_ttft,
                        "p50": p50_ttft * 1000 if isinstance(p50_ttft, float) else p50_ttft,
                        "p90": p90_ttft * 1000 if isinstance(p90_ttft, float) else p90_ttft,
                        "p99": p99_ttft * 1000 if isinstance(p99_ttft, float) else p99_ttft
                    },
                    "latency_per_token_ms": {
                        "average": avg_latency_per_token * 1000 if isinstance(avg_latency_per_token, float) else avg_latency_per_token,
                        "p50": p50_latency_per_token * 1000 if isinstance(p50_latency_per_token, float) else p50_latency_per_token,
                        "p90": p90_latency_per_token * 1000 if isinstance(p90_latency_per_token, float) else p90_latency_per_token
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
            # Extract data from successful requests
            successful = [r for r in self.results if r["success"]]
            if not successful:
                print("No successful requests to plot")
                return
                
            latencies = [r["latency"] * 1000 for r in successful]  # Convert to ms
            
            # Extract TTFT data if available
            has_ttft = "ttft" in successful[0] and successful[0]["ttft"] is not None
            if has_ttft:
                ttfts = [r["ttft"] * 1000 for r in successful if r.get("ttft") is not None]  # Convert to ms
            
            # Extract per-token latency data if available
            has_per_token = "latency_per_token" in successful[0] and successful[0]["latency_per_token"] is not None
            if has_per_token:
                per_token_latencies = [r["latency_per_token"] * 1000 for r in successful 
                                      if r.get("latency_per_token") is not None 
                                      and r.get("tokens_generated", 0) > 0]  # Convert to ms
            
            # Create subplot layout based on available metrics
            num_plots = 1 + int(has_ttft) + int(has_per_token)
            fig, axs = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
            
            if num_plots == 1:
                axs = [axs]  # Make axs a list for indexing when there's only one plot
            
            # Plot 1: Total Latency Distribution
            ax_idx = 0
            axs[ax_idx].hist(latencies, bins=30, alpha=0.7, color='blue')
            axs[ax_idx].axvline(np.mean(latencies), color='red', linestyle='dashed', linewidth=1, 
                              label=f'Mean: {np.mean(latencies):.2f} ms')
            axs[ax_idx].axvline(np.percentile(latencies, 90), color='green', linestyle='dashed', linewidth=1,
                              label=f'p90: {np.percentile(latencies, 90):.2f} ms')
            axs[ax_idx].axvline(np.percentile(latencies, 99), color='orange', linestyle='dashed', linewidth=1,
                              label=f'p99: {np.percentile(latencies, 99):.2f} ms')
            
            axs[ax_idx].set_title('Response Latency Distribution')
            axs[ax_idx].set_xlabel('Latency (ms)')
            axs[ax_idx].set_ylabel('Number of Requests')
            axs[ax_idx].legend()
            axs[ax_idx].grid(True, alpha=0.3)
            
            # Plot 2: Time to First Token (TTFT) Distribution
            if has_ttft:
                ax_idx += 1
                axs[ax_idx].hist(ttfts, bins=30, alpha=0.7, color='green')
                axs[ax_idx].axvline(np.mean(ttfts), color='red', linestyle='dashed', linewidth=1, 
                                  label=f'Mean: {np.mean(ttfts):.2f} ms')
                axs[ax_idx].axvline(np.percentile(ttfts, 90), color='blue', linestyle='dashed', linewidth=1,
                                  label=f'p90: {np.percentile(ttfts, 90):.2f} ms')
                
                axs[ax_idx].set_title('Time to First Token (TTFT) Distribution')
                axs[ax_idx].set_xlabel('TTFT (ms)')
                axs[ax_idx].set_ylabel('Number of Requests')
                axs[ax_idx].legend()
                axs[ax_idx].grid(True, alpha=0.3)
            
            # Plot 3: Latency per Token Distribution
            if has_per_token and per_token_latencies:
                ax_idx += 1
                axs[ax_idx].hist(per_token_latencies, bins=30, alpha=0.7, color='purple')
                axs[ax_idx].axvline(np.mean(per_token_latencies), color='red', linestyle='dashed', linewidth=1, 
                                  label=f'Mean: {np.mean(per_token_latencies):.2f} ms/token')
                axs[ax_idx].axvline(np.percentile(per_token_latencies, 90), color='blue', linestyle='dashed', linewidth=1,
                                  label=f'p90: {np.percentile(per_token_latencies, 90):.2f} ms/token')
                
                axs[ax_idx].set_title('Latency per Token Distribution')
                axs[ax_idx].set_xlabel('Latency per Token (ms/token)')
                axs[ax_idx].set_ylabel('Number of Requests')
                axs[ax_idx].legend()
                axs[ax_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_file.replace('.json', '_metrics.png')
            plt.savefig(plot_file)
            print(f"Metrics plots saved to {plot_file}")
            
            # Add a scatter plot to show latency vs token count relationship
            if "tokens_generated" in successful[0]:
                tokens = [r["tokens_generated"] for r in successful]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(tokens, latencies, alpha=0.6)
                
                # Add trend line
                if len(tokens) > 1:
                    z = np.polyfit(tokens, latencies, 1)
                    p = np.poly1d(z)
                    plt.plot(tokens, p(tokens), "r--", alpha=0.8, 
                             label=f"Trend: y={z[0]:.2f}x + {z[1]:.2f}")
                
                plt.title('Latency vs Token Count')
                plt.xlabel('Number of Tokens Generated')
                plt.ylabel('Latency (ms)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plot_file = output_file.replace('.json', '_latency_vs_tokens.png')
                plt.savefig(plot_file)
                print(f"Latency vs Tokens plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Load test an LLM API endpoint with multiple prompts")
    
    # Test configuration
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions", 
                        help="API endpoint URL")
    parser.add_argument("--model", type=str, 
                        default="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
                        help="Model name")
    parser.add_argument("--prompt", type=str, default="San Francisco is a",
                        help="Single prompt to use for testing (if prompt_file is not provided)")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="File containing prompts (one per line) to use for testing")
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
    parser.add_argument("--no_streaming", action="store_true",
                        help="Disable streaming mode (won't measure TTFT accurately)")
    parser.add_argument("--sequential_prompts", action="store_true",
                        help="Use prompts sequentially instead of randomly")
    
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
    
    # Load testing mode - First determine which prompts to use
    prompts = []
    
    if args.prompt_file:
        # Load prompts from file
        prompts = LLMLoadTester.load_prompts_from_file(args.prompt_file)
    elif args.prompt:
        # Use the single prompt provided
        prompts = [args.prompt]
    else:
        print("Error: Either --prompt or --prompt_file must be provided")
        sys.exit(1)
    
    # Create and run the load tester
    tester = LLMLoadTester(
        endpoint=args.endpoint,
        prompts=prompts,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    tester.run_load_test(
        args.num_requests, 
        args.concurrency, 
        args.output,
        use_random_prompts=not args.sequential_prompts
    )


if __name__ == "__main__":
    main()
