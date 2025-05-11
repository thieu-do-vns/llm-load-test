**LLM Load Testing Tool**

This tool provides comprehensive load testing for LLM APIs with the following features:
- Concurrent request handling
- Latency measurement 
- Throughput calculation
- RPS (Requests Per Second) tracking
- GPU memory usage monitoring (requires running on the LLM server)

***Usage:***
1. Run the load test with cache:
   python llm_load_test.py --num_requests 100 --concurrency 10
   
3. Run in seperate prompts
   python script.py --prompt_file prompts.txt --num_requests 100 --concurrency 10 --output results.json

4. For GPU monitoring (on the server):
   python llm_load_test.py --monitor_gpu --interval 0.5

