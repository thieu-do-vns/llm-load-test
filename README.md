**LLM Load Testing Tool**

This tool provides comprehensive load testing for LLM APIs with the following features:
- Concurrent request handling
- Latency measurement 
- Throughput calculation
- RPS (Requests Per Second) tracking
- GPU memory usage monitoring (requires running on the LLM server)

***Usage:***
1. Run the load test with cache:
   python llm_load_test.py --num_requests 200 --concurrency 20
   
3. Run in seperate prompts
   python llm_load_test_from_file.py --prompt_file prompts.txt --num_requests 200 --concurrency 20 --output results.json

4. For GPU monitoring (on the server):
   python llm_load_test.py --monitor_gpu --interval 0.5

