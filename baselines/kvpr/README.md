# KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation
This is the source code for the paper KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial
Recomputation (ACL 2025). In this work, we propose an efficient CPU-GPU I/O-aware LLM inference method that leverages partial KV cache recomputation and asynchronous overlapping to address the system bottleneck of loading large KV cache from host CPU DRAM to GPU memory.

```
@article{jiang2024efficient,
  title={Efficient llm inference with i/o-aware partial kv cache recomputation},
  author={Jiang, Chaoyi and Gao, Lei and Zarch, Hossein Entezari and Annavaram, Murali},
  journal={arXiv preprint arXiv:2411.17089},
  year={2024}
}
```

## Usage and Examples
Our implementation builds upon FlexGen https://github.com/FMInference/FlexLLMGen, and Hugging Face Transformers https://github.com/huggingface/transformers.

### Command to run our latency-oriented test
```
python run_hf_ds.py --model facebook/opt-6.7b --batch-size 32 --prompt-len 256 --gen-len 32 --kv-partial --csv-file "optimal_latency_benchmark.csv" &
```

### Command to run our throughput-oriented test
```
python -m flexgen.flex_opt --model facebook/opt-6.7b --path OPT_WEIGHTS_FOLDER --num-test-runs 1 --prompt-len 256 --gen-len 32 --gpu-batch-size 32 --num-gpu-batches 1 --offload-dir YOUR_SSD_FOLDER --percent 0 100 0 100 0 100 --kv-partial --overlap --pin-weight
```

### Command to run Hugging Face Accelerate baseline
```
python run_hf_ds.py --model facebook/opt-6.7b --batch-size 32 --prompt-len 256 --gen-len 32 --csv-file "baseline_latency_benchmark.csv" &
```

### Command to run FlexGen baseline
```
python -m flexgen.flex_opt --model facebook/opt-6.7b --path OPT_WEIGHTS_FOLDER --num-test-runs 1 --prompt-len 256 --gen-len 32 --gpu-batch-size 32 --num-gpu-batches 8 --offload-dir YOUR_SSD_FOLDER --percent 0 100 0 100 0 100 --overlap --pin-weight
```

## Contact
If you have any questions, feel free to contact me through email (chaoyij@usc.edu).
