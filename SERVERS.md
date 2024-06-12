# Local Servers

## vLLM

Download an appropriate GPTQ or AWQ quant for your system such as [study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4](https://huggingface.co/study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int4).

Launch a vllm openAI server:

```
python3 -m vllm.entrypoints.openai.api_server --model ~/models/study-hjt-Meta-Llama-3-70B-Instruct-GPTQ-Int4/ --enable-prefix-cache
```

### Multi-GPU

Add `--tensor-parallel-size <N>` if you have multiple GPUs.

### If you have 2x24GB / 4x16GB and want to run Llama3-70B

Add `--enforce-eager --max-model-len 2048 --gpu_memory_utilization 1.0` to vllm server command line.

### If you have P100

Use [vllm-ci](https://github.com/sasha0552/vllm-ci)

`export VLLM_ATTENTION_BACKEND=XFORMERS` to force xformers.

Add `--dtype half` to vllm server command line.

If you see the "Cannot convert f16 to f16 crash" either remove `--enable-prefix-cache` (not recommended) or install vllm from the [vllm-ci](https://github.com/sasha0552/vllm-ci) repo.

## TabbyAPI

TODO

## llama.cpp

Download an appropriate quant for your system from [dolphin-2.9-llama3-70b-GGUF](https://huggingface.co/crusoeai/dolphin-2.9-llama3-70b-GGUF)

Launch llama.cpp server:

```
./server -m ~/models/dolphin-2.9-llama3-70b.Q4_K_M.gguf -ngl 99 -sm row --host 0.0.0.0 -c 8192 --log-format text
```

| :exclamation: Note that you cannot use -fa as this results in all the logits being `null` and its strongly discouraged to launch with any kind of parallelism because this both reduces available context size and seems to break the KV caching so performance suffers.  |
|-----------------------------------------|

## KoboldCpp

The [KoboldCPP API](https://github.com/LostRuins/koboldcpp/issues/469) does not support the `logprobs` feature at this time and it doesn't appear to be on the development roadmap.

## Ollama

The [Ollama API](https://github.com/ollama/ollama/issues/2415) does not support the `logprobs` feature, although a PR is available so this may be supported in the future.

# Cloud Services

## Groq

The [Groq API](https://console.groq.com/docs/openai) does not support the `logprobs` feature at this time.