# The LLooM

Weaving the threads of propability one token at a time.

Sorta like beamsearching, but with a human in the loop.

# Launching

## Usage with llama.cpp

First, launch a llama.cpp server with a good Llama3-70B finetune:

```
./server -m ~/models/dolphin-2.9-llama3-70b.Q4_K_M.gguf -ngl 99 -sm row --host 0.0.0.0 -np 2 -c 8192 --log-format text
```

Note that you cannot use -fa as this results in all the logits being `null`.

Then launch the frontend with `LLAMA_API_URL` set to the host and port of the server:

```
LLAMA_API_URL=http://127.0.0.1:8080 streamlit run loom.py
```

## Usage with OpenAI

Launch the frontend with `OPENAI_API_KEY`

```
OPENAI_API_KEY=sk-... streamlit run loom.py
```

# Parameters

`Depth` How many tokens to generate per suggestion beam.

`Cutoff` The minimum token propability (0.0 - 1.0) to spawn a new thread.

`Multiplier` cutoff per token slope (1.0: fixed cutoff, <1.0 cutoff decreases with depth, >1.0 cutoff increases with depth)

# Using

Give the LLooM a starting prompt, or change the Story any time.

Note that you can edit the suggestions in-line before accepting them.