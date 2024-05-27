# The LLooM

Weave the threads of propability.

## Usage with llama.cpp

Launch a llama.cpp server with a good Llama3-70B finetune:

```
./server -m ~/models/dolphin-2.9-llama3-70b.Q4_K_M.gguf -ngl 99 -sm row --host 0.0.0.0 -np 2 -c 8192 --log-format text
```

Then launch the frontend with `LLAMA_API_URL` set to the host and port of the server:

```
LLAMA_API_URL=http://127.0.0.1:8080 streamlit run loom.py
```

## Usage with OpenAI

Launch the frontend with `OPENAI_API_KEY`

```
OPENAI_API_KEY=sk-... streamlit run loom.py
```