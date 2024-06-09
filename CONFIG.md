# Configuration File Syntax

The following LLM API providers have been tested to work as logit sources:

* llama.cpp server (GGUF)
* vLLM (FP16/GPTQ/AWQ)
* TabbyAPI (GPTQ/EXL2)
* OpenAI (text completion models only - not recommended)

Any openai-compatible provider that supports text completions with the `logprobs` field should also work.

## Structure
The configuration file is a JSON object where each key represents an LLM endpoint identifier, and the value is another JSON object containing the endpoint's settings.

```json
{
    "endpoint_identifier": {
        "api_url": "string",
        "tokenizer": "string",
        "engine": "string",
        "model": "string",
        "api_key": "string",
        "no_system_prompt": boolean,
        "logprobs": integer
    }
}
```

## Fields

- **endpoint_identifier**: A unique identifier for the LLM endpoint. This can be any string and is used to distinguish between different endpoints.

- **api_url** (optional, string): Specifies the URL where the LLM API is hosted. This is the endpoint's main access point. Do not include /v1.
  
- **tokenizer** (optional, string): The HF tokenizer to use with this endpoint, or one of `internal:vicuna`, `internal:alpaca`. `null` will directly pass through User Prompt.

- **engine** (optional, string): The engine or backend system used for executing requests to the endpoint: `"llamacpp"` or `"openai"`.

- **model** (optional, string): Specifies the model name if a particular model needs to be selected or specified directly.  This is optional for engines that support only a single model.

- **api_key** (optional, string): The API key required for authentication, necessary for services like OpenAI but also useful for TabbyAPI with authentication.

- **no_system_prompt** (optional, boolean): A flag indicating whether to omit the system prompt. Default is `false` if not specified.  This is required by Mixtral and other models that have chat templates without System Prompt support.

- **logprobs** (optional, integer): Specifies the number of log probabilities to return, default is 10 but for vLLM in it's default server configuration it can't go past 5.

## Example

Below is an example configuration file with multiple endpoints defined:

```json
{
    // Llama3 with llama.cpp
    "dolphin2p9_70b": {
        "api_url": "http://10.0.0.169:8080",
        "tokenizer": "cognitivecomputations/dolphin-2.9.1-llama-3-70b",
        "engine": "llamacpp"
    },
    // Mixtral with vLLM
    "mixtral_8x7b": {
        "api_url": "http://10.0.0.199:8000",
        "tokenizer": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "no_system_prompt": true,
        "logprobs": 5,
        "engine": "openai"
    },
    // Mistral with TabbyAPI
    "mistral_7b_exl2": {
        "api_url": "http://10.0.0.199:5000",
        "tokenizer": "mistralai/Mistral-7B-Instruct-v0.2",
        "no_system_prompt": true,
        "engine": "openai"
    },
    // OpenAI
    "gpt-3.5-turbo-instruct": {
        "engine": "openai",
        "model": "gpt-3.5-turbo-instruct",
        "api_key": "sk-XXX"
    }
}
```

### Note

- Provide only the server base url, no trailing / or /v1
- Fields `tokenizer`, `model`, `api_key`, `no_system_prompt`, `logprobs`, `api_key` are optional.
- The default vLLM server configuration will only support up to `logprobs=5`
- Mistral models require `no_system_prompt`