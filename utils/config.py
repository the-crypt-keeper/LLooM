from .logits import *
from transformers import AutoTokenizer
import json
from copy import copy

def internal_apply_chat_template(fn):
    def wrapper(messages, tokenize=False, add_generation_prompt=True):
        system = [x['content'] for x in messages if x['role'] == 'system']
        user = [x['content'] for x in messages if x['role'] == 'user']
        assistant = [x['content'] for x in messages if x['role'] == 'assistant']
        
        system = "You are a helpful assistant." if len(system) == 0 else system[0]
        user = user[0]
        assistant = "" if len(assistant) == 0 else assistant[0]
        
        return fn(system, user, assistant)
    return wrapper

tokenizer_internal = {
    'internal:vicuna': internal_apply_chat_template(lambda system, user, assistant:
f"""System: {system}

USER: {user}

ASSISTANT:{assistant}"""),

    'internal:alpaca': internal_apply_chat_template(lambda system, user, assistant:
f"""### Instruction:
{system}

### Input:
{user}

### Response:{assistant}""")
}

def load_config(config_file):
    with open(config_file) as f: config = json.load(f)
    llms = []
    for llm_name, llm_info in config.items():
        llm = copy(llm_info)
        llm['name'] = llm_name
        if not 'no_system_prompt' in llm: llm['no_system_prompt'] = False
        if not 'logprobs' in llm: llm['logprobs'] = 10
        
        llm['get_logprobs'] = get_logprobs_openai if llm_info["engine"] == 'openai' else get_logprobs_llama if llm_info["engine"] == 'llamacpp' else None       
        if llm['get_logprobs'] is None: raise Exception(f'Invalid engine {llm_info["engine"]} for {llm_name}, must be one of: openai, llamacpp')        
        if llm['api_url'] is None: raise Exception(f'Please specify api_url for {llm_name}')
        
        llm['tokenizer'] = tokenizer_internal[llm_info["tokenizer"]] if llm_info["tokenizer"] in tokenizer_internal else AutoTokenizer.from_pretrained(llm_info["tokenizer"], trust_remote_code=True).apply_chat_template if llm_info.get("tokenizer") else None                
        llms.append(llm)
        print(f'Loaded {llm_name} via {llm_info["engine"]} @ {llm["api_url"]} with tokenizer {llm_info.get("tokenizer")}')
    return llms
