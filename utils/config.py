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

    # global defaults
    if 'mode' not in config: config['mode'] = 'completion'

    llms = config['llms']
    for llm in llms:
        # llm defaults
        llm['tokenizer'] = llm.get('tokenizer', None)
        llm['no_system_prompt'] = llm.get('no_system_prompt', False)
        llm['logprobs'] = llm.get('logprobs', 10)

        llm['get_logprobs'] = get_logprobs_openai if llm["engine"] == 'openai' else get_logprobs_llama if llm["engine"] == 'llamacpp' else None       
        if llm['get_logprobs'] is None: raise Exception(f'Invalid engine {llm["engine"]}, must be one of: openai, llamacpp')        
        if llm['api_url'] is None: raise Exception(f'Missing api_url')
        if llm.get('model') is None: llm['model'] = requests.get(llm['api_url']+'/v1/models').json()['data'][0]['id']

        llm['apply_chat_template'] = tokenizer_internal[llm["tokenizer"]] if llm["tokenizer"] in tokenizer_internal else AutoTokenizer.from_pretrained(llm["tokenizer"], trust_remote_code=True).apply_chat_template if llm.get("tokenizer") else None
        
        print(f'Loaded {llm["model"]} via {llm["engine"]} @ {llm["api_url"]} with tokenizer {llm.get("tokenizer")}')

    return config

def build_prompt(llm, user_prompt, completion):
    if llm['tokenizer'] is None:
        return user_prompt+completion
    else:
        messages = []
        if not llm['no_system_prompt']: messages.append({'role': 'system', 'content': llm['system_prompt']})
        messages.append({'role': 'user', 'content': user_prompt })
        messages.append({'role': 'assistant', 'content': completion })
        return llm['apply_chat_template'](messages, tokenize=False)