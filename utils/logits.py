import numpy as np
import requests

class SimpleProbability:
    def __init__(self, token, probability):
        self.token = token
        self.probability = probability

def get_logprobs_llama(prompt, llm):
    api_url = llm['api_url']
    logprobs = llm['logprobs']    
       
    url = api_url+'/completion'
    payload = { 'prompt': prompt,
            'cache_prompt': True,
            'temperature': 1.0,
            'n_predict': 1,
            'top_k': 10,
            'top_p': 1.0,
            'n_probs': logprobs,
            'samplers': []
           }
    try:
        response = requests.post(url, json=payload)
        probs = response.json()['completion_probabilities'][0]['probs']
    except:
        probs = []
    return [ SimpleProbability(prob['tok_str'], prob['prob']) for prob in probs]

openai_model_names = {}
def get_logprobs_openai(prompt, llm):   
    api_url = llm['api_url']
    logprobs = llm['logprobs']
    model = llm['model']
    
    global openai_model_names
    if model is not None:
        openai_model_name = model
    else:
        openai_model_name = openai_model_names.get(api_url)
        
    if openai_model_name is None:
        models = requests.get(api_url+'/v1/models').json()
        if len(models['data']) == 1:
            openai_model_name = models['data'][0]['id']
            print(f'openai at {api_url} model name {openai_model_name}')
            openai_model_names[api_url] = openai_model_name
        else:
            raise Exception(f'openai at {api_url} model name {openai_model_name} has multiple models, please specify model.')

    url = api_url+'/v1/completions'
    payload = {
        "prompt": prompt,
        "n": 1,
        "temperature": 1.0,
        "max_tokens": 1,
        "stream": False,
        "logprobs": logprobs,
        "model": openai_model_name
    }

    headers = {}
    if llm['api_key'] is not None: headers['Authorization'] = f"Bearer {llm['api_key']}"
    response = requests.post(url, json=payload, headers=headers)
    resp = response.json()
    if 'error' in resp or resp.get('object') == 'error':
        print(resp)
        raise Exception('/v1/completions call failed')

    # print(response.text)
    if resp['choices'][0]['logprobs'] is None:
        return []
    
    probs = resp['choices'][0]['logprobs']['top_logprobs'][0]
    return [ SimpleProbability(k,np.exp(v)) for k,v in probs.items()]