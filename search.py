import numpy as np
import os

openai_client = None
def get_logprobs_openai(prompt, model="gpt-3.5-turbo"):
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()
    
    messages = [{'role': 'user', 'content': prompt}]
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        n=1
    )
    
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    for logprob in top_logprobs:
        logprob.probability = np.exp(logprob.logprob)
        
    return top_logprobs

def get_logprobs_llama(prompt, base_url):
    import requests
    
    class LlamaPropability:
        def __init__(self, token, probability):
            self.token = token
            self.probability = probability
   
    url = base_url+'/completion'
    payload = { 'prompt': prompt,
            'cache_prompt': True,
            'temperature': 1.0,
            'n_predict': 1,
            'top_k': 10,
            'top_p': 1.0,
            'n_probs': 10
           }
    
    response = requests.post(url, json=payload)
    probs = response.json()['completion_probabilities'][0]['probs']
    print(probs)

    return [ LlamaPropability(prob['tok_str'], prob['prob']) for prob in probs]

from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_get_logprobs(prompt, acc):
    # Choose which API to use based on environment variables
    if os.getenv('LLAMA_API_URL') is not None:
        api_function = "llama"
    elif os.getenv('OPENAI_API_KEY') is not None:
        api_function = "openai"
    else:
        raise Exception('Please set either OPENAI_API_KEY or LLAMA_API_URL')

    if api_function == "llama":
        logprobs =  get_logprobs_llama(prompt, os.getenv('LLAMA_API_URL'))
    elif api_function == "openai":
        logprobs = get_logprobs_openai(prompt)
    
    return (prompt, acc, logprobs)

def parallel_lloom_search(initial_prompt, max_depth, max_beams, stop_tokens, initial_cutoff, multiplier, maxsplits, parallelism=2):
    
    tasks = [(initial_prompt, 0.0)]
    cutoff = initial_cutoff
    depth = max_depth
    done_beams = 0

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        while tasks:
            # spawn futures
            futures = []
            for task in tasks:
                print("spawning depth:", depth ,"task:", task)
                futures.append(executor.submit(parallel_get_logprobs, *task))
                
            total_futures = len(tasks)
            tasks = []
            done_futures = 0

            # process futures as they come in
            for future in as_completed(futures):
                res = future.result()
                (prompt, acc, logprobs) = res

                count = 0
                for logprob_choice in logprobs:
                    token = logprob_choice.token
                    probability = logprob_choice.probability

                    if count > 0 and probability < cutoff: break        
                    if maxsplits > 0 and count == maxsplits: break

                    count += 1

                    new_prompt = prompt + token
                    early_finish = False

                    if depth == 0 or ((max_beams > 0) and (done_beams+total_futures-done_futures >= max_beams)):
                        yield (acc + probability, new_prompt)
                        early_finish = True
                    else:
                        new_tokens = new_prompt[len(initial_prompt):]
                        for st in stop_tokens:
                            if (not early_finish) and (st in new_tokens):
                                trimmed_prompt = initial_prompt + new_tokens[:new_tokens.find(st)+1]
                                yield (acc + probability, trimmed_prompt)
                                early_finish = True

                    if not early_finish:
                        new_task =(new_prompt, acc + probability)
                        tasks.append(new_task)
                    else:
                        done_beams += 1
                        
                done_futures += 1
            
            # adjust for next cycle            
            cutoff = cutoff * multiplier
            depth = depth - 1
