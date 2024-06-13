from concurrent.futures import ThreadPoolExecutor, as_completed
from .logits import SimpleProbability

def parallel_get_logprobs(llms, prompt, acc):
    logprobs = []
    for llm in llms: logprobs += llm['get_logprobs'](prompt, llm)

    merged_logprobs = {}
    for prob in logprobs:
        if merged_logprobs.get(prob.token) is None:
            merged_logprobs[prob.token] = 0
        merged_logprobs[prob.token] += prob.probability
        
    new_logprobs = [SimpleProbability(k,v) for k,v in merged_logprobs.items()]
    return (prompt, acc, new_logprobs)

def parallel_lloom_search(llms, initial_prompt, max_depth, max_beams, stop_tokens, initial_cutoff, multiplier, maxsplits, parallelism=2):    
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
                futures.append(executor.submit(parallel_get_logprobs, llms, *task))
                
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
                        yield (acc + probability, new_prompt, max_depth - depth)
                        early_finish = True
                    else:
                        new_tokens = new_prompt[len(initial_prompt):]
                        stop_search_tokens = new_tokens
                        
                        for st in stop_tokens:
                            # starting with a stop token is OK, keep searching until there's some meat
                            if stop_search_tokens[0:len(st)] == st: 
                                stop_search_tokens = stop_search_tokens[len(st):]

                            if (not early_finish) and (st in stop_search_tokens):
                                trimmed_prompt = initial_prompt + new_tokens[:new_tokens.find(st)+1]
                                yield (acc + probability, trimmed_prompt, max_depth - depth)
                                early_finish = True

                    if not early_finish:
                        new_task = (new_prompt, acc + probability)
                        tasks.append(new_task)
                    else:
                        done_beams += 1
                        
                done_futures += 1
            
            # adjust for next cycle            
            cutoff = cutoff * multiplier
            depth = depth - 1
