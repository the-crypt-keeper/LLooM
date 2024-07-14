import requests
import json
import sys
import time
from termcolor import colored
from utils.config import load_config, build_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
      
UPDATE_RATE = 1.0

def stream_response(llm, user_input, prompt, n = 8):
    data = {
        "model": llm['model'],
        "n": n,
        "prompt": build_prompt(llm, user_input, prompt),
        "top_p": 0.95,
        "max_tokens": 64,
        "stream": True,
        "stop": ["."]
    }
    completions = [''] * n
    done = [False] * n
    tokens = 0
    last_update = time.time()
    
    try:
        req_start_time = time.time()
        first_token_time = None
        
        with requests.post(llm['api_url']+"/v1/completions", headers={"Content-Type": "application/json"}, json=data, stream=True) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if time.time() - last_update > UPDATE_RATE:
                    last_update = time.time()
                    display_options([], prompt, completions)

                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        json_data = json.loads(decoded_line[6:])
                        if 'choices' in json_data:
                            if done[json_data['choices'][0]['index']]: continue
                            
                            completions[json_data['choices'][0]['index']] += json_data['choices'][0]['text']
                            tokens += 1
                            if first_token_time is None: first_token_time = time.time()
                            
                            if json_data['choices'][0]['finish_reason']:
                                done[json_data['choices'][0]['index']] = True
                                if json_data['choices'][0]['finish_reason'] == 'stop' and json_data['choices'][0]['stop_reason'] is not None:
                                    completions[json_data['choices'][0]['index']] += json_data['choices'][0]['stop_reason']
                                # if json_data['choices'][0]['stop_reason'] is None:
                                #     print(json_data['choices'][0])
                                if all(done): break
                        elif 'event' in json_data:
                            # print(json_data)
                            if json_data['event'] == 'stream':
                                tokens += 1
                                if first_token_time is None: first_token_time = time.time()
                                                            
                                completions[json_data['index']] += json_data['text']
                            if json_data['event'] == 'stop':
                                done[json_data['index']] = True
                            if json_data['event'] == 'done':
                                break
                        else:
                            print(f"Error: {json_data}")
                        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        
    ttfs = first_token_time-req_start_time if (first_token_time is not None) and (req_start_time is not None) else None
    elapsed = time.time()-first_token_time if first_token_time is not None else None
        
    return completions, tokens, ttfs, elapsed

def stream_all_llms(llms, user_input, prompt, n=8):
    completions = {}
    tokens = {}
    ttft = {}
    elapsed = {}
   
    with ThreadPoolExecutor(max_workers=len(llms)) as executor:
        future_to_llm = {executor.submit(stream_response, llm, user_input, prompt, llm.get('n',n)): llm['model'] for llm in llms}
        
        for future in as_completed(future_to_llm):
            llm_name = future_to_llm[future]
            try:
                completions[llm_name], tokens[llm_name], ttft[llm_name], elapsed[llm_name] = future.result()
            except Exception as exc:
                print(f'{llm_name} generated an exception: {exc}')
                raise exc
    
    return completions, tokens, ttft, elapsed

def main():   
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.json")    
    llms = config['llms']

    if len(sys.argv) > 2:
        with open(sys.argv[2], 'r') as f:
            last_story = json.load(f)
        if isinstance(last_story, list): last_story = { 'story': last_story, 'user_input': '' }
        user_input = last_story['user_input']
        story = last_story['story']
    else:       
        if config['mode'] == "completion":
            story = [input("Start a story:").replace("\\n",'\n')]
            user_input = ''
        else:
            user_input = input("Instruction:")
            story = [llms[0]['story_seed']] if llms[0].get('story_seed') is not None else []

    sticky_options = []
    generate = True
    speeds = {}
    
    while True:
        story_text = ''.join(story)
        
        if generate:
            all_options, all_tokens, all_ttft, all_elapsed = stream_all_llms(llms, user_input, story_text)
            options = []
            for llm_options in all_options.values():
                options.extend(llm_options)
            options += sticky_options
            for llm_name, tokens in all_tokens.items():
                speeds[llm_name] = (tokens / all_elapsed[llm_name]) if all_elapsed[llm_name] is not None else None
            generate = False

        display_options(sticky_options, story_text, options, speeds, all_ttft)
        
        next_option = input("Picks or [s]ticky [b]ackspace [c]ontinue [w]ritein [d]one: ")
        try:
            if len(next_option) == 0 or next_option[0] == "c":
                generate = True
                continue
            if next_option[0] == "d":
                break
            if next_option[0] == "w":
                writein = input('Write-in:')
                story += [' ' + writein.replace('\\n','\n')]
                continue
            if next_option[0] == "s":
                sticky_option = input("Sticky options:")
                for so in sticky_option.split(','):
                    sticky_options.append(options[int(so) - 1])
                continue
            if next_option[0] == "b":
                options.append(story.pop())
                continue
                
            option_list = next_option.split(',')
            for idx, option in enumerate(option_list):
                try:
                    padding = ''
                    if idx != 0 and len(option_list) > 1: padding = ' '
                    story += [padding+options[int(option) - 1]]
                except:
                    print(f"Invalid option: {option}")
            for idx, option in enumerate(sorted(option_list, reverse=True)):
                if options[int(option) - 1] in sticky_options:
                    sticky_options.remove(options[int(option) - 1])
                del options[int(option) - 1]
        except Exception as e:
            print(f"Error: {e}")
            input('Press ENTER to continue')
                        
    print("\033c\033[3J", end='')
    print('--- Final Story ---')
    print(colored(story_text,'red'))
    with open(f'story-{int(time.time())}.txt','w') as f:
        f.write(story_text)
    with open(f'story-{int(time.time())}.json','w') as f:
        json.dump({ 'user_input': user_input, 'story': story }, f)

def display_options(sticky_options, story_text, options, speeds=None, ttfts=None):
    # print("\033c\033[3J", end='') #clear screen
    print(colored(story_text,'green'))
    print()
    if speeds:
        for llm_name, speed in speeds.items():
            tfft = ttfts.get(llm_name)
            print(colored(f"{llm_name} generation speed {speed:.2f} tok/sec time to first token = {tfft:.2f}s", "red"))
    for idx, option in enumerate(options):
        color ='blue' if option in sticky_options else 'red'
        display_option = option.replace("\n", "\\n")
        print(f"Option {colored(str(idx + 1),color)}: {display_option}")
    print()
        
if __name__ == "__main__":
    main()