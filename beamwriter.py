import requests
import json
import sys
import time
from termcolor import colored
from utils.config import load_config

def build_prompt(llm, user_prompt, completion):
    if llm['tokenizer'] is None:
        return user_prompt+completion
    else:
        messages = []
        if not llm['no_system_prompt']: messages.append({'role': 'system', 'content': llm['system_prompt']})
        messages.append({'role': 'user', 'content': user_prompt })
        messages.append({'role': 'assistant', 'content': completion })
        return llm['tokenizer'](messages, tokenize=False)
        
model_name = None
UPDATE_RATE = 1.0

def stream_response(llm, user_input, prompt, n = 8):
    global model_name
    if model_name is None:
        model_name = requests.get(llm['api_url']+'/v1/models').json()['data'][0]['id']     

    data = {
        "model": model_name,
        "n": n,
        "prompt": build_prompt(llm, user_input, prompt),
        "top_p": 0.95,
        "max_tokens": 64,
        "stream": True,
        "stop": [".","\n"]
    }
    completions = [''] * n
    done = [False] * n
    tokens = 0
    last_update = time.time()
    
    try:
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
                            if json_data['choices'][0]['finish_reason']:
                                done[json_data['choices'][0]['index']] = True
                                if json_data['choices'][0]['finish_reason'] == 'stop' and llm['engine'] == 'openai':
                                    completions[json_data['choices'][0]['index']] += data['stop'][0]
                                if all(done): break
                        elif 'event' in json_data:
                            if json_data['event'] == 'stream':
                                tokens += 1
                                completions[json_data['index']] += json_data['text']
                            if json_data['event'] == 'stop':
                                done[json_data['index']] = True
                            if json_data['event'] == 'done':
                                break
                        else:
                            print(f"Error: {json_data}")
                        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        
    return completions, tokens

def main():   
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    llm = config[0]

    user_input = input(llm['user_input'])
    if llm.get('tokenizer') is None:
        story = [user_input]
        user_input = ''
    else:
        story = [llm['assistant_seed']] if llm.get('assistant_seed') is not None else []
    sticky_options = []
    generate = True
    speed = None
    
    while True:
        story_text = ''.join(story)
        
        if generate:
            t0 = time.time()
            options, tokens = stream_response(llm, user_input, story_text)
            options += sticky_options
            t1 = time.time()
            delta = t1 - t0
            speed = tokens/delta
            generate = False            

        display_options(sticky_options, story_text, options, speed)
        
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
                generate = True
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
        json.dump(story, f)

def display_options(sticky_options, story_text, options, speed=None):
    print("\033c\033[3J", end='') #clear screen
    print(colored(story_text,'green'))
    print()
    if speed is not None:
        print(colored(f"Generation speed {speed:.2f} tok/sec", "red"))
    for idx, option in enumerate(options):
        color ='blue' if option in sticky_options else 'red'
        print(f"Option {colored(str(idx + 1),color)}: {option.strip()}")
    print()
        
if __name__ == "__main__":
    main()