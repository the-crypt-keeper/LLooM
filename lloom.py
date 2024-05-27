import numpy as np
import streamlit as st
import hashlib
import requests
import os

STARTING_STORIES = [
    "Once upon a time",
    "The forest seemed darker then usual, but that did not bother Elis in the least.",
    "In the age before man,"
]

def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()

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

def spawn_threads(prompt, depth, cutoff, multiplier, acc = 0.0):

    if os.getenv('LLAMA_API_URL') is not None:
        logprobs = get_logprobs_llama(prompt, os.getenv('LLAMA_API_URL'))
    elif os.getenv('OPENAI_API_KEY') is not None:
        logprobs = get_logprobs_openai(prompt)
    else:
        raise Exception('Please set either OPENAI_API_KEY or LLAMA_API_URL')

    chosen = []
    
    for logprob_choice in logprobs:
        token = logprob_choice.token
        probability = logprob_choice.probability
        print('CHOICE:', token, probability)
        
        if len(chosen) > 0 and probability < cutoff: break
        chosen.append(token)
        
        new_prompt = prompt + token
        print(f"Spawning new thread: {new_prompt} (prob: {probability:.4f})")
                
        # Increase cutoff for deeper levels and recurse
        new_cutoff = cutoff * multiplier
        if depth > 0:
            for final_thread in spawn_threads(new_prompt, depth - 1, new_cutoff, multiplier, acc+probability):
                yield final_thread
        else:
            yield (acc+probability, new_prompt)

def main():
    st.set_page_config(layout='centered', page_title='The LLooM')
    st.title("The LLooM")
        
    if 'page' not in st.session_state:
        st.session_state.page = 0
        st.session_state.threads = None
    
    if st.session_state.page == 0:
        config_cols = st.columns((1,1,1))
        depth = config_cols[0].number_input("Depth", min_value=1, max_value=10, value=6)
        cutoff = config_cols[1].number_input("Cutoff", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        multiplier = config_cols[2].number_input("Multiplier", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        st.session_state.config = (depth, cutoff, multiplier)
    else:
        (depth, cutoff, multiplier) = st.session_state.config   
        
    if st.session_state.page == 0:
        start_prompt = st.text_input("Start Prompt", "In the age before man,", label_visibility='hidden')    
        if st.button("Start"):
            st.session_state.story_so_far = start_prompt
            st.session_state.page = 1
            st.rerun()
    else:
        story_so_far = st.session_state.story_so_far
        
        new_story_so_far = st.text_area("Story so far", story_so_far, label_visibility='hidden', height=300)
        if st.button('Suggest Again'):
            story_so_far = new_story_so_far
            st.session_state.story_so_far = story_so_far
            st.session_state.threads = None
        
        if st.session_state.threads == None:

            with st.status('Please wait..') as status:
                threads = []
                for thread in spawn_threads(story_so_far, depth, cutoff, multiplier):
                    label = thread[1][len(story_so_far):]
                    status.update(label=label, state="running")
                    threads.append(thread)
                status.update(label="Threading complete.", state="complete", expanded=False)

            sorted_threads = sorted(threads, key=lambda x: x[0], reverse=True)
            
            # remove duplicate threads
            dedupe = {}
            good_threads = []
            add_space = False
            for prob, thread in sorted_threads:
                new_tokens = thread[len(story_so_far):]
                if new_tokens[0] == ' ': 
                    new_tokens = new_tokens[1:]
                    thread = story_so_far + " " + thread[len(story_so_far):]
                    add_space = True
                if dedupe.get(new_tokens) is None:
                    dedupe[new_tokens] = prob
                    good_threads.append( (prob, new_tokens) )

            st.session_state.threads = good_threads
            st.session_state.add_space = add_space
            
        threads = st.session_state.threads
        add_space = st.session_state.add_space

        controls = st.container()        
        buttons = st.container()
        
        with controls:
            user_add_space = st.checkbox("Prefix space", value=add_space, key=computeMD5hash('prefix-'+story_so_far))
        
        sum_probs = sum([prob for prob, _ in threads])
        with buttons:
            for prob, thread in threads:
                col1, col2 = st.columns((3,1))
                col2.progress(value=prob/sum_probs)
                new_text = col1.text_input(thread, value=thread, key='text-'+computeMD5hash(thread), label_visibility='hidden')
                if col2.button(':arrow_right:', key='ok-'+computeMD5hash(thread)):
                    st.session_state.story_so_far += (" " if user_add_space else "") + new_text
                    st.session_state.threads = None
                    st.rerun()                

if __name__ == "__main__":
    main()
