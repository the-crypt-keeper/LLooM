from openai import OpenAI
client = OpenAI()
import numpy as np
import streamlit as st
import hashlib

def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()

def get_logprobs(prompt, model="gpt-3.5-turbo"):
    
    messages = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1,
        logprobs=True,
        top_logprobs=10,
        n=1
    )
    return response.choices[0].logprobs.content[0].top_logprobs

def spawn_threads(prompt, depth, cutoff, multiplier, acc = 0.0):
    if depth == 0:
        return [(acc, prompt)]
    
    logprobs = get_logprobs(prompt)
    threads = []
    chosen = []
    
    for logprob_choice in logprobs:
        token = logprob_choice.token
        logprob = logprob_choice.logprob
        
        probability = np.exp(logprob)
        print('CHOICE:', token, probability)
        
        if len(chosen) > 0 and probability < cutoff: break
        chosen.append(token)
        
        new_prompt = prompt + token
        print(f"Spawning new thread: {new_prompt} (prob: {probability:.4f})")
                
        # Increase cutoff for deeper levels and recurse
        new_cutoff = cutoff * multiplier
        sub_threads = spawn_threads(new_prompt, depth - 1, new_cutoff, multiplier, acc+probability)
        threads.extend(sub_threads)
    
    return threads

def main():
    st.set_page_config(layout='centered', page_title='The LLooM')
    st.title("The LLooM")
        
    if 'page' not in st.session_state:
        st.session_state.page = 0
        st.session_state.threads = None
    
    if st.session_state.page == 0:
        config_cols = st.columns((1,1,1))
        depth = config_cols[0].number_input("Depth", min_value=1, max_value=10, value=4)
        cutoff = config_cols[1].number_input("Cutoff", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        multiplier = config_cols[2].number_input("Multiplier", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        
        st.session_state.config = (depth, cutoff, multiplier)
    else:
        (depth, cutoff, multiplier) = st.session_state.config   
        
    if st.session_state.page == 0:
        start_prompt = st.text_input("Start Prompt", "In the dark ages before man,", label_visibility='hidden')    
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

            with st.spinner('Please wait..'):
                threads = spawn_threads(story_so_far, depth, cutoff, multiplier)        
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
                print(prob, thread)
                col1, col2 = st.columns((3,1))
                col2.progress(value=prob/sum_probs)
                new_text = col1.text_input(f'', value=thread, key='text-'+computeMD5hash(thread), label_visibility='hidden')
                if col2.button(':arrow_right:', key='ok-'+computeMD5hash(thread)):
                    st.session_state.story_so_far += (" " if user_add_space else "") + new_text
                    st.session_state.threads = None
                    st.rerun()                

if __name__ == "__main__":
    main()
