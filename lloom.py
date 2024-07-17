import streamlit as st
import hashlib
import time
from sys import argv

from utils.viz import visualize_common_prefixes
from utils.search import parallel_lloom_search
from utils.config import load_config

@st.cache_resource
def load_llms():
    config_file = 'config.json' if len(argv) < 2 else argv[1]
    return load_config(config_file)['llms']

STARTING_STORIES = [
    "Once upon a time,",
    "The forest seemed darker then usual, but that did not bother Elis in the least.",
    "In the age before man,"
]

def computeMD5hash(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()

def main():
    st.set_page_config(layout='wide', page_title='The LLooM')
    st.markdown("""
            <style>
                .block-container {
                        padding-top: 2rem;
                        padding-bottom: 0rem;
                        padding-left: 3rem;
                        padding-right: 3.5rem;
                    }
                .row-widget {
                    padding-top: 0rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
    if 'page' not in st.session_state:
        st.session_state.page = 0
        st.session_state.threads = None

    llms = load_llms()
    for llm in llms: print("Loaded LLM:", llm)
        
    logo, config = st.columns((1,5))
    logo.markdown("### The LLooM :green[v0.3]")

    with config.expander('Configuration', expanded=False):
        parallel_requests = st.number_input('Parallel LLM Requests', min_value=1, max_value=16, value=4, help="Raise this value if using vLLM or another batch-capable server")
        
        config_cols = st.columns((1,1))
        config_cols[0].markdown('_Stop conditions_')
        story_depth = config_cols[0].checkbox("Auto-Stop (early terminate if a period or comma is encountered)", value=True)
        depth = config_cols[0].number_input("Maximum Depth", min_value=1, max_value=50, value=12, help="Terminate a sugguestion when it gets this long")
        maxsuggestions = config_cols[0].number_input("Beam Limit", min_value=5, max_value=100, value=25, help="Stop spawning new beams when the number of suggestions hits this limit")
        
        config_cols[1].markdown('_Split conditions_\n\nLower the Cutoff to get more variety (at the expense of quality and speed), raise Cutoff for a smaller number of better suggestions.')
        cutoff = config_cols[1].number_input("Cutoff", help="Minimum propability of a token to have it split a new suggestion beam", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        multiplier = config_cols[1].number_input("Multiplier", help="The cutoff is scaled by Multiplier each time a new token is generated", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
        maxsplits = config_cols[1].number_input("Split Limit", help="The maximum number of splits from a single source token, raise to get more variety.", min_value=0, max_value=10, value=3)
        
    left, right = st.columns((2,3))
    please_wait = right.empty()
    
    if st.session_state.page == 0:
        st.write('Open the Configuration panel above to adjust settings, Auto-depth mode is particularly useful at the expense of longer generation speeds. You will be able to change settings at any point and regenerate suggestions.\n\nThe starting prompts below are just suggestions, once inside the playground you can fully edit the prompt.')
        start_prompt = st.selectbox("Start Prompt", STARTING_STORIES, index=1)
        if st.button("Start"):
            st.session_state.story_so_far = start_prompt
            st.session_state.auto_rounds = 0
            st.session_state.completion = ""
            st.session_state.page = 1
            st.rerun()
    else:
        story_so_far = st.session_state.story_so_far
        new_story_so_far = left.text_area("Story so far", story_so_far, label_visibility='hidden', height=300)

        model_enables = [True] * len(llms)
        model_cols = left.columns(len(llms)+1)
        for llm_idx, llm in enumerate(llms):
            model_enables[llm_idx] = model_cols[llm_idx+1].checkbox(llm["name"], key=f'llm-enable-{llm_idx}', value=True)
        llms_enabled = [x[0] for x in zip(llms, model_enables) if x[1]]
        
        if model_cols[0].button('Suggest Again'):
            story_so_far = new_story_so_far
            st.session_state.story_so_far = story_so_far
            st.session_state.completion = ""
            st.session_state.threads = None
            st.session_state.auto_rounds = 0
        
        if model_cols[0].button('Autopilot x10'):
            story_so_far = new_story_so_far
            st.session_state.story_so_far = story_so_far
            st.session_state.completion = ""
            st.session_state.threads = None
            st.session_state.auto_rounds = 10
                    
        if st.session_state.threads == None:
            t0 = time.time()
            tokens = 0
            
            with please_wait.status('Searching for suggestions, please wait..') as status:
                threads = []
                for thread in parallel_lloom_search(llms_enabled, story_so_far, depth, maxsuggestions, ['.'] if story_depth else [], cutoff, multiplier, maxsplits, parallel_requests):
                    label = thread[1][len(story_so_far):]
                    status.update(label=label, state="running")
                    threads.append(thread)
                    tokens += thread[2]

                delta = time.time() - t0
                tps = tokens/delta
                status.update(label=f"Search completed, found {len(threads)} suggestion in {delta:.2f}s @ {tps:.2f} tokens/sec", state="complete", expanded=False)
            
            sorted_threads = sorted(threads, key=lambda x: x[0], reverse=True)
            
            # remove duplicate threads
            dedupe = {}
            good_threads = []
            add_space = False
            for prob, thread, depth in sorted_threads:
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
            
            # if there is only one option - take it.
            if len(good_threads) == 1 or st.session_state.auto_rounds > 0:
                st.session_state.story_so_far += (" " if add_space else "") + good_threads[0][1]
                st.session_state.threads = None
                if st.session_state.auto_rounds > 0:
                    st.session_state.auto_rounds -= 1
                st.rerun()  
            
        threads = st.session_state.threads
        add_space = st.session_state.add_space
        
        labels = [ thread for prob, thread in threads ]        
        with right:
            control_cols = st.columns((1,1,1))
            level_merge = control_cols[0].number_input('Level Merge', min_value=1, max_value=8, value=2)
            top_to_bottom = control_cols[1].checkbox('Top-to-Bottom', value=False)
            viz = visualize_common_prefixes(labels, level_merge, 'TB' if top_to_bottom else 'LR')
            st.graphviz_chart(viz)
            control_cols[2].download_button('Download DOT Graph', viz.source, 'graph.dot', 'text/plain')
            control_cols[2].download_button('Download PNG', viz.pipe(format='png'), 'graph.png', 'image/png')
            
        controls = st.container()        
        buttons = st.container()
        
        with controls:
            user_add_space = st.checkbox("Prefix space", value=add_space, key=computeMD5hash('prefix-'+story_so_far))
        
        sum_probs = sum([prob for prob, _ in threads])
        with buttons:
            # for prob, thread in threads:
            #     col1, col2 = st.columns((3,1))
            #     col2.progress(value=prob/sum_probs)
            #     new_text = col1.text_input(thread, value=thread, key='text-'+computeMD5hash(thread), label_visibility='hidden')
            #     if col2.button(':arrow_right:', key='ok-'+computeMD5hash(thread)):
            #         st.session_state.story_so_far += (" " if user_add_space else "") + new_text
            #         st.session_state.threads = None
            #         st.rerun()                
            options = {}
            for prob, thread in threads:
                if st.session_state.completion != "" and (not thread.startswith(st.session_state.completion)):
                    continue
                remaining = thread[len(st.session_state.completion):]
                words = remaining.split(' ')
                if len(remaining) == 0:
                    continue
                next_word = words[0]+(' ' if len(words)>1 else '')
                if next_word not in options: options[next_word] = []
                options[next_word].append(' '.join(words[1:]))
                
            for idx, option in enumerate(options.keys()):
                c0, c1, c2 = st.columns((1,2,5))
                if idx == 0:
                    c0.write(st.session_state.completion)
                if c1.button(option.replace('\n','\\n'), key=f'option-{len(st.session_state.completion)}-{idx}'):
                    st.session_state.completion += option
                    st.rerun()
                c2.write(' | '.join(options[option]))
            
            # c0, c1 = st.columns((3,5))
            # write_in = c0.text_input(label='Write-in',value='')
            # if c1.button('Override'):
            #     st.session_state.story_so_far += (" " if add_space else "") + st.session_state.completion + write_in
            #     st.session_state.completion = ""
            #     st.session_state.threads = None                
            #     st.rerun()               
            
            if len(options) == 0:
                st.session_state.story_so_far += (" " if add_space else "") + st.session_state.completion
                st.session_state.completion = ""
                st.session_state.threads = None                
                st.rerun()
            elif len(options) == 1:
                st.session_state.completion += list(options.keys())[0]
                st.rerun()
                

if __name__ == "__main__":
    main()
