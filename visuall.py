from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from sys import argv
import pandas as pd
import plotly.express as px
from utils.config import load_config

if 'results' not in st.session_state: st.session_state.results = []

@st.cache_resource
def load_llms():
    config_file = 'config.json' if len(argv) < 2 else argv[1]
    return load_config(config_file)['llms']

st.set_page_config(layout='wide', page_title='VisuaLL')
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
st.title("VisuaLL - Logit Inspector/Debugger")
llms = load_llms()

COLORS = ["red","green","blue","orange","purple","cyan"]
with st.expander("Inference Configuration", expanded=True):    
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant that always answers the user's query.")
    maxtokens = st.number_input("Tokens to Generate", min_value=1, max_value=2048, value=128)

user_prompt = st.text_area("User Prompt", value="Create an outline for a series of books about a young woman who discovers her past was hidden from her as she's pulled into a magical world of mystery, deception and hidden secrets.")

mecol, cmcol = st.columns((2,1))

model_enables = [True] * len(llms)
mecol.write('Model Enables')
model_cols = mecol.columns(len(llms))
for llm_idx, llm in enumerate(llms):
    model_enables[llm_idx] = model_cols[llm_idx].checkbox(f':{COLORS[llm_idx]}[{llm["name"]}]', key=f'llm-enable-{llm_idx}', value=True)

completion_mode = cmcol.radio('Multi-LLM Mode', options=['Round Robin','Democratic','Reference'], horizontal=True)
if completion_mode == 'Reference':
    master_llm = cmcol.selectbox('Reference LLM', options=[x['name'] for x in llms])
else:
    master_llm = None
if completion_mode == 'Round Robin':
    num_tokens_per_model = cmcol.number_input("Number of Tokens per Model", min_value=1, max_value=10, value=1)
else:
    num_tokens_per_model = 1

def results_to_markdown(results, bold_idx = None):
    completion_md = ''
    count = 0
    for _, llm_idx, response1_text in results:
        if response1_text.strip() == "":
            completion_md += response1_text
        else:
            color = COLORS[llm_idx] if llm_idx is not None else 'black'
            left_stripped = response1_text.lstrip()
            right_stripped = response1_text.rstrip()
            colored_bit = response1_text.strip()
            
            if left_stripped != response1_text:
                left_ws = len(response1_text) - len(left_stripped)
                completion_md += response1_text[0:left_ws].replace('\n',"<br>")
            if colored_bit != '':            
                #completion_md += f':{color}[{colored_bit}]'
                if bold_idx is not None and count == bold_idx:
                    completion_md += f'<b>{colored_bit}</b>'
                else:
                    completion_md += f'<span style="color:{color};">{colored_bit}</span>'
            if right_stripped != response1_text:
                right_ws = len(response1_text) - len(right_stripped)
                completion_md += response1_text[-right_ws:].replace('\n',"<br>")
        count += 1
    return completion_md

placeholder = st.empty()

run_double_inference_button = st.button("Run Inference", type='primary')
if run_double_inference_button:
    # Compute initial_message for each LLM
    for llm in llms: 
        if llm['tokenizer'] is None:
            llm['initial_message'] = user_prompt
        else:
            messages = []
            if not llm['no_system_prompt']: messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': user_prompt })
            print(llm.get('no_system_prompt'))
            llm['initial_message'] = llm['tokenizer'](messages, tokenize=False, add_generation_prompt=True)
    
    completion = ""
    completion_md = ""
    running = True
    token_count = 0
    null_rounds = 0
    results = []
    allprobs = []
    tokens_this_model = 0
    
    if sum(model_enables) == 0:
        st.error('Please select at least one model')
        exit(0)
        
    def next_active_idx(active_idx, model_enables):
        while True:
            active_idx += 1
            if active_idx >= len(llms): active_idx = 0
            if model_enables[active_idx]: break
        return active_idx

    active_idx = next_active_idx(0, model_enables)
    
    def get_logprobs_parallel(fn, prompt, llm):
        return llm, fn(prompt, llm)
       
    while running:
        all_llm_probs = []
        futures = []            
        token_count = len(results)
        token_idx = 0
        
        # Make calls
        with ThreadPoolExecutor(max_workers=3) as executor:
            for llm_idx, llm in enumerate(llms):
                if not model_enables[llm_idx]: 
                    all_llm_probs.append((llm_idx,[],0))
                else:
                    futures.append(executor.submit(lambda prompt, llm: (llm, llm['get_logprobs'](prompt, llm)), llm['initial_message'] + completion, llm))

        for future in as_completed(futures):
            llm, logprobs = future.result()
            llm_idx = llms.index(llm)
            psum = 0
            for lp in logprobs:
                psum += lp.probability
                if lp.probability > 0:
                    print(f'{token_count} {llm["name"]} #{token_idx}: {lp.token} {len(lp.token)} {lp.probability}')
                    
            all_llm_probs.append((llm_idx,logprobs,psum))
            
            if len(logprobs) == 0: st.error(f"No logprobs by {llm['name']} at position {token_count}")
            elif logprobs[0].token == '': st.error(f"Empty token suggested by {llm['name']} at position {token_count}")
            if psum < 0.5: st.error(f"Low cumulative probability by {llm['name']} at position {token_count}")

        # which one to consider?
        if completion_mode == 'Round Robin':
            logprobs = None
            for llm_idx, lp, _ in all_llm_probs:
                if llm_idx == active_idx: logprobs = lp
            if len(logprobs) != 0:
                response1_text = logprobs[0].token
                results.append( (all_llm_probs, active_idx, response1_text) )

            tokens_this_model += 1
            if tokens_this_model == num_tokens_per_model:
                active_idx = next_active_idx(active_idx, model_enables)
                tokens_this_model = 0
        elif completion_mode == 'Reference':
            logprobs = None
            for master_llm_idx, llm in enumerate(llms):
                if llm['name'] == master_llm:
                    active_idx = master_llm_idx
                    for llm_idx, lp, _ in all_llm_probs:
                        if llm_idx == master_llm_idx: 
                            logprobs = lp
                            response1_text = logprobs[0].token
                            results.append( (all_llm_probs, active_idx, response1_text) )
                    break
                
            for llm_idx, lp, _ in all_llm_probs:
                if len(lp) == 0: continue
                if lp[0].token != response1_text:
                    st.error(f'Position {token_count} reference mismatch {llms[llm_idx]["name"]} {lp[0].token} != {response1_text}')
                    
        elif completion_mode == 'Democratic':
            summed_probs = defaultdict(lambda: 0)
            for llm_idx, logprobs, psum in all_llm_probs:
                for lp in logprobs:
                    summed_probs[lp.token] += lp.probability
            max_prob = 0
            max_token = ''
            for token, prob in summed_probs.items():
                if prob > max_prob:
                    max_token = token
                    max_prob = prob
            active_idx = None
            active_prob = 0
            for llm_idx, logprobs, psum in all_llm_probs:
                if len(logprobs) > 0 and logprobs[0].token == max_token:
                    if logprobs[0].probability > active_prob:
                        active_idx = llm_idx
                        active_prob = logprobs[0].probability

            response1_text = max_token
            results.append( (all_llm_probs, active_idx, response1_text) )
            
        completion += response1_text
        placeholder.markdown(results_to_markdown(results), unsafe_allow_html=True)            
        
        # Stop conditions    
        STOPS = ['<|endoftext|>','<|eot_id|>','<|im_end|>']
        for stop in STOPS:
            if stop in response1_text: running = False           
        if len(results) > maxtokens: running = False

    st.success('Inference complete.')    
    st.session_state.results = results
    
if len(st.session_state.results) > 0:
    data_top = {}
    data_sum = {}
    
    results = st.session_state.results
    for token_idx, (result, _, _) in enumerate(results):
        for llm_idx, logprobs, psum in result:
            llm_name = llms[llm_idx]['name']
            if llm_name not in data_sum: data_sum[llm_name] = []
            if llm_name not in data_top: data_top[llm_name] = []
            
            data_sum[llm_name].append(0 if len(logprobs) == 0 else psum)
            data_top[llm_name].append(0 if len(logprobs) == 0 else logprobs[0].probability)  
            
    df = pd.DataFrame(data_sum)
    fig1 = px.line(df, title="Cumulative Probability", color_discrete_sequence=px.colors.qualitative.G10, height=250)
    df2 = pd.DataFrame(data_top)
    fig2 = px.line(df2, title="Top Probability", color_discrete_sequence=px.colors.qualitative.G10, height=250)
    
    sl, sr = st.columns((1,1))
    sl.plotly_chart(fig1, use_container_width=True)
    sl.plotly_chart(fig2, use_container_width=True)
    logit_index = sr.number_input('Token Index', min_value=0, max_value=len(results), value=0)
    for llm_name, probs in data_top.items():
        determinism = sum(probs)/len(probs)
        creativity = 1-determinism
        if sum(probs) != 0:
            sl.write(f"{llm_name} determinism {determinism:.4f} creativity {creativity:.4f}")
            
    llm_result_cols = sr.columns(len(llms))
    all_llm_probs, active_idx, response1_text = results[logit_index]
    for llm_idx, logprobs, psum in all_llm_probs:
        color = COLORS[llm_idx]
        llm_result_cols[llm_idx].markdown(f'<b><span style="color:{color};">{llms[llm_idx]["name"]}</span></b>', unsafe_allow_html=True)        
        for lp in logprobs:
            token_str = f"```{lp.token}```" if response1_text != lp.token else f'<b>{lp.token}</b>'
            token_str = token_str.replace('\n','\\n')
            llm_result_cols[llm_idx].markdown(f"{token_str} {lp.probability:.3f}", unsafe_allow_html=True)

    placeholder.markdown(results_to_markdown(results, logit_index), unsafe_allow_html=True)