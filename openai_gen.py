import openai
import json
import os
from typing import List
import re
import concurrent.futures
from functools import partial
import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from transformers import GPT2Tokenizer
from argparse import ArgumentParser
from collections import Counter
import random
from utils.utils import get_unique_keys
from utils.model_utils import extract_text_answer
from template.perturbation_template import fs_shot,fs_shot_qd
from template.prompt_template import format_llama_prompt,cot_template

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def cross_check_ans(objective,ia,pa):
    redo=False
    if objective == 'para' and ia != pa:
        redo=True
    elif objective == 'add_mistakes' and ia == pa:
        redo=True
    return redo
    


def get_label(text):
    """
    Given a text with "(x),....", get x
    """
    ans_pattern = r'\((.*?)\)'
    ans = re.findall(ans_pattern,text.lower())
    if len(ans) == 0:
        return None
    else:
        return ans[0]

def generate_answer(context_dict,max_tokens,cot_type):
    if not isinstance(context_dict,list):
        context_dict = [context_dict]
    qa_prompts = [format_llama_prompt(c['question'],c['choices'],fs_prompt = cot_template[args.dataset],prompt_type = cot_type,explanation = c['explanation'],num_shot=2) for c in context_dict] 
    outs = send_openai_request(None,qa_prompts,'text-davinci-003',max_tokens=max_tokens)
    ans = [get_label(io) for io in outs]
    return qa_prompts,ans

# for single request instead of messages
def send_openai_request(gen_fn,prompt,engine,max_tokens = 1024,batch_size = 0,t=1.0,p=0.95,freq_penalty = 0.6,initial_in=None):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=t,
        # top_p=p,
        frequency_penalty = freq_penalty # add penalty to avoid repeating
    )
    if isinstance(prompt,list): # batching
        out = [''] * len(prompt)
        for choice in response.choices:
            out[choice.index] = choice.text
    else:
        out = [response['choices'][0]['text']]
    return out

def create_prompt(inputs,objective='para',completion=True,dataset = 'obqa',cot_type = 'cot_sc'):
    """
    Create prompt to do the following
    1) paraphrase a given sentence
    2) add mistakes to an explanation corresponding to the question and answer
    3) change the input question such that the answer is different
    fs_shot few shot examples
    """
    ## Paraphrase
    para_template = 'Please rewrite the following text, conveying exactly the same information but using different wording.\nText: {explanation}'
    para_template_qd = 'Please rewrite the following question and answer, conveying exactly the same information but using different wording.\nQuestion: {subquestion}\nAnswer:{subanswer}' 
    
    # Add mistakes
    fs_input_mistakes = 'Given a reasoning sentence and extracted rationale from the reasoning sentence. Change the rationale such that it results in a erroneous reasoning sentence when substitued in. Finally, generate the erroneous reasoning sentence.\nReasoning: {explanation}\nRationale: {rationale}'
    fs_output_mistakes = 'Erroneous reasoning: {output}'
    
    ## Rationale extraction
    fs_input_rationale = 'Given a question, answer and a reasoning sentence that was used to help to achieve the answer. Please extract a contiguous span of words, denoted as the rationale from the reasoning sentence. This rationale should denote a strong causal relationship with the answer.\nQuestion: {question}\nAnswer: {answer}\nReasoning: {explanation}'
    fs_output_rationale = 'Rationale: {output}'
    
    ## rationle for QD
    fs_input_rationale_qd = "Given a question and an answer. Please extract a contiguous span of words, denoted as the rationale from the answer. This rationale should be important for answering the question.\nQuestion: {question}\nAnswer: {answer}"
    fs_output_raitonale_qd = "Rationale: {output}"
    
    ## Add mistakes for QD
    fs_input_mistakes_qd = 'Given a question, answer and the extracted rationale from the answer. Change the rationale such that it results in a erroneous answer. Finally, generate the erroneous answer.\nQuestion: {question}\nAnswer: {answer}\nRationale: {rationale}'
    fs_output_mistakes_qd = 'Erroneous answer: {output}'

    # Counterfactual
    fs_input_cf = 'Given a question and corresponding answer, please choose the second most likely answer from the answer choices and generate a new question such that the new question will correspond to the second likely answer. You are to make minimal changes to the question.\nQuestion: {question}\nAnswer: {answer}\nAnswer Choices:\n{choices}\n\n Lets do this step-by-step'
    fs_output_cf = 'The second most likely answer is {target_answer}\nCounterfactual question: {output}'
    
    # Counterfactual edit
    fs_input_cf_edit = 'Please highlight the difference in text between the original statement and changed statement. Only highlight the extra or changed words and ignore the missing ones. If there are more than one contiguous text, use a comma to separate them.\nOriginal statement: {question}\nChanged statement: {cf_question}'
    fs_output_cf_edit = 'Original: {original}\nChanged: {changed}'
    
    
    ## Setup templates for both QD and other variants of CoT
    templates = {'para':para_template,'add_mistakes':fs_input_mistakes,'cf':fs_input_cf,'cf_edit':fs_input_cf_edit,'rationale_extraction':fs_input_rationale}
    
    qd_templates = {'para': para_template_qd,'add_mistakes':fs_input_mistakes_qd,'rationale_extraction':fs_input_rationale_qd}
    
    fs_chat_templates = {'add_mistakes':[fs_input_mistakes,fs_output_mistakes],'cf':[fs_input_cf,fs_output_cf],'cf_edit':[fs_input_cf_edit,fs_output_cf_edit],'rationale_extraction':[fs_input_rationale,fs_output_rationale]}
    
    fs_chat_templates_qd = {'add_mistakes':[fs_input_mistakes_qd,fs_output_mistakes_qd],'rationale_extraction':[fs_input_rationale_qd,fs_output_raitonale_qd]}

    # Mistake insertion done in two stages:
    # 1) identify rationales
    # 2) change rationales and generate new sentence
    if objective == 'add_mistakes':
        objective = ['rationale_extraction','add_mistakes']
    else:
        objective = [objective]
    
    all_out,all_fs = {},{}
    for obj in objective:
        ## For question decomp, few shots and templates are slightly different
        if cot_type == 'cot_qd':
            template = qd_templates[obj]
            fs_temp = fs_chat_templates_qd.get(obj,None)
            if obj in ['rationale_extraction','add_mistakes']:
                fs_examples = fs_shot_qd.get(obj,None)
            elif obj == 'cf':
                fs_examples = fs_shot['cf']
            else:
                fs_examples = None
        else:
            template = templates[obj]
            if not completion:
                fs_temp = fs_chat_templates.get(obj,None)
            
            fs_examples = fs_shot.get(obj,None)
        
        out_prompt = []
        fs_separater = '\n\n'
        alpha_mapping = {i:chr(ord('a')+i) for i in range(len(inputs.get('choices',[])))} # for original inputs
        out_fs = []
        if fs_examples is not None:
            if obj == 'cf':
                fs_examples = fs_examples[dataset]
            for fs in fs_examples:
                copy_fs = deepcopy(fs)
                fs_mapping = {i:chr(ord('a')+i) for i in range(len(fs.get('choices',[])))} # for few-shot only, only 1 dataset, choices might be different from original
                if completion:
                    out_prompt.append(template.format_map(SafeDict(copy_fs)))
                else:
                    out_f = {}
                    for i,inp_type in enumerate(fs_temp):
                        req_keys = re.findall(r'{(.*?)}',inp_type)
                        if 'choices' in req_keys:
                            choices = ['(' + fs_mapping[i] + ')' + ' '+ c for i,c in enumerate(fs['choices'])]
                            copy_fs['choices'] = '\n'.join(choices)
                        inp_mapped = inp_type.format_map(SafeDict(copy_fs))
                        if i == 0:
                            out_f['input'] = inp_mapped
                        else:
                            out_f['response'] = inp_mapped
                    out_fs.append(out_f)
        all_fs[obj] = out_fs

        # for inputs
        copy_inputs = deepcopy(inputs)
        if obj not in ['para','add_mistakes']:
            if inputs.get('choices',None) is not None:
                inp_choices = ['(' + alpha_mapping[i] + ')' + ' '+ c for i,c in enumerate(inputs['choices'])]
                copy_inputs['choices'] = '\n'.join(inp_choices)
                chosen_ans = inputs['choices'][inputs['answer']]
                copy_inputs['answer'] = f"({alpha_mapping[inputs['answer']]}) {chosen_ans}"
        else:
            if cot_type != 'cot_qd':
                copy_inputs['answer'] = inputs['choices'][inputs['answer']]               
        if cot_type == 'cot_qd' and obj != 'cf':
            copy_inputs['question'] = copy_inputs['explanation'][0]
            copy_inputs['answer'] = copy_inputs['explanation'][1]
        if obj == 'cf_edit':
            copy_inputs['cf_question'] = copy_inputs['cf']['cf_question']
        input_prompt = template.format_map(SafeDict(copy_inputs)).strip()
        if obj == 'add_mistakes':
            if cot_type == 'cot_qd':
                input_prompt += '\nErroneous answer'
            else:
                input_prompt += '\nErroneous reasoning'
        out_prompt.append(input_prompt)
        if completion:
            out_prompt = fs_separater.join(out_prompt)
        else:
            out_prompt = input_prompt
        
        all_out[obj] = out_prompt
            
    return all_out,all_fs,inputs

# for chat interface
def send_openai_message(prompt,engine,tokenizer,max_tokens = 1024,objective='cf',args=None,freq_penalty = 0.6,cot_type='cot',initial_in=False):
    """
    Given prompt, send request to openai chat interface, single request
    Max timeout of 60s, allow 1 more retry
    Allow max of 3 redos for counterfactual gen (by using check condition in check_cf)
    """
    def openai_call(messages,engine,max_tokens = 1024,freq_penalty = 0.6):
        try:
            response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            temperature=args.temp,
            # top_p = args.top_p,
            # n = args.n,
            max_tokens=max_tokens,
            request_timeout=60, # if hangs
            # frequency_penalty = freq_penalty # add penalty to avoid repeating
            )
        except Exception as e:
            print (f'Error in chat request, error : {e} retrying...')
            time.sleep(2)
            response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            temperature=args.temp,
            # top_p = args.top_p,
            # n = args.n,
            max_tokens=max_tokens,
            request_timeout=60, 
            # frequency_penalty = freq_penalty # add penalty to avoid repeating
            )
        return response
    
    sys_msg = 'You are a helpful assistant who is good at following instructions and has expertise in linguistics.'
    
    if initial_in:
        prompt, initial_inputs = prompt
    
    prompt, fs_prompt,data_dict = prompt[0],prompt[1],prompt[2] # fs_shot
    
    ## This portion is to accomodate the 2-stage generation for add_mistakes
    if 'rationale_extraction' in prompt.keys():
        prompt_order = ['rationale_extraction','add_mistakes']
    else:
        prompt_order = list(prompt.keys())
    
    repeat_out = {}
    repeating_messages = {}
    
    for obj in prompt_order: # usually only 1 key except for add_mistakes
        messages = [
        {"role": "system", "content": sys_msg},
        ]
        obj_fs_prompt = fs_prompt[obj]
        obj_prompt = prompt[obj]

        if len(obj_fs_prompt)> 0: 
            for fs in obj_fs_prompt:
                messages += [{"role":"user","content":fs['input']},
                            {"role":"assistant","content":fs['response']}]
                
        if len(repeat_out) >0 and obj == 'add_mistakes':
            pre_add_rationale = deepcopy(obj_prompt) # for repeating msg, dont add the current rationale, leave for it for later
            obj_prompt = obj_prompt.format_map(SafeDict(repeat_out))
            pre_add_msg = messages + [{"role": "user", "content":pre_add_rationale}]
            repeating_messages[obj] = pre_add_msg
            
        messages += [{"role": "user", "content":obj_prompt}]
        response = openai_call(messages,engine,max_tokens = max_tokens,freq_penalty = freq_penalty)
        out_text = response['choices'][0]['message']['content']
        if obj == 'rationale_extraction':
            if ':' in out_text:
                out_text = out_text.split(':')[1].strip()
            repeat_out = {'rationale':out_text}
            repeating_messages[obj] = messages
        elif obj != 'add_mistakes':
            repeating_messages[obj] = messages
    num_out_tokens = 0
    out = {}
    num_out_tokens += len(tokenizer.encode(out_text,add_special_tokens = False))
    out_text = clean_answer(out_text,mode = objective,cot_type = cot_type,choices = data_dict['choices'])
    
    if objective == 'cf': # only do check for cf
        if len(out_text) > 0:
            check = check_cf(out_text,data_dict)
        else:
            check = False
        redo_count = 0
        while not check:
            if redo_count >0: # only 1 redo allowed
                print (f"Error in generating cf, keep getting either same answer or out of choices answer")
                out_text = {} # outputs an empty list, if this happens, skip the entire generation
                break
            print (f'redoing for cf, iter {redo_count+1}')
            response = openai_call(messages,engine,max_tokens = max_tokens,freq_penalty = freq_penalty)
            out_text = response['choices'][0]['message']['content']
            # print (out_text)
            num_out_tokens += len(tokenizer.encode(out_text,add_special_tokens = False))
            out_text = clean_answer(out_text,mode = objective,cot_type = cot_type,choices = data_dict['choices'])
            redo_count += 1
            if len(out_text) > 1:
                check = check_cf(out_text,data_dict)
        if len(out_text) > 0:
            edited_ques = out_text[-1]
            edited_ans = out_text[0]
            # edit_n_original = out_text[1]
            # if edit_n_original is None: # means did not process properly, so we instead to just compare original and changed text.
            #     original_ques = data_dict['question']
            #     original_text = set(original_ques.split())
            #     changed_text = set(edited_ques.split())
            #     original_ = list(original_text.difference(changed_text)) # set as list, can be split into tokens later on
            #     edit_ = list(changed_text.difference(original_text))
            # else:
            #     original_ = edit_n_original[0]
            #     edit_ = edit_n_original[1]
            
            # Not good to redo the entire thing, so we just redo the last part
            c_data_dict = deepcopy(data_dict)
            c_data_dict['cf'] = {}
            c_data_dict['cf']['cf_question'] = edited_ques
            edit_inp,edit_fs,_ = create_prompt(c_data_dict,objective = 'cf_edit',completion=args.completion,dataset = args.dataset,cot_type = cot_type)
            edit_messages = [{"role": "system", "content": sys_msg},]
            for efs in edit_fs:
                edit_messages += [{"role":"user","content":efs['input']},
                            {"role":"assistant","content":efs['response']}]
            edit_messages += [{"role": "user", "content":edit_inp}]
            
            edit_engine = 'gpt-3.5-turbo' # rmb change it to gpt-3.5, good enough ,save $$$
            edit_response = openai_call(edit_messages,edit_engine,max_tokens = max_tokens,freq_penalty = freq_penalty)
            edit_text = edit_response['choices'][0]['message']['content']
            edit_text = clean_answer(edit_text,mode = 'cf_edit',cot_type = cot_type,choices = data_dict['choices'])
            edited_text = edit_text['changed']
            original_text = edit_text['original']

            out_text = {'cf_question': edited_ques,'cf_answer':edited_ans,'edit':edited_text,'original':original_text}
                        # 'edit':list(diff_set)}
                        
    elif objective in ['para']: # if para or add mistakes, we check according to the conditions.
        if 'add_mistakes' in repeating_messages.keys(): # keep a copy of the base msg without any rationale
            base_msg = deepcopy(repeating_messages['add_mistakes'])
        total_retries = 3
        initial_inputs['explanation'] = out_text
        post_prompts,post_ans = generate_answer(initial_inputs,20,cot_type)
        num_out_tokens += len(tokenizer.encode(post_prompts,add_special_tokens = False))
        initial_ans = initial_inputs['answer']
        redo = cross_check_ans(objective,initial_ans,post_ans[0])
        redo = True
        curr_retry = 0
        while redo:
            if 'add_mistakes' in repeating_messages.keys():
                repeating_messages['add_mistakes'] = base_msg
            for obj in prompt_order:
                redo_msg = repeating_messages[obj]
                if obj == 'add_mistakes':
                    if ':' in out_text:
                        out_text = out_text.split(':')[1].strip()
                    redo_msg[-1]['content'] = redo_msg[-1]['content'].format_map(SafeDict({'rationale':out_text}))
                response = openai_call(redo_msg,engine,max_tokens = max_tokens,freq_penalty = freq_penalty)
                out_text = response['choices'][0]['message']['content']

            out_text = clean_answer(out_text,mode = objective,cot_type = cot_type,choices = data_dict['choices'])
            num_out_tokens += len(tokenizer.encode(out_text,add_special_tokens = False))
            # print (out_text)
            initial_inputs['explanation'] = out_text
            post_prompts,post_ans = generate_answer(initial_inputs,20,cot_type)
            num_out_tokens += len(tokenizer.encode(post_prompts,add_special_tokens = False))
            redo = cross_check_ans(objective,initial_ans,post_ans[0])
            curr_retry += 1
            if curr_retry >= total_retries:
                break

    out['num_out_tokens'] = num_out_tokens
    out['text'] = out_text
        
    return out

def batch_send_requests(prompt_fn,prompt_list,engine,tokenizer,max_tokens=1024,batch_size=5,freq_penalty = 0.6,objective = 'cf',args=None,cot_type='cot',initial_inputs=None):
    """
    Batch send requests using prompt_fn, only for chat interface (send_openai_message)
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        if initial_inputs is not None:
            prompt_list = [(pl,ii) for pl,ii in zip(prompt_list,initial_inputs)]
            initial_in = True
        else:
            initial_in = False
        request_fn = partial(prompt_fn,engine=engine,max_tokens=max_tokens,objective=objective,tokenizer = tokenizer,args=args,cot_type=cot_type,initial_in=initial_in) # fixed variables
        results = list(executor.map(request_fn,prompt_list))
    return results

def majority_element_indexes(lst):
    # Step 1: Find the majority element
    count = Counter(lst)
    majority_element = max(count.keys(), key=count.get)

    # Step 2: Find the indices of the majority element
    indices = [index for index, element in enumerate(lst) if element == majority_element]

    return indices

def clean_answer(text,mode = 'para',cot_type = 'cot_sc',choices = None):
    """
    Clean answer to be used in prompt
    """
    if mode in ['para','add_mistakes']:
        if mode == 'add_mistakes':
            if '\n' in text:
                out = text.split('\n')[0].strip()
            else:
                out = text
            if 'Erroneous reasoning:' in out:
                out = out.split('Erroneous reasoning:')[1].strip()
            elif "Erroneous answer:" in out:
                out = out.split('Erroneous answer:')[1].strip()
            if 'Rationale' in out:
                out = out.split('Rationale')[0].strip()
            elif 'rationale' in out:
                out = out.split('rationale')[0].strip()
        if cot_type == 'cot_qd':
            if mode != 'add_mistakes': # only add mistake to ans
                out = []
                qas = text.split('\n')[:2]
                if len(qas)< 2:
                    print (f'Error in generating paraphrase.\nOutput:{text}')
                    return None
                for qa in qas:
                    if ':' in qa:
                        out.append(qa.split(':')[1].strip())
                    else:
                        out.append(' '.join(qa.split()[1:]).strip())
            
    elif mode == 'cf':
        sep_text = text.split('\n')
        out =[]
        if len(sep_text) < 2: # have to be at least 3 lines
            return []
        for sen_i,t in enumerate(sep_text):
            if len(t.strip()) == 0:
                continue
            if sen_i == 0 or 'second most likely answer' in t:
                num_ans,_  = extract_text_answer(t,choices = choices)
                out.append(num_ans)
            # elif sen_i == 1:
            #     changed_words = re.findall(r'"(.*?)"', t) # list where there should be 2 elements, 1st is original, 2nd is changed
            #     if len(changed_words) != 2:
            #         out.append(None)
            #     else:
            #         out.append(changed_words)
            elif sen_i == 1 or 'counterfactual question' in t:
                if ':' in t:
                    out.append(t.split(':')[1].strip())
                elif 'question' in t:
                    out.append(t.split('question')[1].strip())
                else:
                    out.append(t.strip())
        # if len(out) != 3:
        #     return []
    elif mode == 'cf_edit':
        sep_text = text.split('\n')
        out = {}
        for st in sep_text:
            if st.lower().startswith('original'):
                out['original'] = st.split(':')[1].strip()
            elif st.lower().startswith('changed'):
                out['changed'] = st.split(':')[1].strip()
        
        try:
            if 'original' not in out or 'changed' not in out:
                sep_text = sep_text[:2]
                out['original'] = sep_text[0].split(':')[1].strip()
                out['original'] = sep_text[1].split(':')[1].strip()
        except:
            print (f'Error in generating cf_edit.\nOutput:{text}')
            return {}
    return out   

def check_cf(gen,data_dict):
    """
    This is only used to check cf generation to fufil the 2 conditions:
    1) answer cannot be same as original
    2) answer must be in choices
    """
    ori_ans = data_dict['answer']
    question = data_dict['question']
    changed_question = gen[-1]
    changed_ans = gen[0]
    if changed_ans == ori_ans or changed_ans == -1:
        return False
    if changed_question == question:
        return False
    return True

def save_file(out_path,data):
    """
    Save data to out_path
    """
    with open(out_path,'w', encoding="utf-8") as f:
        if isinstance(data,list):
            for d in data:
                json.dump(d,f)
                f.write('\n')
        else:
            json.dump(data,f)

def append_to_jsonl(data, filename,overwrite = False): # single dumping, safer in case halfway script stops, $$$ fly away
    if overwrite:
        wm = 'w'
    else:
        wm = 'a'
    with open(filename, wm, encoding="utf-8") as f:
        f.write(json.dumps(data))
        f.write('\n')

def cal_cost(model,input_length,output_length):
    if model in ["text-davinci-003","gpt-3.5-turbo"]:
        inp_cost = 0.0015/1000
        out_cost = 0.003/1000
    else:
        inp_cost = 0.03/1000
        out_cost = 0.06/1000
    return inp_cost*input_length + out_cost*output_length

def batchgen_w_retries(batch_gen_fn,gen_fn,prompts,model,tokenizer,max_decode_len = 128,curr_size = 2,objective='cf',args=None,cot_type='cot',initial_inputs=None):
    """
    batch generation with batch_gen_fn that uses multithreading to perform gen_fn.
    Allow single retries
    """
    batch_gen = None
    try:
        batch_gen = batch_gen_fn(gen_fn,prompts,model,tokenizer,max_tokens = max_decode_len,batch_size = curr_size,objective=objective,args= args,cot_type=cot_type,initial_inputs=initial_inputs)
    except Exception as e:
        print (f'Error: {e}')
        print ('Error in batch request, retrying...')
        time.sleep(60) # sleep for 60 seconds to avoid rate limit
        try:
            batch_gen = batch_gen_fn(gen_fn,prompts,model,tokenizer,max_tokens = max_decode_len,batch_size = curr_size,objective=objective,args= args,cot_type=cot_type,initial_inputs=initial_inputs)
        except Exception as e:
            print (f'Error: {e}')
            exit('Error in batch request, exiting...')
    if batch_gen is None:
        exit('Error in batch request, exiting...')
    return batch_gen

def get_explanation(data_dict,cot_type = 'cot_qd'):
    """
    cot_qd returns a list of questions and answers in the format of (ques1,ques2,ans1,ans2)
    cot_cf returns a lit of explanations for all choices
    """
    explanation = data_dict['explanation']
    choices = data_dict['choices']
    answer = data_dict['answer']
    if cot_type == 'cot_qd':
        if len(explanation) %2 != 0 or len(explanation) == 0:
            return None
        else:
            # print (explanation)
            num_qa = len(explanation)//2
            random_id = random.randint(0,num_qa-1)
            q,a = explanation[random_id],explanation[random_id+num_qa]
            q = ' '.join(q.split()[1:]) # throw away the prefix Q1. and A1., etc..
            a = ' '.join(a.split()[1:])
            out = [q,a,random_id]
        
    elif cot_type == 'cot_cf':
        if len(explanation) != len(choices):
            return None
        else:
            out =  explanation[answer]
    return out

def post_process(data_dict,gen_outputs,gen_inputs,objective='cf',cot_type = 'cot_sc'):
    """
    This fn mainly created for cot_qd and cot_sc, others are just regular subbing in to data dict
    data_dict: original data dict
    gen: generated text, either a list or str.
    gen_inputs: the inputs given to generate, for some cot_type and objective, might need to infer from here.
    """
    for i,gen in enumerate(gen_outputs):
        if cot_type not in ['cot_qd','cot_cf'] and objective != 'cf_edit': 
            data_dict[i][objective] = gen
        elif cot_type in ['cot_qd','cot_cf'] and objective != 'cf_edit':
            if cot_type == 'cot_qd':
                chosen_id = gen_inputs[i]['explanation'][-1] # which qa did we choose to change.
                num_qa = len(data_dict[i]['explanation'])//2
                if objective == 'para':
                    data_dict[i][objective] = deepcopy(data_dict[i]['explanation']) # copy the original explanation
                    if gen is None: # meaning theres an error.
                        data_dict[i][objective] = ''
                    else:
                        data_dict[i][objective][chosen_id] = f'Q{chosen_id+1}. {gen[0]}' # sub in the q
                        data_dict[i][objective][chosen_id+num_qa] = f'A{chosen_id+1}. {gen[1]}' # then the a
                elif objective == 'add_mistakes':
                    data_dict[i][objective] = deepcopy(data_dict[i]['explanation'])
                    data_dict[i][objective][chosen_id+num_qa] = f'A{chosen_id+1}. {gen}' # sub in the a only
                else:
                    data_dict[i][objective] = gen
            elif cot_type == 'cot_cf':
                # For cf, since we have explanation for each answer and we only add mistakes to the corresponding expl for answer
                if objective in ['para','add_mistakes']:
                    data_dict[i][objective] = deepcopy(data_dict[i]['explanation'])
                    expl_id = data_dict[i]['answer']
                    data_dict[i][objective][expl_id] = gen
                else:
                    data_dict[i][objective] = gen
        elif objective == 'cf_edit':
            data_dict[i]['cf']['original'] = gen['original']
            data_dict[i]['cf']['edit'] = gen['changed']
                    
    return data_dict

def main(args):
    # Model engine
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # just to cal cost, rough estimate
    # data to generate q&a
    random.seed(args.seed)
    # chat_model = {'para': "gpt-3.5-turbo",'add_mistakes':"gpt-3.5-turbo",'cf':"gpt-3.5-turbo"} # cf use stronger model
    if args.completion:
        gen_fn = send_openai_request
        batch_gen_fn = gen_fn # for completion is same, just outputs a list
        model = 'davinci-003'
    else:
        gen_fn = send_openai_message
        batch_gen_fn = batch_send_requests
        model = 'gpt-3.5-turbo'

    meta_data = {'gpt-3.5-turbo':{'num_input_tokens':0,
                'num_output_tokens':0,
                'total_cost':0},
                'gpt-4':{'num_input_tokens':0,
                'num_output_tokens':0,
                'total_cost':0}}
    
    get_model = {'para':'gpt-3.5-turbo','add_mistakes':'gpt-3.5-turbo','cf':'gpt-4','cf_edit': 'gpt-3.5-turbo'}

    for cot_type in args.cot_type: # Do every cot_type inside, we want the cf to be same for those with similar answer so we can do downstream comparison. Saves cost as well..
        print (f'Generating permutations for {cot_type} for {args.dataset}.')
        print (f'-'*100)
        starting_time = time.time()
        if cot_type != 'cot_cf':
            max_decode_len = 128
        else:
            max_decode_len = 512
        data_dir = f'data/{args.dataset}/{str(args.seed)}'
        if args.model_size == 70:
            if args.num_shot >= 7:
                main_expl_dir = os.path.join(data_dir,f'cot')
            else:
                main_expl_dir = os.path.join(data_dir,f'cot_{args.num_shot}')
        else:
            main_expl_dir = os.path.join('./data',args.dataset,f'cot_{args.model_size}B')
        
        if 'cf' in args.perturbation_types:
            datapath = os.path.join(data_dir,'test.jsonl') # default name for the test file {seed}.jsonl
            meta_data_path = os.path.join(main_expl_dir,f'meta_data.jsonl')
            perturbation_path = os.path.join(data_dir,f'cf_{args.seed}.jsonl')  # for cf, we only do 1 for every template since it is irregardless of the explanation type.
        elif 'cf_edit' in args.perturbation_types:
            datapath = os.path.join(data_dir,f'cf_{args.seed}.jsonl')
            meta_data_path = os.path.join(main_expl_dir,f'meta_data.jsonl')
            perturbation_path = os.path.join(data_dir,f'cf_{args.seed}_edited.jsonl')
        else:
            per_dir = os.path.join(main_expl_dir,cot_type)
            per_dir += args.path_suffix
            datapath = os.path.join(per_dir,f'{args.seed}.jsonl') # default name for the test file {seed}.jsonl
            # Meta data path
            meta_data_path = os.path.join(per_dir,f'meta_data.jsonl')
            if 'add_mistakes' in args.perturbation_types:
                perturbation_path = os.path.join(per_dir,f'pertubated_{args.seed}_mistakes.jsonl') # save path
            else:
                perturbation_path = os.path.join(per_dir,f'pertubated_{args.seed}.jsonl') # save path
        with open(datapath) as f:
            ds = [json.loads(line) for line in f]   

        existing_keys = set()
        
        if os.path.exists(perturbation_path):
            with open(perturbation_path) as f:
                collected = [json.loads(line) for line in f]
                print ('Loading from existing perturbations, total collected: ',len(collected))
                existing_keys = get_unique_keys(collected) # continue from where we left off
                all_keys = get_unique_keys(ds)
                remaining_keys = set(all_keys.keys()) - set(existing_keys.keys())
                if len(remaining_keys) > 0:
                    print ('Remaining {} samples'.format(len(remaining_keys)))
                    ds = [all_keys[k] for k in remaining_keys]
                else:
                    print('All samples collected, nothing to generate.')
                    continue
                
                print (f'Previously collected {len(collected)} samples, left {len(ds)} samples to collect')
            with open(meta_data_path, 'r') as mf:
                meta_data = [json.loads(line) for line in mf][0]
                print (f'Previous cost for gpt-3.5-turbo: ${meta_data["gpt-3.5-turbo"]["total_cost"]}, gpt-4: ${meta_data["gpt-4"]["total_cost"]}')
                
        max_batches = int(np.ceil(len(ds)/args.batch_size))
        for data_iter,data_id in tqdm(enumerate(range(0,len(ds),args.batch_size)),total = max_batches,desc = 'Generating Q&A from OpenAI'):
            curr_data = ds[data_id:data_id+args.batch_size]
            original_data = deepcopy(curr_data) # copy an instance for out.
            out_d = [] # we keep track of this original data to do comparison later
            # define the batch data
            checked_data = []
            ## Doing checks and changing some keys and stuff to accom the generation
            for data_i,data in enumerate(curr_data):
                if 'question' in data:
                    qu  = data['question']
                else:
                    qu = data['context']
                data['question'] = qu
                if 'cf' not in args.perturbation_types or 'cf_edit' not in args.perturbation_types:
                    if 'explanation' not in data.keys():
                        continue
                    if cot_type in ['cot_qd','cot_cf']: # for qd, select a single qa at random, cot_cf choose the explanation corresponding to answer.
                        expl = get_explanation(data,cot_type = cot_type)
                        if expl is None:
                            continue
                    else:
                        expl  = data['explanation']
                    data['explanation'] = expl
                checked_data.append(data)
                out_d.append(original_data[data_i])
            print (f'Original size: {len(curr_data)}, Current: {len(checked_data)}')
            if len(checked_data)< 1: # if nth to generate,skip it.
                continue
            for objective in args.perturbation_types:
                
                model = get_model[objective]
                prompts = [create_prompt(c,objective = objective,completion=args.completion,dataset = args.dataset,cot_type = cot_type) for c in checked_data] # contains both prompt input and fs_shot (fs_shot to be used only for chat)
                
                ## get initial answer from GPT3.5
                if objective in ['para']:
                    initial_prompts,initial_ans = generate_answer(checked_data,20,cot_type)
                    initial_inputs = deepcopy(checked_data)
                    for i,ii in enumerate(initial_inputs):
                        ii['answer'] = initial_ans[i]
                    for ip in initial_prompts:
                        meta_data[model]['num_input_tokens'] += len(tokenizer.encode(ip,add_special_tokens = False))
                else:
                    initial_inputs = None

                # record num of input tokens
                for p in prompts: 
                    for p_obj in p[0].keys():
                        meta_data[model]['num_input_tokens'] += len(tokenizer.encode(p[0][p_obj],add_special_tokens = False))
                        if not args.completion and len(p[1][p_obj])>0:
                            curr_fs = [pp for pp in p[1][p_obj]]
                            all_text = [v for fs in curr_fs for v in fs.values()]
                            meta_data[model]['num_input_tokens'] += len(tokenizer.batch_encode_plus(all_text,add_special_tokens = False))
                # Send in batch request
                curr_size = len(prompts)
                batch_gen = batchgen_w_retries(batch_gen_fn,gen_fn,prompts,model,tokenizer,max_decode_len = max_decode_len,curr_size = curr_size,objective=objective,args = args,cot_type=cot_type,initial_inputs=initial_inputs)
                out_gen = [bg['text'] for bg in batch_gen]
                total_out_tokens = sum([bg['num_out_tokens'] for bg in batch_gen])
                meta_data[model]['num_output_tokens'] += total_out_tokens
                
                out_d = post_process(out_d,out_gen,checked_data,objective=objective,cot_type = cot_type)

            # print (out_dict)
            
            ## due to the fact that there can be trouble generating cf, skip entire instance then
            if 'cf' in args.perturbation_types:
                out_d = [od for od in out_d if od['cf'] != {}]
                for od in out_d:
                    del od['explanation']
                if len(out_d) < len(curr_data):
                    print (f'Pruned {len(curr_data)-len(out_d)} instances due to cf generation error')
            
            for od in out_d:
                append_to_jsonl(od,perturbation_path)
            for m in ['gpt-3.5-turbo','gpt-4']:
                meta_data[m]['total_cost'] = cal_cost(m,meta_data[m]['num_input_tokens'],meta_data[m]['num_output_tokens'])
                append_to_jsonl(meta_data,meta_data_path,overwrite=True)
            time.sleep(3) # sleep for 2 seconds to avoid rate limit
            print (f"Time for {data_id+args.batch_size} instances: {np.round(time.time()-starting_time,2)}s")
            for m in ['gpt-3.5-turbo','gpt-4']:
                print (f"Model: {m}, Cost: ${np.round(meta_data[m]['total_cost'],4)}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',type=str,default='obqa',help='Dataset to generate perturbations for')
    parser.add_argument('--completion',type=bool,default=False,help='If not completion, use chat model.')
    parser.add_argument('--cot_type',type=str,default='cot',nargs = '+',help='prompt type')
    parser.add_argument('--temp',type=float,default=1,help='temp for decoding')
    parser.add_argument('--top_p',type=float,default=0.95,help='nuclues sampling')
    parser.add_argument('--n',type=int,default=1,help='number of return sequences')
    parser.add_argument('--batch_size',type=int,default=8,help='batch size')
    parser.add_argument('--seed',type=int,default=41,help='batch size')
    parser.add_argument('--perturbation_types',type=str,default=8,nargs = '+',help='perturbation types')
    parser.add_argument('--num_shot',type=int,default=7,help='number of fs shots, just for naming')
    parser.add_argument('--model_size',type=int,default=70,help='size of model, in B')
    parser.add_argument('--path_suffix',type=str,default='',help='path_suffix, additional for ablation runs')
    args = parser.parse_args()
    main(args)
    
    
        

    




