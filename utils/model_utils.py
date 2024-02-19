import torch
import numpy as np
import os
from rouge import Rouge 
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,pipeline
from utils.data_helper import load_raw_dataset,paraphrase_inp
from tqdm import tqdm
import json
import re
import copy
import random
from copy import deepcopy


def get_rouge(pred,gold,tokenizer):
    all_score = []
    rouge = Rouge()
    for p,g in zip(pred,gold):
        p_text = clear_pad(p.tolist(),tokenizer)
        g_text = clear_pad(g.tolist(),tokenizer)
        rouge_score = rouge.get_scores(p_text,g_text,avg=True)
        rouge_score_l = rouge_score['rouge-l']['f']
        all_score.append(rouge_score_l)
    return np.mean(all_score)

def clear_pad(x,tokenizer,clear_expl=False):
    if x[0] == tokenizer.pad_token_id:
        x = x[1:]
    if tokenizer.eos_token_id in x:
        x = x[:x.index(tokenizer.eos_token_id)]
    elif tokenizer.pad_token_id in x:
        x = x[:x.index(tokenizer.pad_token_id)]
    out= tokenizer.decode(x,skip_special_tokens=True)
    if clear_expl:
        if 'Explanation:' in out:
            out = out.replace("Explanation:","").strip()
        elif 'explanation:' in out:
            out = out.replace("explanation:","").strip()
    return out
    
def faithfulness_test(inputs,mask,targets,label,model,tokenizer,args):
    out = {}
    ## Paraphrase
    for k in inputs.keys():
        in_embeds = False
        m = mask[k]
        i = inputs[k]
        if k == 'noisy':
            in_embeds = True
        if args.gen:
            acc,probs,gen_ids,_ = inference_generate(i,m,targets,model,tokenizer,args,in_embeds)
        else:
            acc,probs,gen_ids = simple_forward(i,m,targets,label,model,args.num_choices,in_embeds)
        if k == 'paraphrase':
            out['p_acc'] = acc
            out['p_probs'] = probs
            out['p_gen_ids'] = gen_ids
        elif k == 'scambled':
            out['s_acc'] = acc
            out['s_probs'] = probs
            out['s_gen_ids'] = gen_ids
        elif k == 'noisy':
            out['n_acc'] = acc
            out['n_probs'] = probs
            out['n_gen_ids'] = gen_ids
    return out
    

def get_log_probs(logits,target,num_choices):
    log_probs = - F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-100, reduction='none') 
    log_probs = log_probs.view(-1, target.size(-1)).sum(dim=-1)
    seq_lengths = (target != -100).sum(dim=-1) * 1.0
    log_probs /= seq_lengths
    log_probs = log_probs.view(-1,num_choices)
    
    return log_probs

def simple_forward(i,m,t,l,model,nc,in_embeds=False):
    if in_embeds:
        p_out = model(inputs_embeds = i,
                      attention_mask = m,
                      labels = t)
    else:
        p_out = model(input_ids = i,
                    attention_mask = m,
                    labels = t)
    p_out_logits = p_out.logits
    t_mask = (t != -100).long()
    p_out_id = torch.argmax(p_out_logits,dim=-1) * t_mask
    p_out_id = p_out_id.reshape(-1,nc,p_out_id.size(-1))
    p_logprobs = get_log_probs(p_out_logits,t,nc)
    p_pred = torch.argmax(p_logprobs,dim=-1)
    p_acc  = p_pred.eq(l).float().mean()
    p_probs = (torch.gather(p_logprobs,dim=-1,index=l.unsqueeze(-1)).squeeze(-1)).exp()
    
    return p_acc.item(),p_probs,p_out_id

def inference_generate(i,m,t,model,tokenizer,args,in_embeds=False):
    if in_embeds:
        p_out = model.generate(inputs_embeds = i,
                               attention_mask = m,
                               top_p = args.top_p,
                               num_beams = args.num_beams,
                               early_stopping = True,
                               max_length = 24,
                            ) # short answer
    else:
        p_out = model.generate(input_ids = i,
                               attention_mask = m,
                               top_p = args.top_p,
                               num_beams = args.num_beams,
                               early_stopping = True,
                               max_length = 24) # short answer
    total_acc = []
    out_gold_text = []
    for p,g in zip(p_out,t):
        curr_take_last = False
        g_text,curr_take_last = extract_text_answer(clear_pad(g.tolist(),tokenizer))
        if g_text == '': # if gold cant be extracted (shld not be)
            continue
        out_gold_text.append(g_text)
        p_text,_ = extract_text_answer(clear_pad(p.tolist(),tokenizer),curr_take_last)
        total_acc.append(p_text == g_text)
    return np.mean(total_acc),[],p_out,out_gold_text

def extract_text_answer(x,directly_take_last=False,choices = None): # only for answer in the form of (a) Answer
    take_last = False
    original_x = copy.deepcopy(x)
    alpha_mapping = {chr(ord('a') + i):i for i in range(len(choices))}
    if not directly_take_last:
        if '(' in x and ')' in x :
            x = re.findall(r'\(([a-zA-Z])\)', x)
            if len(x) > 0:
                x = x[0].strip().lower()
                try:
                    return alpha_mapping[x],take_last
                except KeyError:
                    pass

    if choices is not None: # give answer choices to get text, for example (a) tom (b) jerry, pred answer is ... (a tom, cant parse (a), take tom
        for choice in choices:
            if choice.lower() in original_x.lower():
                return choices.index(choice),take_last
        return -1,None # if not inside, just return -1
    else:
        if len(original_x.strip()) < 1:
            return '',False
        x = take_last_text(original_x)
    return x, take_last

def take_last_text(x):
    out = x.lower().strip().split(' ')
    if len(out) > 1:
        out = out[1:]
    out = ' '.join(out)
    if out[-1] == '.':
        out = out[:-1]
    return out

def separate_expl(text):
    """
    use if text is in the form of <explanation>. So the answer is (a)
    """
    pattern = r'(.*?\.)\s*So the answer is \(([a-zA-Z])\)'
    match = re.match(pattern, text)
    ans_match = re.search(r'\(([a-zA-Z])\)', text)

    if match:
        explanation = match.group(1)
        explanation = re.sub(r"^[A-Z]: ", "", explanation)
        answer_letter = match.group(2)  # This captures the alphabet inside the parentheses
    else:
        # Split from the last period
        parts = text.rsplit('.', 1)
        
        if len(parts) > 1:  # Ensure there's an explanation and some other text
            explanation = parts[0] + '.'  # Add the period back to the explanation
            explanation = re.sub(r"^[A-Z]: ", "", explanation)

            # Extract any alphabets inside parentheses from the sentence after the last period
            # post_explanation = parts[1]
            # bracket_match = re.search(r'\(([a-zA-Z])\)', post_explanation)
            # if bracket_match:
            #     answer_letter = bracket_match.group(1)
            if ans_match:
                answer_letter = ans_match.group(1)
            else:
                answer_letter = -1
        else:
            explanation = text
            if ans_match:
                answer_letter = ans_match.group(1)
            else:
                answer_letter = -1
    return explanation, answer_letter

def separate_qd_expl(text,cot_type = 'cot_qd',num_choices = 2):
    """
    Format:
    Q1 : subq 1
    Q2 : subq 2 ....
    A1: ans to Q1
    A2: ans to Q2 ....
    Thus .... So the answer is (ans)
    Get A1,A2 and ans
    """
    full_text = deepcopy(text)
    text_segs = text.split('\n')
    curr_id = 1
    curr_q_id,curr_a_id = 1,1
    expls = []
    ans = -1
    ans_pattern = r'\(([a-zA-Z])\)'
    for text in text_segs:
        if cot_type == 'cot_qd':
            header = [f'Q{curr_q_id}',f'A{curr_a_id}']
            
        if len(text) <1:
            continue
        if cot_type == 'cot_cf':
            # if text.startswith(header + '.'):
            if ')' in text and 'so the answer' not in text.lower():
                split_ = text.split(')')
                expls.append(split_[1].strip())
                # curr_id += 1
            elif 'so the answer' in text.lower():
                ans_match = re.search(ans_pattern,text) # search the entire string
                if ans_match:
                    ans = ans_match.group(1)
                    break
        elif cot_type == 'cot_qd':
            if text.startswith(header[0]):
                expls.append(text)
                curr_q_id += 1
            elif text.startswith(header[1]):
                expls.append(text)
                curr_a_id += 1
            else:
                expl_ans_pattern = r'(.*?\.)\s*So the answer is \(([a-zA-Z])\)'
                ea_match = re.match(expl_ans_pattern, text)
                if ea_match:
                    ans = ea_match.group(2)
                    break
    # finish getting explanation, get answer. Check conditions
    """
    For cot-qd, explanation need to be > 0 and ans != -1
    For cot-cf, explanation needs to be = choices, ans != -1
    If not fufil, for cot-qd, return empty expls and -1 or try to find ans again if expl is not empty.
    """
    if cot_type == 'cot_cf':
        if ans != -1 and len(expls) == num_choices:
            return expls,ans
        if len(expls) != num_choices:
            return [],-1
        elif ans == -1:
            for text in text_segs:
                ans_match = re.search(ans_pattern,text)
                if ans_match:
                    ans = ans_match.group(1)
                    return expls,ans
            return expls,-1
    else: # for cot-qd
        if ans!= -1 and len(expls) >0:
            return expls,ans
        if len(expls) < 1:
            return [],-1
        elif ans == -1:
            take_last = full_text.split('\n\n')[0].strip() # separate from the last part which is nonsense.
            ans_match = re.search(ans_pattern,take_last)
            if ans_match:
                ans = ans_match.group(1)
                return expls,ans
            else:
                return expls,-1

def separate_ans(text):
    if '(' and ')' in text:
        bracket_match = re.search(r'\(([a-zA-Z])\)', text)
        if bracket_match:
            answer_letter = bracket_match.group(1)
            return '',answer_letter
    if ('So the answer is') in text:
        answer_letter = text.split('So the answer is')[-1].strip()
        if len(answer_letter) > 0:
            answer_letter = answer_letter[0]
        else:
            answer_letter = ''
    else:
        answer_letter = text.split(' ')[-1] # take last word
    return '',answer_letter
        
def load_paraphrase(args):
    load_para = False
    if not args.same_module:
        data_dir = f'./data/{args.dataset}/para.jsonl'
    else:
        data_dir = f'./data/{args.dataset}/para_same_{args.seed}.jsonl'
    if not os.path.exists(data_dir):
        if not args.same_module:
            print ('generating paraphrased explanations')
            para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(f'cuda:{args.gpu_no[0]}')
            para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            all_para = []
            testset = load_raw_dataset('test', args)
            gen_batch = 4
            total_len = len(testset)
            for ex_id in tqdm(range(0,total_len,gen_batch),total=int(total_len//gen_batch),desc='paraphrasing'):
                expl = [ex.explanation for ex in testset[ex_id:ex_id+gen_batch]]
                para_expl = paraphrase_inp(expl,para_tokenizer,para_model)
                all_para.extend([{'para_explanation':p} for p in para_expl]) # para_expl is list of list where each list is one instance
            with open(data_dir,'w') as f:
                for para in all_para:
                    f.write(json.dumps(para))
                    f.write('\n')
        else: # if same module, we have to use the loaded model to generate explanations first before paraphrasing, in this case we will save it during test time. Since seeded, assume it remains the same within each seed run.
            all_para = None
            load_para = True
    else:
        print ('paraphrased explanations already generated')
        with open(data_dir,'r') as f:
            all_para = [json.loads(line) for line in f] # list of dicts
            
    return all_para,load_para


def setup_pipeline(model,tokenizer,args):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_dec_length,
        top_p=args.top_p,
        repetition_penalty=1.15,
        return_full_text=False,
        do_sample=True,
        temperature=args.temp,
        # device = f'cuda:{args.gpu_no[0]}'
    )
    return pipe

def insert_special_tok(tokenizer,model,tok,tok_name):
    added_tokens = tokenizer.add_special_tokens({tok_name: tok})
    # If a token was added, resize the model’s token embeddings
    if added_tokens > 0:
        # 2. Resize the model's embedding layer
        model.resize_token_embeddings(len(tokenizer))
        
    # 3. Set the model's pad_token_id
    if tok_name == "pad_token":
        model.config.pad_token_id = tokenizer.pad_token_id
        print ("Added pad token")
    elif tok_name == "sep_token":
        model.config.sep_token_id = tokenizer.sep_token_id  
        print ("Added sep token")
    else:
        raise ValueError(f"tok_name should be pad_token or sep_token, got {tok_name}.")


def clean_generate(text,checker,num_choices,prompt_type): # checker is the text to check the start of the instance
    """
    Only used for generate function as output contains input as well.
    """
    checker = checker.strip()
    original_text = copy.deepcopy(text)
    expl_a = text
    if '\n\n' in text:
        texts = text.split('\n\n')
        for t in texts:
            ques_only = t.split('Q:')[-1].strip().split('\n')[0].strip() # split from Q: get after Q, then split using newline and get the first part
            if ques_only == checker or ' '.join(ques_only.split(' ')[:3]) == ' '.join(checker.split(' ')[:3]): # if the first 3 words are the same or exact same
                expl_a = extract_w_objective(t,prompt_type)
    elif '\n' in text:
        texts = text.split('\n')
        for i,t in enumerate(texts):
            ques_only = t.split('Q:')[-1].strip().split('\n')[0].strip() 
            if ques_only == checker or ' '.join(ques_only.split(' ')[:3]) == ' '.join(checker.split(' ')[:3]):
                expl_a = extract_w_objective(t,prompt_type)
        
    if original_text == expl_a:
        # print (original_text)
        return ''
    else:
        return expl_a

def extract_w_objective(text,prompt_type):
    """
    Supplements clean_generate with prompt_type
    different prompt_type extracts text differently
    """
    t_split = text.split('\n')
    if prompt_type in ['cot','cot_sc']:
        out = t_split[-1].strip()
    elif prompt_type in ['cot_qd','cot_cf']:
        start_id = 0
        for i,t in enumerate(t_split):
            if t.startswith('A:'):
                start_id = i
        out = [ts.strip() for ts in t_split[start_id:]]
    
    return out
    
                

def find_subsequence(seq):
    """
    Used to find the last index of the explanation and return it. 
    Only used for tgi since the log probs correspond to word level.
    """
    tar_seq = 'so the answer is'
    for i in range(len(seq)-4):
        curr_seq = ''.join(seq[i:i+4]).strip().lower()
        if curr_seq == tar_seq:
            return i
    # if cant find the tar seq in the seq
    last_seq = '('
    for j,s in enumerate(seq):
        if s.strip() == last_seq:
            return j
    return -1

def clean_answer_only(text,checker=None,num_choices=None):
    text = text.split('\n')[0].strip()
    bracket_match = re.search(r'\(([a-zA-Z])\)', text)
    if bracket_match:
        answer_letter = bracket_match.group(1)
        return answer_letter.strip()
    else:
        return ''

def custom_shuffle(lst):
    original = lst.copy()
    shuffled = lst.copy()
    random.shuffle(shuffled)
    
    for i in range(len(lst)):
        # Check if the item is in its original position
        while shuffled[i] == original[i]:
            swap_with = random.choice(range(len(lst)))
            shuffled[i], shuffled[swap_with] = shuffled[swap_with], shuffled[i]
    
    return shuffled

                    