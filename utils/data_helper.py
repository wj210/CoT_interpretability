import json
import os 
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, List, Optional,Union
import random

import torch
from torch.utils.data import Dataset, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from rouge import Rouge
import numpy as np
from collections import defaultdict
from template.prompt_template import format_llama_prompt,cot_template,prompt_input
from template.refine_template import format_refine_prompt
from utils.data_collator_pinto import format_input,get_label_tensor

@dataclass(frozen=True)
class InputExample:

    context: str
    explanation: Union[str, List[str]]
    choices: List[str]
    answer: int
    pred_answer: int
    para_explanation: Union[str, List[str]]
    noisy_explanation: Union[str, List[str]]
    cf_context: str
    cf_answer: int
    cf_edit: str
    cf_original: str

class TrainingDataset(Dataset):
    features: List[InputExample]

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputExample:
        return self.features[i]

def load_raw_dataset(split, args,data_path=None,have_expl=False,eval=False):
    """
    split = train, test
    args = args
    data_path is the path to the dataset file, provide if clear on where to load
    have_expl = whether dataset is supposed to have full set of explanation
    eval is used only when evaluating, do not use when not using perturbations
    
    """
    if eval:
        assert have_expl == True, 'Eval mode must have explanation'
    if data_path is None: # load using split
        data_path = os.path.join('./data', args.dataset, '{}.jsonl'.format(split))
    dataset = []
    if len(args.remaining_ds) > 0 and not have_expl: # this is only for gathering explanation, we will check if any remaining instances to generate and do so.
        all_examples = args.remaining_ds
    else:
        if not isinstance(data_path,list):
            data_path = [data_path]
        all_examples = []
        for dp in data_path:
            with open(dp, 'r') as fr:
                exampls = [json.loads(line) for line in fr]
                all_examples.extend(exampls)
    
    # Load this way as examples from two files may not be in same order, use question+choices as key, is unique.
    existing_keys = {}
    
    for example_id, example in tqdm(enumerate(all_examples), desc='processing {}'.format(data_path)):
        if "choice" in example:
            example['choices'] = example['choice']
        
        choices = example["choices"]
        if 'context' in example:
            example['question'] = example['context']
        
        context = example["context"] if "context" in example else example["question"]
        
        
        curr_key = f'{context}.{"".join(choices)}'
        ## Keep a dict of examples with unique keys as some examples may be overlapping with different keys
        if curr_key not in existing_keys:
            existing_keys[curr_key] = example
        else:
            for k,v in example.items():
                if k not in existing_keys[curr_key].keys():
                    existing_keys[curr_key][k] = v
        # if example_id >=16: # for testing
        #     break
    
    # only if have_expl and eval mode, will only accept those that have cf
    checked_existing_keys = {}
    no_count = 0
    if eval:
        for kk,vv in existing_keys.items():
            if 'cf' in vv.keys():
                checked_existing_keys[kk] = vv 
        existing_keys = checked_existing_keys
        
        if 'text' in args.eval_results:
            eval_keys = set(args.eval_results['text']) # list of strs {ques}.{choice} # only used if have_expl, meaning is in eval mode
        else:
            eval_keys = set()

        
    eval_already = 0
    no_expl = 0
    for kk,vv in existing_keys.items():
        ## check if have explanation and if empty
        if have_expl:
            if vv.get('explanation','') == '':
                no_expl += 1
                continue
            elif isinstance(vv['explanation'],list):
                if len(vv['explanation']) == 0:
                    no_expl += 1
                    continue
            elif isinstance(vv['explanation'],str):
                if len(vv['explanation'].strip()) == 0:
                    no_expl += 1
                    continue
            explanation = vv['explanation']
        else:
            explanation = ''
        
        
        if eval:
            ## This is for eval checkpointing, if already evaluated, skip
            curr_key = f'{vv["question"]}.{"".join(vv["choices"])}' # exist, continue on
            if curr_key in eval_keys:
                eval_already += 1
                continue

            cf_ques = vv['cf']['cf_question']
            cf_ans = vv['cf']['cf_answer']
            cf_edit = vv['cf']['edit']
            cf_original = vv['cf']['original']
            ## if there is None in cf_edit, skip
            if 'None' in cf_edit:
                no_count += 1
                continue
            if not isinstance(cf_ques,str): # might be int during generation
                no_count += 1
                continue
            # check explaantion based on prompt type
            chk_expls = [explanation]
            if vv.get('para','') == '':
                no_count += 1
                continue
            else:
                chk_expls.append(vv['para'])
            if vv.get('add_mistakes','') == '':
                no_count += 1
                continue
            else:
                chk_expls.append(vv['add_mistakes'])
            
            for expl in chk_expls:
                if args.prompt_type == 'cot_qd':
                    if len(expl) %2 != 0:
                        continue
                elif args.prompt_type == 'cot_cf':
                    if len(expl) != len(vv['choices']):
                        continue    
            
        else:
            cf_ques,cf_ans,cf_edit,cf_original = '','','',''

        
        dataset.append(
                InputExample(
                        context=vv["question"],
                        explanation=explanation,
                        choices=vv['choices'],
                        answer=vv["answer"],
                        pred_answer = vv.get('pred_answer',-1),
                        para_explanation=vv.get('para',''),
                        noisy_explanation=vv.get('add_mistakes',''),
                        cf_context=cf_ques,
                        cf_answer = cf_ans,
                        cf_edit = cf_edit,
                        cf_original = cf_original
                    ))
        
    dataset = dataset[:args.max_ds_size]

    print (f'Total Sample size: {len(dataset)}')
    print (f'Total no count due to counterfactuals: {no_count}')
    print ('Total no expl: {}'.format(no_expl))
    if have_expl:
        print (f'Total tested already: {eval_already}') 
    # exit()

    return TrainingDataset(dataset)
    

## Purely for text outputs for llama model, using pipeline.
class TextCollator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.fs_prompt = cot_template[args.dataset]
        self.prompt_template = format_llama_prompt
    
    def __call__(self,examples):
        context = [ex.context for ex in examples]
        choices = [ex.choices for ex in examples]
        answer = [ex.answer for ex in examples]
        explanation = [ex.explanation for ex in examples]
        pred_answer = [ex.pred_answer for ex in examples]
        para_expl = [ex.para_explanation for ex in examples]
        noisy_expl = [ex.noisy_explanation for ex in examples]
        cf_context = [ex.cf_context for ex in examples]
        cf_answer = [ex.cf_answer for ex in examples]
        cf_edit = [ex.cf_edit for ex in examples]
        cf_originals = [ex.cf_original for ex in examples]
            
        out = {}
        if self.args.get_expl:
            full_prompt = []
            for i,cont in enumerate(context):
                if 'refine'in self.args.prompt_type: # is a dict has 3 keys init, feedback, refine, only during getting explanation.
                    inp_prompt = cont # just pass context
                else:
                    inp_prompt = self.prompt_template(cont,choices[i],self.fs_prompt,prompt_type= self.args.prompt_type,num_shot = self.args.num_shot)
                full_prompt.append(inp_prompt)
        else:
            full_prompt = defaultdict(list)
            for i,cont in enumerate(context):

                if 'paraphrase' in self.args.perturbation_type:
                    full_prompt['paraphrase'].append(self.prompt_template(cont,choices[i],self.fs_prompt,prompt_type= self.args.prompt_type,explanation = para_expl[i],num_shot = self.args.num_shot))
                    
                if 'noisy' in self.args.perturbation_type:
                    full_prompt['noisy'].append(self.prompt_template(cont,choices[i],self.fs_prompt,prompt_type= self.args.prompt_type,explanation = noisy_expl[i],num_shot = self.args.num_shot))
                    
                if 'cf' in self.args.perturbation_type:
                    if 'refine'in self.args.prompt_type:
                        full_prompt['cf'].append(cf_context[i])
                    else:
                        full_prompt['cf'].append(self.prompt_template(cf_context[i],choices[i],self.fs_prompt,prompt_type= self.args.prompt_type,num_shot = self.args.num_shot))

        out['prompt'] = full_prompt
        out['gold_ans'] = answer
        out['choices'] = choices
        out['inp_only'] = context
        out['cf_inp_only'] = cf_context
        out['cf_answer'] = cf_answer
        out['cf_edits'] = cf_edit
        out['cf_originals'] = cf_originals
        out['perturbated_expl'] = {'paraphrase': para_expl,'noisy': noisy_expl}
        
        if explanation[0] is not None:
            out['explanation'] = explanation
        if pred_answer[0] is not None:
            out['pred_answer'] = pred_answer
            
        return out

class T5Collator(object):
    def __init__(self, tokenizer, args,split='train'):
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        if split == 'train':
            self.mask_expl = True
        else:
            self.mask_expl = False # to mask explantion occassionally for LAS score 
        self.alpha_numbering = {choice_id : chr(ord('a')+choice_id) for choice_id in range(self.args.num_choices)}
    
    def __call__(self,examples):
        inps,masks,labels,text_labels = [],[],[],[]
        x_masks,e_masks = [],[]
        all_choices = []
        
        for example in examples:
            context = example.context
            choices = example.choices
            all_choices.append(choices)
            explanation = example.explanation
            s_context = self.tokenizer.encode(format_input(context, choices,is_explain=False), add_special_tokens=False)
            s_mask = [1] * len(s_context)
            
            if self.split == 'test':
                e_choice_mask = [0] * len(s_context)
                x_choice_mask = deepcopy(s_mask)
                
            if self.mask_expl:
                if random.random() >0.5:
                    mask_ = True
                else:
                    mask_ = False
            formatted_expl = prompt_input(self.args.prompt_type,explanation,add_postfix=False)
            expl_id = self.tokenizer.encode(formatted_expl, add_special_tokens=True)
            s_context += expl_id
            if self.split == 'train':
                if mask_:
                    s_mask += [0] * len(expl_id)
                else:
                    s_mask += [1] * len(expl_id)
            else:
                e_choice_mask += [1] * len(expl_id)
                x_choice_mask += [0] * len(expl_id)
                s_mask += [1] * len(expl_id)
            
            ans_label = '(' + self.alpha_numbering[example.answer] + ') ' + choices[example.answer]
            text_labels.append(torch.tensor(get_label_tensor(ans_label, self.tokenizer, self.args)))
            labels.append(example.answer)
            inps.append(torch.tensor(s_context))
            masks.append(torch.tensor(s_mask))
            if self.split == 'test':
                x_masks.append(torch.tensor(x_choice_mask))
                e_masks.append(torch.tensor(e_choice_mask))
        
        inps = self.pad_tensors(inps,self.tokenizer.pad_token_id)
        masks = self.pad_tensors(masks,0)
        text_labels = pad_sequence(text_labels,batch_first=True,padding_value=-100)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if self.split == 'test':
            x_masks = self.pad_tensors(x_masks,0)
            e_masks = self.pad_tensors(e_masks,0)
        else:
            x_masks,e_masks = None, None
        
        return {'input_ids': inps,
                'attention_mask': masks,
                'labels': labels,
                'x_mask': x_masks,
                'e_mask': e_masks,
                'choices':all_choices,
                "text_labels": text_labels}
        
    def pad_tensors(self,x,pad_value):
        padded_x = pad_sequence(x,batch_first=True,padding_value=pad_value)
        return padded_x


def paraphrase_inp(x,tokenizer,model,encoded=False):
    """
    paraphrase a batch of text where each instance has num choices of explanation
    x is the text to be paraphrase
    """
    rouge = Rouge()
    if not encoded:
        num_choices = len(x[0])
        x_flatten = sum(x,[]) # flatten list
        x_in = ['paraphrase: ' + _x + ' </s>' for _x in x_flatten]
        enc_x = tokenizer.batch_encode_plus(x_in, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    else:
        enc_x,x_flatten,num_choices = x['enc_expl'],x['original_expl'],x.get('num_choices',None)
    x_inps,x_mask = enc_x['input_ids'],enc_x['attention_mask']
    
    outputs = model.generate(input_ids=x_inps.to(model.device), attention_mask=x_mask.to(model.device),
            max_length=128,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
        )
    
    decoded_out = tokenizer.batch_decode(outputs, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    nested_list = [decoded_out[i:i+5] for i in range(0, len(decoded_out), 5)] # each list is one example
    out_para = []
    for ids,para_list in enumerate(nested_list):
        para_list = [p for p in para_list if len(p.strip())>0] # remove empty string
        if len(para_list) == 0:
            out_para.append(x_flatten[ids]) # if no paraphrase, just use the original
            continue
        try:
            rouge_score = rouge.get_scores(para_list,[deepcopy(x_flatten[ids]) for _ in para_list],avg=False)
        except:
            out_para.append(x_flatten[ids]) # if error, just use the original
            continue
        rouge_f1 = [r['rouge-1']['f'] for r in rouge_score] # use unigrams 
        best_id = np.argmin(rouge_f1) # pick the least 
        out_para.append(para_list[best_id])
    if num_choices is not None:
        out_para = [out_para[i:i+num_choices] for i in range(0, len(out_para), num_choices)] # arrange back to batches
    return out_para
        
        
        
        
    
    
    
    
    
        
        

