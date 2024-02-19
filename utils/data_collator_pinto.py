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
from template.prompt_template import format_llama_prompt,cot_template
from template.refine_template import format_refine_prompt
from utils.data_helper import *

def get_label_tensor(raw_label, tokenizer, args):
    if args.generative:
        raw_label = 'So the answer is ' + raw_label
    label_ids = tokenizer.encode(raw_label, add_special_tokens=True) # change to add eos token
    return label_ids

def format_input(question, choices=None,is_explain=False):
    if is_explain:
        input_seq = "Explain Question: {}".format(question.strip())
    else:
        input_seq = "Question: {}".format(question.strip())
    # input_seq += " Answer: {}.".format(choice.strip())
        if choices is not None:
            input_seq += " Answer Choices:"
            for choice_id, choice in enumerate(choices):
                input_seq += " ({}) {}".format(chr(ord('a')+choice_id), choice)
            input_seq += '.'
    return input_seq

def format_explanation(explanation,answer = None,gen=False):
    if not gen:
        input_seq = ' Explanation: ' + explanation.strip()
    else:
        input_seq = ' Explanation: ' + explanation.strip() + ' Answer: ' + answer.strip()
        if input_seq[-1] != '.':
            input_seq += '.'
    return input_seq

def format_ans_choice(choices):
    answers = []
    for choice_id, choice in enumerate(choices):
        answers.append(" ({}) {}".format(chr(ord('a')+choice_id), choice))
    return answers

class Data_Collator_for_Training(object):
    def __init__(self, tokenizer, args,dropout_context=0,split='train',para_model = None,para_tokenizer=None):
        self.tokenizer = tokenizer
        self.dropout_context = dropout_context 
        self.args = args
        self.split = split
        self.para_model = para_model
        self.para_tokenizer = para_tokenizer
        
    def __call__(self, examples):

        encoder_input_tensor = []
        encoder_attention_mask_tensor = []
        encoder_ans_tensors,encoder_ans_mask_tensors = [],[] # for separate learning objective
        
        e_masked_tensor,e_masked_tensor_mask = [],[]
        
        decoder_label_tensor = []
        decoder_explanation_label = []
        label_tensor = []
        smoothing_tensor = []
        
        context_only_tensor,context_only_mask = [],[]

        answer_choices_text = []
        
        raw_data = [{} for _ in range(len(examples))]
        
        for example_idx, example in enumerate(examples):

            context = example.context
            raw_data[example_idx]['question'] = context
            raw_data[example_idx]['choice'] = example.choices
            if not self.args.same_module:
                raw_data[example_idx]['explanation'] = example.explanation
            # input_ids += self.tokenizer.encode(context.strip(), add_special_tokens=False)

            starting_context = self.tokenizer.encode(format_input(context, example.choices,is_explain=self.args.same_module), add_special_tokens=False)
            starting_mask = [1] * len(starting_context)
            
            choices_input_ids = [deepcopy(starting_context) for _ in example.choices]
            choices_attention_mask = [deepcopy(starting_mask) for _ in choices_input_ids]
            
            masked_inps = [deepcopy(starting_context) for _ in example.choices]
            masked_inps_mask = [deepcopy(starting_mask) for _ in example.choices]
            
            
            context_only_ids = self.tokenizer.encode(format_input(context, example.choices,is_explain=False), add_special_tokens=False)
            context_only_m = [1] * len(context_only_ids)
            choices_context_only_ids = [deepcopy(context_only_ids) for _ in example.choices]
            choices_context_only_mask = [deepcopy(context_only_m) for _ in choices_context_only_ids]
            
            context_only_tensor.append(choices_context_only_ids)
            context_only_mask.append(choices_context_only_mask)
                    
            if self.args.same_module:
                decoder_choice_tensor = []
                choices_answer_input_ids = deepcopy(choices_context_only_ids)
                choices_answer_attention_mask = deepcopy(choices_context_only_mask)

            explanation = example.explanation
            smoothing_tensor.append(self.args.label_smoothing_no_inference)
            if not self.args.gen: # if not generative and not same module
                for choice_id in range(len(example.choices)):
                    added_ids = self.tokenizer.encode(format_explanation(explanation[choice_id]), add_special_tokens=False)
                    added_length = len(added_ids) 
                    masked_inps,masked_inps_mask = self.get_masked(masked_inps,masked_inps_mask,added_ids,choice_id)
                    
                    ## For normal
                    if self.args.same_module: # add as label instead
                        decoder_choice_tensor.append(added_ids+ [self.tokenizer.eos_token_id])
                        if self.split != 'test': # only add gold explanation during training, during eval/test we use the generated explanation
                            choices_answer_input_ids[choice_id] += added_ids
                            choices_answer_attention_mask[choice_id] += [1] * added_length
                    else:
                        choices_input_ids[choice_id] += added_ids
                        choices_attention_mask[choice_id] += [1] * added_length 
                        
                ## for labels 
                choices_tensor = []
                for choice_id, choice in enumerate(example.choices):
                    if self.args.same_module:
                        answer_choice = 'Answer: ' + choice
                        answer_choice_tensor = get_label_tensor(answer_choice, self.tokenizer, self.args)
                        choices_input_ids[choice_id] += answer_choice_tensor
                        choices_attention_mask[choice_id] += [1] * len(answer_choice_tensor)
                    choice_tensor = get_label_tensor(choice, self.tokenizer, self.args)
                    choices_tensor.append(choice_tensor)
                    
            ### GENERATIVE MODE
            else: 
                chosen_label = example.answer
                all_ans_choices = format_ans_choice(example.choices) # all answer choices in the form of (a) Answer 
                answer_choices_text.append([a.strip() + '.' for a in all_ans_choices])
                # masking for counterfactual
                for mi in range(len(example.choices)):
                    if mi != chosen_label: # if not answer, no need to mask
                        gen_mask_id = self.tokenizer.encode(format_explanation(explanation[mi],answer = all_ans_choices[mi],gen=True), add_special_tokens=False)
                        masked_inps[0] += gen_mask_id
                        masked_inps_mask[0] += [1] * len(gen_mask_id)
                    else:
                        gen_mask_id = self.tokenizer.encode(format_explanation(explanation[mi]), add_special_tokens=False)
                        gen_prefix = self.tokenizer.encode(' Answer: ' + all_ans_choices[mi].strip() + '.', add_special_tokens=False)
                        masked_inps,masked_inps_mask = self.get_masked(masked_inps,masked_inps_mask,gen_mask_id,0)
                        masked_inps[0] +=  gen_prefix # dont mask the prefix, only the explanation
                        masked_inps_mask[0] += [1] * len(gen_prefix)
                masked_inps = masked_inps[0] + [self.tokenizer.eos_token_id]
                masked_inps_mask = masked_inps_mask[0] + [1]
                        
                # Get normal inputs
                all_added_ids = [self.tokenizer.encode(format_explanation(explanation[ci],answer = all_ans_choices[ci],gen=True), add_special_tokens=False) for ci in range(len(example.choices))] # all formatted explanation with answer
                
                if not self.args.same_module:
                    for ai in all_added_ids:
                        choices_input_ids[0] += ai
                        choices_attention_mask[0] += [1] * len(ai)
                    choices_input_ids[0] += [self.tokenizer.eos_token_id]
                    choices_attention_mask[0] += [1]
                    choices_input_ids = choices_input_ids[0]
                    choices_attention_mask = choices_attention_mask[0]
                else:
                    if self.split != 'test':
                        for ai in all_added_ids:
                            choices_answer_input_ids[0] += ai
                            choices_answer_attention_mask[0] += [1] * len(ai)
                    choices_answer_input_ids[0] += [self.tokenizer.eos_token_id]
                    choices_answer_attention_mask[0] += [1]
                    choices_answer_input_ids = choices_answer_input_ids[0]
                    choices_answer_attention_mask = choices_answer_attention_mask[0]
                
                gen_label = all_ans_choices[chosen_label]
                choices_tensor = get_label_tensor(gen_label, self.tokenizer, self.args)
                    
                if self.args.same_module: # add explanation label and inputs
                    ans_prefix = ' Answer: '
                    for ci,ex in enumerate(example.choices):
                        added_ids = self.tokenizer.encode(format_explanation(explanation[ci]), add_special_tokens=False)
                        added_length = len(added_ids)
                        ## add explanation
                        decoder_choice_tensor.append(added_ids+ [self.tokenizer.eos_token_id])
                        ## add answer
                        encoded_ans = self.tokenizer.encode(ans_prefix + all_ans_choices[ci], add_special_tokens=True)
                        encoded_ans_mask = [1] * len(encoded_ans)
                        choices_input_ids[ci] += encoded_ans
                        choices_attention_mask[ci] += encoded_ans_mask
            
            encoder_input_tensor.append(choices_input_ids)
            encoder_attention_mask_tensor.append(choices_attention_mask)
            decoder_label_tensor.append(choices_tensor)
            label_tensor.append(example.answer)
            e_masked_tensor.append(masked_inps)
            e_masked_tensor_mask.append(masked_inps_mask)
            
            if self.args.same_module:
                encoder_ans_tensors.append(choices_answer_input_ids)
                encoder_ans_mask_tensors.append(choices_answer_attention_mask)
                decoder_explanation_label.append(decoder_choice_tensor)
        
        # do dynamic padding saves memory and time
        encoder_input_tensor,encoder_attention_mask_tensor = self.pad_tensors(encoder_input_tensor,encoder_attention_mask_tensor,self.tokenizer.pad_token_id) # normal
        e_masked_tensor,e_masked_tensor_mask = self.pad_tensors(e_masked_tensor,e_masked_tensor_mask,self.tokenizer.pad_token_id) # masking
        decoder_label_tensor = self.pad_tensors(decoder_label_tensor,None,-100) # normal target
        
        context_only_tensor,context_only_mask = self.pad_tensors(context_only_tensor,context_only_mask,self.tokenizer.pad_token_id) # context only
        
        ## decoder_label has to be same shape as we are doing cross entropy later on comparing sequence level.
        size_diff = encoder_input_tensor.shape[1] - decoder_label_tensor.shape[1]
        if size_diff > 0:
            padding = torch.full((decoder_label_tensor.shape[0], size_diff), -100)
            decoder_label_tensor = torch.cat((decoder_label_tensor, padding), dim=1)
        else:
            decoder_label_tensor = decoder_label_tensor[:,:encoder_input_tensor.shape[1]]
            
        label_tensor = torch.tensor(label_tensor, dtype=torch.long)
        smoothing_tensor = torch.tensor(smoothing_tensor, dtype=torch.float)
        
        if self.args.same_module:
            encoder_ans_tensors,encoder_ans_mask_tensors = self.pad_tensors(encoder_ans_tensors,encoder_ans_mask_tensors,self.tokenizer.pad_token_id)
            if len(decoder_explanation_label[0]) > 0:
                decoder_explanation_label = self.pad_tensors(decoder_explanation_label,None,-100)
            else:
                decoder_explanation_label = torch.tensor([])
        else:
            encoder_ans_tensors,encoder_ans_mask_tensors,decoder_explanation_label = None, None, None
        
        out =  {'input_ids': encoder_input_tensor,
                'attention_mask': encoder_attention_mask_tensor,
                'm_input_ids': e_masked_tensor,
                'm_attention_mask': e_masked_tensor_mask,
                'target_ids': decoder_label_tensor,
                'expl_target_ids': decoder_explanation_label,
                'labels': label_tensor,
                'smoothing_weights': smoothing_tensor,
                'ans_input_ids': encoder_ans_tensors,
                'ans_attention_mask': encoder_ans_mask_tensors,
                'context_only_ids': context_only_tensor,
                'context_only_mask': context_only_mask}
        out['raw_data'] = raw_data
        
        if self.split == 'test': ## add in additional perturbated inputs only for same_module
            perturbations = ['paraphrase','scambled','noisy']
            out_perturbated = defaultdict(list)
            out_perturbated_mask = {}
            # gen_para = []
            all_base_inps = []
            all_base_expl = []
            p_to_pad = True
            
            for ex in examples:
                base_inp = self.tokenizer.encode(format_input(ex.context, ex.choices,is_explain=False), add_special_tokens=False)
                base_inps = [deepcopy(base_inp) for _ in ex.choices]
                all_base_inps.append(base_inps)
                all_base_expl.append(ex.explanation)
                chosen_p_label = ex.answer
                
            for perturb in perturbations:
                copy_base_inps = deepcopy(all_base_inps) # copy for each perturbation just question
                copy_base_expl = deepcopy(all_base_expl)
                if perturb == 'paraphrase':
                    para_text = [ex.para_explanation for ex in examples]
                    original_expl = [ex.explanation for ex in examples]
                    if len(para_text[0]) != len(examples[0].choices): # need to generate outside of this since pytorch lightning dont allow shifting of tenor to gpu
                        to_gen = {'base_inp':copy_base_inps,'expl':copy_base_expl,'original_expl':original_expl}
                        out_perturbated[perturb].append(to_gen)
                        p_to_pad = False
                        
                    else: # already generated just add it in.
                        for p_i,(curr_pe,curr_oe) in enumerate(zip(para_text,original_expl)):
                            if self.args.gen:
                                copy_base_inps[p_i] = copy_base_inps[p_i][0]
                            for ids,(pe,oe) in enumerate(zip(curr_pe,curr_oe)):
                                if self.args.gen:
                                    add_e = oe if ids != chosen_p_label else pe # paraphrase only the explanation of the chosen answer
                                    copy_base_inps[p_i] += self.tokenizer.encode(format_explanation(add_e,answer =answer_choices_text[p_i][ids],gen=True), add_special_tokens=False)
                                else:
                                    copy_base_inps[p_i][ids] += self.tokenizer.encode(format_explanation(pe), add_special_tokens=False)
                        out_perturbated[perturb].extend(copy_base_inps)
                    
                elif perturb == 'scambled' and not self.args.same_module: # only for normal, same module need to use generated explanation
                    if not self.args.gen:
                        for o_idx,base_expl in enumerate(copy_base_expl):
                            random.shuffle(base_expl)
                            for i_idx,se in enumerate(base_expl):
                                copy_base_inps[o_idx][i_idx] += self.tokenizer.encode(format_explanation(se), add_special_tokens=False)
                    else: # for generative, no point scambling, just don't add in explanation for target answer.
                        for o_idx,base_expl in enumerate(copy_base_expl):
                            copy_base_inps[o_idx] = copy_base_inps[o_idx][0]
                            for i_idx,se in enumerate(base_expl):
                                if i_idx != chosen_p_label:
                                    copy_base_inps[o_idx] += self.tokenizer.encode(format_explanation(se,answer =answer_choices_text[o_idx][i_idx],gen=True), add_special_tokens=False)
                    out_perturbated[perturb].extend(copy_base_inps)

                elif perturb == 'noisy': # add noise during inference after converting to embeddings
                    encoded_noisy_expl = sum(copy_base_expl,[])
                    to_perturb = {'base_inp':copy_base_inps,'expl':encoded_noisy_expl}
                    out_perturbated[perturb].append(to_perturb)
            
            # pad the inputs to longest and create mask
            for pk,p in out_perturbated.items():
                if (pk == 'paraphrase' and not p_to_pad) or pk == 'noisy': # noisy also dont pad
                    out_perturbated[pk] = p
                    out_perturbated_mask[pk] = None
                else:
                    padded_p = self.pad_tensors(p,None,self.tokenizer.pad_token_id) # flattened
                    p_mask = (padded_p != self.tokenizer.pad_token_id).float()
                    out_perturbated_mask[pk] = p_mask
                    out_perturbated[pk] = padded_p
                
            out['perturbated_input_ids'] = out_perturbated
            out['perturbated_attention_mask'] = out_perturbated_mask
            if self.args.gen:
                out['answer_choices'] = answer_choices_text
            else:
                out['answer_choices'] = None
        return out

    def pad_tensors(self,inputs,mask,pad): # account for eos token id
        if isinstance(inputs[0][0],list): # list within a list of list of ids
            inputs =  [torch.tensor(inner[:self.args.max_enc_length-1] + [self.tokenizer.eos_token_id]) for outer in inputs for inner in outer] # unroll it down
            if mask is not None:    
                mask =  [torch.tensor(inner[:self.args.max_enc_length-1] + [1]) for outer in mask for inner in outer] # unroll it down
        else:
            inputs = [torch.tensor(ids[:self.args.max_enc_length-1]+ [self.tokenizer.eos_token_id]) for ids in inputs]
            if mask is not None:
                mask = [torch.tensor(ids[:self.args.max_enc_length-1] + [1]) for ids in mask]
        padded_inputs = pad_sequence(inputs,batch_first=True,padding_value=pad)
        if mask is not None:
            padded_mask = pad_sequence(mask,batch_first=True,padding_value=0)
            return padded_inputs, padded_mask
        return padded_inputs

    def get_masked(self,inps,mask,added_ids,choice_id):
        ## For masking 
        added_length = len(added_ids)
        if random.random() > self.args.mask_prob:
            # mask_ratio = random.uniform(self.args.min_mask_ratio, 1.0) 
            mask_idx = random.sample(range(added_length), int(added_length * self.args.replace_ratio))
            replaced_ids = [random.choice(range(len(self.tokenizer))) if _idx in mask_idx else added_ids[_idx] for _idx in range(added_length)]
            inps[choice_id] += replaced_ids
            mask[choice_id] += [1] * added_length
        else:
            mask_idx = random.sample(range(added_length), int(added_length * self.args.mask_ratio))
            inps[choice_id] += added_ids
            mask[choice_id] += [0 if _idx in mask_idx else 1 for _idx in range(added_length)]
        return inps,mask
    
class SimplerCollator(object):
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
        inps,masks,targs,labels,text_labels = [],[],[],[],[]
        x_masks,e_masks = [],[]
        all_choices = []
        
        for example in examples:
            context = example.context
            choices = example.choices
            all_choices.append(choices)
            explanation = example.explanation
            s_context = self.tokenizer.encode(format_input(context, choices,is_explain=False), add_special_tokens=False)
            s_mask = [1] * len(s_context)
            if isinstance(explanation,list):
                choice_ids = [deepcopy(s_context) for _ in choices]
                choice_mask = [deepcopy(s_mask) for _ in choices]
            else:
                choice_ids = [deepcopy(s_context)]
                choice_mask = [deepcopy(s_mask)]
                explanation = [explanation]
            
            if self.split == 'test':
                e_only_mask = [0] * len(s_context)
                if len(explanation) == len(choices):
                    e_choice_mask = [deepcopy(e_only_mask) for _ in choices]
                else:
                    e_choice_mask = [deepcopy(e_only_mask)]
                x_choice_mask = deepcopy(choice_mask)
            
            if self.mask_expl:
                if random.random() >0.5:
                    mask_ = True
                else:
                    mask_ = False
            for id, expl in enumerate(explanation):
                expl_id = self.tokenizer.encode(format_explanation(expl), add_special_tokens=True)
                choice_ids[id] += expl_id
                if self.split == 'train':
                    if mask_:
                        choice_mask[id] += [0] * len(expl_id)
                    else:
                        choice_mask[id] += [1] * len(expl_id)
                else:
                    e_choice_mask[id] += [1] * len(expl_id)
                    x_choice_mask[id] += [0] * len(expl_id)
                    choice_mask[id] += [1] * len(expl_id)
            
            tar_choices = []
            if not self.args.generative: # generative only has one answer
                for c_id,choice in enumerate(choices):
                    c_tensor = get_label_tensor(choice, self.tokenizer, self.args)
                    tar_choices.append(c_tensor)
            else: # no target, only label
                ans_label = '(' + self.alpha_numbering[example.answer] + ') ' + choices[example.answer]
                text_labels.append(torch.tensor(get_label_tensor(ans_label, self.tokenizer, self.args)))
            labels.append(example.answer)
            
            inps.append(choice_ids)
            targs.append(tar_choices)
            
            masks.append(choice_mask)
            if self.split == 'test':
                x_masks.append(x_choice_mask)
                e_masks.append(e_choice_mask)
        
        inps = self.pad_tensors(inps,self.tokenizer.pad_token_id)
        masks = self.pad_tensors(masks,0)
        if not self.args.generative:
            targs = self.pad_tensors(targs,-100)
        else:
            targs = torch.tensor([0]) # not used
        if self.args.generative:
            text_labels = pad_sequence(text_labels,batch_first=True,padding_value=-100)
        else:
            text_labels = None
        labels = torch.tensor(labels, dtype=torch.long)

        if self.split == 'test':
            x_masks = self.pad_tensors(x_masks,0)
            e_masks = self.pad_tensors(e_masks,0)
        else:
            x_masks,e_masks = None, None
        
        if len(explanation) != len(choices) and inps.shape[1] == 1:
            inps = inps.squeeze(1)
            masks = masks.squeeze(1)
            if self.split == 'test':
                x_masks = x_masks.squeeze(1)
                e_masks = e_masks.squeeze(1)
        
        return {'input_ids': inps,
                'attention_mask': masks,
                'target_ids': targs,
                'labels': labels,
                'x_mask': x_masks,
                'e_mask': e_masks,
                'choices':all_choices,
                "text_labels": text_labels}

    def pad_tensors(self,x,pad_value):
        x = [torch.tensor(outer) for inner in x for outer in inner]
        padded_x = pad_sequence(x,batch_first=True,padding_value=pad_value)
        return padded_x