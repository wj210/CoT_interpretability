import json
import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import math

import torch
torch.set_float32_matmul_precision('medium')
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import set_seed, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup,AdamW,AutoModelForCausalLM,pipeline
from transformers.optimization import Adafactor

from utils.data_helper import *
from utils.utils import *
from utils.model_utils import *
import math
from copy import deepcopy,copy
from time import time
from template.prompt_template import cot_template,format_llama_prompt,create_entailment_prompt
from template.refine_template import process_answer,process_feedback,format_refine_prompt
from text_generation import Client
import concurrent.futures
from utils.las import las_cot
import yaml
import pickle
from functools import partial


# os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_choices = {'csqa':5,'obqa':4,'qasc':8,'strategyqa':2}

def default_factory():
    return defaultdict(list)

class LlamaTester(object):
    """
    Can use both TGI or normal model.generate.
    Standard generate uses gpu memory for batch, while TGI uses number of workers instead
    """
    def __init__(self, args):
        self.pipe = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if not args.tgi: 
            if args.model_name.startswith('t5'):
                self.model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                device_map = 'auto'
                )

            else:
                self.model = AutoGPTQForCausalLM.from_quantized(args.model_name,
                        inject_fused_attention=False, # Required for Llama 2 70B model at this time.
                        use_safetensors=True,
                        max_memory=args.memory_map,
                        trust_remote_code=False,
                        device_map='auto',
                        use_triton=False,
                        quantize_config=None)
                
                if '70B' in args.model_name:
                    if args.num_seq > 1:
                        # self.model = exllama_set_max_input_length(self.model, args.eval_batch_size*args.num_seq*4096)
                        self.model = exllama_set_max_input_length(self.model, args.num_seq*4096)
                    else:
                        self.model = exllama_set_max_input_length(self.model, args.eval_batch_size*4096)
            
            ## add pad token
            if not self.tokenizer.pad_token:
                insert_special_tok(self.tokenizer,self.model,"<pad>","pad_token")
            # device = torch.device(f"cuda:{self.device}")
            if args.get_expl: # this is to collect all the tensors before gathering at the end ( used for distributed )
                self.expl,self.ans,self.pred_ans,self.prompt,self.choices = [],[],[],[],[]
                if not self.tokenizer.sep_token:
                    insert_special_tok(self.tokenizer,self.model,"<sep>","sep_token",)
            
            ## setup pipeline
            if args.use_pipe:
                self.pipe = setup_pipeline(self.model,self.tokenizer,args=self.args)
        else:
            # TGI set up client
            self.client = Client(f"http://127.0.0.1:{args.port}",timeout = 1000)

        # set alpha to numerical mapping
        self.alpha_to_num = {chr(ord('a') + i):i for i in range(args.num_choices)} 
        
        # no yes mapping
        if args.prompt_type == 'cot_sec':
            args.yes_token_id = self.tokenizer.encode('yes',add_special_tokens=False)[0]
            args.no_token_id = self.tokenizer.encode('no',add_special_tokens=False)[0]
            args.yes_token_id_cap = self.tokenizer.encode('Yes',add_special_tokens=False)[0]
            args.no_token_id_cap = self.tokenizer.encode('No',add_special_tokens=False)[0]
        
        self.args = args

    def evaluate(self,prompt,choices,answer,pred_answer,inp_only,explanation,cf_answer,cf_edits,cf_originals,cf_inp_only,perturbated_expl): 
        """
        Function for evaluating based on args.perturbation_type [cf, noisy, paraphrase]
        prompt: dict of prompt type to list of perturbation prompts
        choices: list of choices
        answer: gold label
        pred_answer: predicted label during normal inference
        inp_only: list of context only
        explanation: list of explanation
        cf_answer: counterfactual gold label
        cf_edits: edit tokens for cf
        cf_originals: original tokens for cf
        cf_inp_only: context only for cf
        perturbated_expl: dict of perturbation type to list of perturbed explanations (noisy and paraphrase)
        Note:
        During eval, gen mode is always greedy regardless of prompt_type, thus set to greedy.
        For cf, need to generate explanation (thus have to set to previous mode), for sc and sec, need to to set to previous num seq and on sample mode.
        
        """
        test_results = {}
        # answer derivation clean_answer_only
        p_labels,p_explanations,p_incl_ids = {},{},{}
        all_explanations, all_labels = {},{}
        ans_fn = {'context_only': separate_ans} # rest is separate_expl
        
        prev_num_seq,sample_m = self.set_greedy() # first set to greedy. to store the num_seq and sample mode
        
        for p_name,p_inputs in prompt.items():
            if p_name == 'cf': # for cf, we only change context and get the explanation to see if it includes the edits,
                self.unset_greedy(prev_num_seq,sample_m) # set to previous mode
                p_text,_ = self.get_explanations(p_inputs,choices,answer,cf_inp_only,eval=True) 
                prev_num_seq,sample_m = self.set_greedy() # set it back to greedy afterwards.
            else:
                if self.args.tgi:
                    p_text = self.tgi_generate(p_inputs,clean=True) # must set clean to True to clean the answer
                else:
                    p_text = self._generate(p_inputs,encoded=False) 
            p_ans,p_expl = [],[] # contain the final numerical ans index
            redo_p = []
            redo_ids = []
            for i,pt in enumerate(p_text):
                if p_name == 'cf':
                    pe,p_a = pt['explanation'],pt['pred_answer']
                    
                else:
                    pe = perturbated_expl[p_name][i]
                    if self.alpha_to_num.get(pt,None) is None: # means generated answer is not in the choices
                        p_a,_ = extract_text_answer(pt,choices = choices[i]) # is already numerical
                    else:
                        p_a = self.alpha_to_num[pt]
                if p_a == -1:
                    redo_p.append((p_inputs[i],inp_only[i]))
                    redo_ids.append(i)
                
                p_ans.append(p_a)
                p_expl.append(pe)
            # re-gen for ans not found in choice, max allowed generation = 3
            
            ## REDO
            redo_iters = 0
            max_redo_ans = 0 # Set to 0 to disable redo
            if max_redo_ans > 0:
                while len(redo_p) > 0:
                    print (f'Redoing answer generation for {self.args.dataset}, with {p_name} for {len(redo_p)} samples, left {3-redo_iters} tries')
                    next_redo_ids = []
                    next_redo_p = []
                    redo_inps = torch.stack([r[0] for r in redo_p])
                    if self.args.tgi:
                        redo_text = self.tgi_generate(redo_inps)
                    else:
                        redo_text = self._generate(redo_inps,encoded=True)
                    for r,redo_pt in enumerate(redo_text):
                        redo_pe, redo_p_a = ans_fn.get(p_name,separate_expl)(redo_pt)
                        redo_n_na = self.alpha_to_num.get(redo_p_a,-1)
                        if redo_n_na != -1:
                            p_ans[redo_ids[r]] = redo_n_na
                            p_expl[redo_ids[r]] = redo_pe
                        else:
                            next_redo_ids.append(redo_ids[r])
                            next_redo_p.append(redo_p[r])
                    redo_ids = next_redo_ids
                    redo_p = next_redo_p
                    redo_iters += 1
                    if redo_iters >= max_redo_ans:
                        break
            ### END REDO
            redo_ids += [i for i,pa in enumerate(pred_answer) if pa == -1] # add any pred answer that is -1
            redo_ids = list(set(redo_ids))
            incl_ids = list(range(len(answer)))
            
            # store all first to print to file later
            all_explanations[p_name] = p_expl
            all_labels[p_name] = p_ans
            
            ## Redo only contain samples when the predicted answer cannot be derived from the generated text. Or if the explanation is not derivable.
            # In eval, we care more about explanation thus explanation's derivation is more strictly checked. It should only be derivable when the answer is derivable.
            if len(redo_ids) > 0 :
                pert_l = [p for i,p in enumerate(p_ans) if i not in redo_ids]
                ori_l = [p for i,p in enumerate(pred_answer) if i not in redo_ids]
                if p_name != 'cf':
                    gold_l = [p for i,p in enumerate(answer) if i not in redo_ids]
                else:
                    gold_l = [p for i,p in enumerate(cf_answer) if i not in redo_ids]
                
                edits = [p for i,p in enumerate(cf_edits) if i not in redo_ids]
                originals = [p for i,p in enumerate(cf_originals) if i not in redo_ids]
                p_expln = [p for i,p in enumerate(p_expl) if i not in redo_ids]
                gold_l_all = [p for i,p in enumerate(deepcopy(answer)) if i not in redo_ids] # only used for cf
                incl_ids = [i for i in incl_ids if i not in redo_ids]
                self.args.eval_no_count[p_name] += len(redo_ids)
            else:
                pert_l = p_ans
                ori_l = pred_answer
                gold_l = answer if p_name != 'cf' else cf_answer
                edits = cf_edits
                originals = cf_originals
                p_expln = p_expl
                gold_l_all = deepcopy(answer)

            p_labels[p_name] = (pert_l,ori_l,gold_l,edits,originals,gold_l_all)
            p_explanations[p_name] = p_expln # keep to print out
            p_incl_ids[p_name] = set(incl_ids)
        
        self.unset_greedy(prev_num_seq,sample_m) # set to previous mode * This is important, otherwise it will affect the next batch.
        ## Extra not-counted if pred answer is -1
        gold_ans,ori_ans = [],[]
        for i,pa in enumerate(pred_answer):
            if pa != -1:
                gold_ans.append(answer[i])
                ori_ans.append(pa)

        # Get accuracy against gold answer (answer), label flipped against original pred ans (pred_answer)
        test_results['original'] = {}
        test_results['original']['acc'] = (np.array(ori_ans) == np.array(gold_ans)).mean()
        for pn,pl in p_labels.items():
            pred_labels = pl[0]
            original_labels = pl[1]
            gold_labels = pl[2]
            edits = pl[3]
            originals = pl[4]
            original_targs = pl[5]
            pred_expls = p_explanations[pn]
            
            if len(pred_labels) == 0: # if no samples, skip
                continue
            
            test_results[pn] = {} # init a empty dict for each results
            p_acc = (np.array(pred_labels) == np.array(gold_labels)).mean()
            test_results[pn]['acc'] = p_acc
            test_results[pn]['label_flip'] = (1 - (np.array(pred_labels) == np.array(original_labels)).mean())
            test_results[pn]['acc_diff'] = test_results['original']['acc'] - p_acc
            if pn == 'cf':
                # We only compute faithfulness test on instances when pred_labels = gold_labels and original labels = original_targs (correct in both normal and cf cases)
                correct_ids = []
                for ll,(pl,gl,ol,ot) in enumerate(zip(pred_labels,gold_labels,original_labels,original_targs)):
                    if pl == gl and ol == ot: # only assess when both yhat_c = y_c and yhat = y
                        correct_ids.append(ll)
                if len(correct_ids) > 0:
                    p_e,cf_e,cf_o = [pred_expls[i] for i in correct_ids],[edits[i] for i in correct_ids],[originals[i] for i in correct_ids]
                    if self.args.prompt_type == 'cot_cf': # only keep the explanation that is the correct answer.
                        pls = [pred_labels[i] for i in correct_ids]
                        selected_p_e = []
                        for e,p in zip(p_e,pls):
                            selected_p_e.append(e[p])
                        p_e = selected_p_e
                    test_results[pn]['unfaithfulness'] = detect_words(p_e,cf_e,cf_o,self.args.prompt_type)
        
        # Write text to file
        for k,v in all_explanations.items():
            new_v = []
            for vv in v:
                if isinstance(vv,list):
                    new_v.append( '\n' + '\n'.join(vv))
                else:
                    new_v.append(vv)
            all_explanations[k] = new_v
            
        new_explanation = []
        for el in explanation:
            if isinstance(el,list):
                new_explanation.append('\n' + '\n'.join(el))
            else:
                new_explanation.append(el)
        explanation = new_explanation
        
        if len(self.args.perturbation_type) > 1:
            all_incl_ids = list(p_incl_ids[self.args.perturbation_type[0]].intersection(*[p_incl_ids[pn] for pn in self.args.perturbation_type[1:]]))
        else:
            all_incl_ids = list(p_incl_ids[self.args.perturbation_type[0]])

        
        if len(all_incl_ids) > 0:
            with open(self.args.text_path,'a') as f:
                write_template = '{p_type}: {ex} So the answer is {ans}\n'
                for inc_i in all_incl_ids:
                    ex = explanation[inc_i]
                    curr_choices = choices[inc_i]
                    oa = answer[inc_i]
                    ques = inp_only[inc_i]
                    cf_ques = cf_inp_only[inc_i]
                    try: 
                        ori_ans = curr_choices[oa]
                    except IndexError:
                        ori_ans = oa
                    f.write(f'Question: {ques}\n')
                    f.write(write_template.format_map({'p_type':'original','ex':ex.strip(),'ans':str(ori_ans)}))
                    
                    for pn,pl_s in all_labels.items():
                        pa = pl_s[inc_i]
                        coe = all_explanations[pn][inc_i]
                        try:
                            pred_ans = curr_choices[pa]
                        except IndexError:
                            pred_ans = pa
                        f.write(write_template.format_map({'p_type':pn,'ex':coe,'ans':str(pred_ans)}))
                        if pn == 'cf':
                            f.write(f'Cf Question: {cf_ques}\n')
                            try:
                                cf_ans = curr_choices[cf_answer[inc_i]]
                            except IndexError:
                                cf_ans = cf_answer[inc_i]
                            f.write(f'CF edits:{cf_edits[inc_i]}\nCF original:{cf_originals[inc_i]}\nCF answer: {cf_ans}\n')
                    f.write('-'*100 + '\n')
        
        return test_results
    
    def test_run(self):
        if not self.args.get_expl:
            if 'results' not in self.args.eval_results:
                all_out = defaultdict(default_factory) # for storing all the results
            else:
                all_out = self.args.eval_results['results']
            
            if 'text' not in self.args.eval_results:
                self.args.eval_results['text'] = []
            
            if 'eval_no_count' not in self.args.eval_results:
                self.args.eval_no_count = {p:0 for p in self.args.perturbation_type}
            else:
                self.args.eval_no_count = self.args.eval_results['eval_no_count']
                if len(self.args.eval_no_count.keys()) < 3:
                    for p in self.args.perturbation_type:
                        if p not in self.args.eval_no_count:
                            self.args.eval_no_count[p] = 0
        else:
            all_out = []

        for batch in tqdm(self.loader,desc='Testing',total=len(self.loader)):
            out = self.test_step(batch)
            if self.args.get_expl:
                with open(self.args.expl_path,'a') as fp: # append to file
                    for expl in out:
                        json.dump(expl,fp)
                        fp.write('\n')
            else:
                for o_k,o_v in out.items():
                    for i_k,i_v in o_v.items():
                        all_out[o_k][i_k].append(i_v)
                
                self.args.eval_results['results'] = all_out
                self.args.eval_results['eval_no_count'] = self.args.eval_no_count
                
                for q,c in zip(batch['inp_only'],batch['choices']): # store the keys after a sucessful processing.
                    self.args.eval_results['text'].append(f'{q}.{"".join(c)}')
                
                with open(self.args.eval_results_file,'wb') as f: # always override the eval results file.
                    pickle.dump(self.args.eval_results,f)
            # break
        return all_out
    
    def test_step(self, batch): # insert checking function to check if ans is legit, if not re-generate for a maximum of 3 times.
        prompt = batch['prompt']
        choices = batch['choices']
        answer = batch['gold_ans']
        inp_only = batch['inp_only']
        explanation = batch.get('explanation',None)
        pred_answer = batch.get('pred_answer',None)
        cf_answer = batch.get('cf_answer',None)
        cf_edits = batch.get('cf_edits',None)
        cf_originals = batch.get('cf_originals',None)
        cf_inp_only = batch.get('cf_inp_only',None)
        perturbated_expl = batch.get('perturbated_expl',None)
        # redo_ans = 0 # set to off
        if self.args.get_expl:
            out,redo_data = self.get_explanations(prompt,choices,answer,inp_only)
            if len(redo_data) > 0:
                for rd in range(len(redo_data['prompt'])):
                    out.append({'question': redo_data['inp_only'][rd],'choices':redo_data['choices'][rd],'answer':redo_data['answer'][rd],'explanation':''})
                last_remaining = len(redo_data['prompt'])
                print (f'Still left remaining {last_remaining} samples to redo.')
            
            # REDO
            # while len(redo_data) > 0:
            #     remaining_len = len(redo_data['prompt'])
            #     print (f'Redoing answer generation for {self.args.dataset} for {remaining_len} samples, left {3-redo_ans} tries')
            #     redo_out, redo_data = self.get_explanations(**redo_data)
            #     if len(redo_out) > 0:
            #         out.extend(redo_out)
            #     # redo_ans += 1
            #     # if redo_ans >= 3:
            #         for rd in range(len(redo_data['prompt'])):
            #             out.append({'question': redo_data['inp_only'][rd],'choices':redo_data['choices'][rd],'answer':redo_data['answer'][rd],'explanation':''})
            #         last_remaining = len(redo_data['prompt'])
            #         print (f'Still left remaining {last_remaining} samples to redo.')
            #         break
            return out
        else:
            test_results = self.evaluate(prompt,choices,answer,pred_answer,inp_only,explanation,cf_answer,cf_edits,cf_originals,cf_inp_only,perturbated_expl)
            return test_results

    def tgi_gen(self,inp,return_scores):
        if self.args.num_seq <= 1 and not return_scores:
            return self.client.generate(inp,max_new_tokens = self.args.max_dec_length,do_sample = self.args.sample).generated_text
        else:
            return self.client.generate(inp,
                                        max_new_tokens = self.args.max_dec_length,
                                        # top_p = self.args.top_p,
                                        temperature = self.args.temp,
                                        top_k = self.args.top_k,
                                        best_of = self.args.num_seq,
                                        do_sample= self.args.sample) # dont pass out only the generated text
    
    def batch_tgi_gen(self,inp,return_scores=False):
        num_workers = len(inp)
        tgi_gen_fn = partial(self.tgi_gen,return_scores = return_scores)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            gen_text = list(executor.map(tgi_gen_fn,inp))
        return gen_text
    
    def tgi_generate(self,inp,choices=None,clean=False,is_expl=False,return_scores=False):
        if 'refine' not in self.args.prompt_type or not is_expl: # not explanation dont refine, use normal generate
            gen_text = self.batch_tgi_gen(inp,return_scores)
        else:
            gen_text = self.refine_generate(inp,choices= choices)
            return gen_text # return immediately, no need clean, gen_text is both expl and ans
        # Clean
        if isinstance(gen_text[0],str) or clean: # only clean when it returns a str per sample rather than list (which is the case for sc)
            if not is_expl:
                gen_text = [clean_answer_only(g) for g in gen_text]
            else:
                gen_text = [g.lstrip().split('\n\n')[0].strip() for g in gen_text] # includes expl, throw out the rest, this is used for getting explanation, clean later
        return gen_text
    
    def refine_generate(self,inps,choices=None):
        """
        Given a list of inps, choices, do Self-Refine iterating between feedback and refine after initial generation.
        init -> feedback -> refine -> feedback -> refine -> end
        return list of answers and explanations
        """
        out_ans,out_expl,out_order = [],[],[]
        finished_all = False # flag to check if all samples are done
        curr_ans,curr_feedback = '',''
        curr_process = 'init'
        original_order = list(range(len(inps)))
        post_process  = {'init':process_answer,
                        'feedback':process_feedback,
                        'refine':process_answer}
        for refine_iter in range(self.args.max_refine):
            print (f'Stage: {curr_process}, Iter: {refine_iter}, Left {len(inps)} samples to refine')
            if refine_iter == 0:
                in_feats = [{'question':i,'choices':c} for i,c in zip(inps,choices)]
                curr_inp = [format_refine_prompt(ifeats,stage = 'init',num_shot=self.args.num_shot,dataset=self.args.dataset) for ifeats in in_feats]
            else: # limit to 3 shots due to length
                in_feats = [{'question':i,'choices':c,'feedback':f,'answer':a} for i,c,f,a in zip(inps,choices,curr_feedback,curr_ans)]
                curr_inp = [format_refine_prompt(ifeats,stage = 'refine',num_shot=min(self.args.num_shot,3),dataset=self.args.dataset) for ifeats in in_feats]
                curr_process = 'refine'

            if self.args.tgi:
                curr_ans = self.batch_tgi_gen(curr_inp)
            else:
                curr_ans = self._generate(curr_inp,encoded=False)
            curr_ans,curr_ans_id,curr_expl = post_process[curr_process](curr_ans,self.args.num_choices)

            in_feats = [{'question':i,'choices':c,'answer':a} for i,c,a in zip(inps,choices,curr_ans)]
            feedback_inp = [format_refine_prompt(ifeats,stage = 'feedback',num_shot = self.args.num_shot,dataset=self.args.dataset) for ifeats in in_feats]
            if self.args.tgi:
                curr_feedback = self.batch_tgi_gen(feedback_inp)
            else:
                curr_feedback = self._generate(feedback_inp,encoded=False)
            curr_feedback,curr_stop = post_process['feedback'](curr_feedback) # feedback is list of feed, stop is stop condition
            # filter out those that already stop
            next_inps,next_choices,next_ans,next_feedback,next_order = [],[],[],[],[]
            for i,s in enumerate(curr_stop):
                if not s:
                    next_inps.append(inps[i])
                    next_choices.append(choices[i])
                    next_ans.append(curr_ans[i])
                    next_feedback.append(curr_feedback[i])
                    next_order.append(original_order[i])
                else:
                    out_ans.append(curr_ans_id[i])
                    out_expl.append(curr_expl[i])
                    out_order.append(original_order[i])
            inps,choices,curr_ans,curr_feedback,original_order = next_inps,next_choices,next_ans,next_feedback,next_order
            if len(inps) == 0:
                print (f'All samples are done at iter {refine_iter}')
                finished_all = True
                break
        
        # last iteration, if remaining
        if not finished_all:
            out_ans.extend(curr_ans_id)
            out_expl.extend(curr_expl)
            out_order.extend(original_order)
        
        ## remaining stuff, sort as well.
        all_items = zip(out_ans,out_expl,out_order)
        sort_items = sorted(all_items,key=lambda x:x[2])
        out_ans,out_expl,_ = zip(*sort_items)

        return out_ans,out_expl

    def model_generate(self,inp,is_expl =False,return_scores=False):
        """
        return_scores used as helper flag to return scores for situations where num_seq is 1
        """
        if not is_expl:
            num_seq = 1
            do_sample = False
        else: # only when getting explanation, cot-sc uses beam search
            num_seq = self.args.num_seq
            do_sample = self.args.sample

        out = self.model.generate(inputs=inp,
                                    max_new_tokens = self.args.max_dec_length,
                                    # top_p = self.args.top_p,
                                    temperature = self.args.temp,
                                    top_k = self.args.top_k,
                                    num_return_sequences = num_seq,
                                    do_sample= do_sample,
                                    return_dict_in_generate=True,
                                    output_scores = True)
            
        new_out = defaultdict(list)
        
        if num_seq > 1:
            out_seq = out.sequences.reshape(self.args.eval_batch_size,num_seq,-1)
            new_out['score'] = self.model.compute_transition_scores(out.sequences,out.scores,normalize_logits=True)
        elif return_scores and num_seq <=1:
            out_seq = [[o] for o in out.sequences]
            new_out['score'] = self.model.compute_transition_scores(out.sequences,out.scores,normalize_logits=True)
        else:
            out_seq = [[o] for o in out.sequences]
        
        for ip,outer_o in zip(inp,out_seq):
            for inner_o in outer_o:
                new_out['sequence'].append(inner_o[ip.shape[0]:])

        return new_out
    
    
    def _generate(self,inp,encoded=False,return_scores=False,is_expl=False): # insert checking for max of 3 times on cleaning of generated text. when using model.generate, it includes the input in.
        if self.pipe is not None:
            out = self.pipe(inp)
            out_text = [o[0]['generated_text'] for o in out]
        else:
            if not encoded:
                inp_ids = self.tokenizer(inp,return_tensors='pt',padding='longest', truncation=False).input_ids.cuda()
            else:
                inp_ids = inp
            with torch.no_grad():
                out = self.model_generate(inp_ids,is_expl = is_expl,return_scores=return_scores)
                out_seq = out['sequence']
                out_text = self.tokenizer.batch_decode(out_seq,skip_special_tokens=True)
                if self.args.max_dec_length >= 16: # if too little, wont have to do this anyways
                    out_text = [o.split('\n\n')[0].strip() if '\n\n' in o  else o for o in out_text] # remove the rest of the text

                if 'score' not in out or not return_scores:
                    return out_text
                else:
                    out_scores = out['score']
                    out_text_len = [len(self.tokenizer.encode(o,add_special_tokens = False)) for o in out_text]
                    out_scores = [torch.clamp(o[:l],min=-1e8,max=0) for o,l in zip(out_scores,out_text_len)] # logprobs clamp away the -inf. and take until the length of the text after removing \n\n 
                    if self.args.num_seq > 1: # only when cot-sc uses more than >1 num seq then list them into nested list
                        out_text = [out_text[i:i+self.args.num_seq] for i in range(0,len(out_text),self.args.num_seq)]
                        out_scores = [out_scores[i:i+self.args.num_seq] for i in range(0,len(out_scores),self.args.num_seq)]
                    return out_text,out_scores
    
    def get_explanations(self,prompt,choices,answer=None,inp_only=None,eval=False):
        if self.args.tgi:
            expl_n_ans = self.tgi_generate(prompt,choices=choices,is_expl=True,return_scores = True if self.args.prompt_type in ['cot_sc','cot_sec'] else False)
        else:
            if self.args.prompt_type == 'cot_refine':
                expl_n_ans = self.refine_generate(prompt,choices=choices)
            else:

                expl_n_logprobs = self._generate(prompt,return_scores=True if self.args.prompt_type in ['cot_sc','cot_sec'] else False,is_expl = True)
                if self.args.num_seq > 1:
                    expl,logprobs = expl_n_logprobs
                    expl_n_ans = [(ex,lp) for ex,lp in zip(expl,logprobs)]
                    
                else: # no logprobs just raw text
                    expl_n_ans = expl_n_logprobs 
        out = []
        if self.args.prompt_type == 'cot_refine': # refit the expl and ans to be list of tuples
            ans,expl =expl_n_ans

            expl_n_ans = [(a,e) for a,e in zip(ans,expl)]
        redo_data = defaultdict(list)
        for i,ea in enumerate(expl_n_ans): # each iter
            nd = {}
            if self.args.prompt_type not in  ['cot_sc','cot_sec']: 
                if self.args.prompt_type != 'cot_refine': 
                    if self.args.prompt_type in ['cot_qd','cot_cf']:
                        expl,ans = separate_qd_expl(ea,self.args.prompt_type,self.args.num_choices) # expl is a list, ans is similarly alphabet
                    else:
                        expl,ans = separate_expl(ea) # need to convert ans from alphabet to numerical
                    ans = self.alpha_to_num.get(ans,-1)
                else:
                    ans,expl = ea # already cleaned and ready, a is only -1 if not found in choices
                if ans != -1:
                    nd = {'explanation':expl,'pred_answer':ans,'question':inp_only[i],'choices':choices[i],'answer':answer[i]}
                    out.append(nd)
                else: # add to redo later
                    if eval: # during evaluate, no need save it down, no redo as well
                        out.append({'explanation':expl,'pred_answer':ans})
                    # else:
                    redo_data['prompt'].append(prompt[i])
                    redo_data['choices'].append(choices[i])
                    redo_data['answer'].append(answer[i])
                    redo_data['inp_only'].append(inp_only[i])
            # only for cot-sc , ea is a response object
            else: 
                out,redo_data = self.majority_sc(out,redo_data,ea,prompt,inp_only,choices,answer,i,eval=eval) # use another fn to process self-consistency.
        return out,redo_data
            
    def load_dataloader(self,split='test',data_path = None):
        testset = load_raw_dataset(split, self.args,data_path = data_path,have_expl = not self.args.get_expl,eval = not self.args.get_expl)
        test_collator = TextCollator(self.tokenizer, self.args)
        test_dataloader = DataLoader(testset, collate_fn=test_collator,batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers,shuffle=False,drop_last=False)
        self.loader = test_dataloader
    
    def majority_sc(self,out,redo_data,ea,prompt,inp_only,choices,answer,indx,eval=False):
        """
        Used for both self-consistency and self-entailment correction (ours) for each sample in the batch.
        Difference:
        1) sc takes final explanation based on highest cum probability
        2) self-entailment queries the model to compute entailment score between e and context (Q+yhat) and IOU between (e,Q+yhat) and pick highest winner
        Answer does not change and is based on majority answer. 
        If num seq > num choices, will have more than 1 count of majority answer. (this has to happen for sec to have any difference, else falls back to sc)
        """
        valid_expl,valid_ans,valid_ids = [],[],[]
        if self.args.tgi: # tgi format differently.
            all_ea = [ea.generated_text] + [s.generated_text for s in ea.details.best_of_sequences]
                
            all_log_probs,all_tokens = [],[]
            all_log_probs.append([t.logprob for t in ea.details.tokens]) # add log prob
            # for more than 1 seq> sub sequences
            all_tokens.append([t.text for t in ea.details.tokens]) # add tokens to find explanation subseq and sum up log prob
            for rest_seq in ea.details.best_of_sequences:
                all_log_probs.append([t.logprob for t in rest_seq.tokens])
                all_tokens.append([t.text for t in rest_seq.tokens])
            assert len(all_ea) == len(all_log_probs) == len(all_tokens), 'Length of all_ea, all_log_probs, all_tokens not equal'
        else:
            all_ea,all_log_probs = ea
            
        for j in range(len(all_ea)):
            expl,ans = separate_expl(all_ea[j]) # need to convert ans from alphabet to numerical

            ans = self.alpha_to_num.get(ans,-1)
            if ans != -1: # only add indices if valid answer, after that find the best match seq
                valid_ids.append(j)
                valid_expl.append(expl)
                valid_ans.append(ans)
        # First find majority answer, then gather the ids of those answers,expl and log probs
        if len(valid_ans ) > 0:
            majority_ans = max(set(valid_ans), key = valid_ans.count)
            majority_ans_id = [i for i,v in enumerate(valid_ans) if v == majority_ans]
            majority_expl = [valid_expl[i] for i in majority_ans_id]
            majority_valid_ids = [valid_ids[i] for i in majority_ans_id]
            if len(majority_valid_ids) > 1: 
            # only if there is more than single selection,  we find the best id which = highest logprob of explanation for sc or scoring system for sec
                valid_log_probs = [all_log_probs[k] for k in majority_valid_ids]
                if self.args.prompt_type == 'cot_sc':
                    if self.args.tgi:
                        valid_tokens = [all_tokens[k] for k in majority_valid_ids]
                        if self.args.rank_type == 'max':
                            chosen_expl_id = self.max_cumprobs(valid_log_probs,tokens=valid_tokens) # experiment with both random and max probs
                        else: # chosen a random id
                            chosen_expl_id = random.randint(0,len(majority_expl)-1)
                    else:
                        if args.rank_type == 'max':
                            chosen_expl_id = self.max_cumprobs(valid_log_probs,explanation=majority_expl)
                        else: # chosen a random id
                            chosen_expl_id = random.randint(0,len(majority_expl)-1)
                elif self.args.prompt_type == 'cot_sec': 
                        chosen_expl_id = self.max_entailment(majority_expl,majority_ans,inp_only[indx],choices[indx])  
                else:
                    exit('Wrong prompt type for self-consistency')
            else:
                chosen_expl_id = 0
                
            chosen_expl = majority_expl[chosen_expl_id]
            nd = {'explanation':chosen_expl,'pred_answer':majority_ans,'question':inp_only[indx],'choices':choices[indx],'answer':answer[indx]}
            out.append(nd)
        else:
            if eval: # during evaluate, no need save it down, no redo as well
                out.append({'explanation':expl,'pred_answer':ans})
            # else:
            redo_data['prompt'].append(prompt[indx])
            redo_data['choices'].append(choices[indx])
            redo_data['answer'].append(answer[indx])
            redo_data['inp_only'].append(inp_only[indx])
        return out,redo_data
    
    def max_cumprobs(self,log_probs,tokens=None,explanation=None):
        if tokens is not None:
            assert explanation is None, 'Only one of tokens or explanation should be passed in'
            last_id = [find_subsequence(t) for t in tokens]
        elif explanation is not None:
            last_id = [len(self.tokenizer.encode(t,add_special_tokens = False)) for t in explanation]
        summed_log_probs = torch.tensor([sum(v[:k]) for v,k in zip(log_probs,last_id)])
        
        return torch.argmax(summed_log_probs).item()

    def max_entailment(self,explanation,yhat,context,choices):
        """
        For a batch of num sequences for each sample in a batch
        explanation: str
        yhat: integer index pointing to pos of choices
        context: question inp only. (dont include answer choices)
        1) turn yhat to text format
        2) create entailment prompt input
        3) pass input to model to generate entailment scores (prob of yes or no)
        4) compute iou score between expl and (context+ yhat)
        5) sum scores and pick max
        """
        y_text = choices[yhat]
        
        # Entailment
        if self.args.alignment != 'overlap': # overlap only
            ent_inps = [create_entailment_prompt(context,y_text,e) for e in explanation]
            prev_dec_length = copy(self.args.max_dec_length)
            self.args.max_dec_length = 3 # allow newspace or A:, garbage tokens
            prev_num_seq,sample_mode = self.set_greedy() # need to set greedy to not get num sequence, just need 1.
            if self.args.tgi:
                ent_seq = self.tgi_generate(ent_inps,clean=False,is_expl=True,return_scores=True) # only used when gathering explanations, dont clean
                ent_probs = []
                for ea in ent_seq:
                    added=False
                    ea_logprobs = [t.logprob for t in ea.details.tokens]
                    ea_tokens = [t.text.lower() for t in ea.details.tokens] # set each token to lower
                    for e_ans in ['yes','no']:
                        if e_ans in ea_tokens:
                            log_prob_pos = ea_tokens.index(e_ans)
                            e_prob = np.exp(ea_logprobs[log_prob_pos])
                            if e_ans == 'no':
                                e_prob = 1.0-e_prob
                            ent_probs.append(e_prob)
                            added=True
                            break
                    if not added:
                        ent_probs.append(0.0) # if no yes or no, append 0.0
                    
            else:
                ent_ans,ent_logprobs = self._generate(ent_inps,return_scores=True,is_expl = True)
                ent_probs = [get_probs_by_token(el,ea,self.args,self.tokenizer) for el,ea in zip(ent_logprobs,ent_ans)]
            
            self.args.max_dec_length = prev_dec_length
            self.unset_greedy(prev_num_seq,sample_mode) # set back to prev mode.
        else:
            ent_probs = [0.0 for _ in explanation]

        # Overlap
        if self.args.alignment != 'entailment': # entailment only
            iou_tar = context + ' '+ y_text
            token_iou = [compute_intersection_tokens(e,iou_tar) for e in explanation]
        else:
            token_iou = [0.0 for _ in explanation]
            
        total_score = np.array(ent_probs) + np.array(token_iou)
        return np.argmax(total_score)

    def set_greedy(self):
        """
        Temp set to greedy mode.
        return the old args to reset back
        """
        prev_num_seq = copy(self.args.num_seq)
        prev_sample = copy(self.args.sample)
        self.args.num_seq = 1
        self.args.sample=False
        return prev_num_seq,prev_sample
    
    def unset_greedy(self,num_s,sample_mode):
        self.args.num_seq = num_s
        self.args.sample=sample_mode
    
def main(args, seed):
    # ----------------------------------------------------- #
    # prepare logger
    log_path = os.path.join(args.save_dir, 'train_seed{}.log'.format(seed))
    logger = get_logger("model", log_path)
    logger.info('args: {}'.format(args))
    
    # read config file
    config_path = os.path.join('configs',f'{args.prompt_type}.yaml')
    with open(config_path,'r') as f:
        config = yaml.safe_load(f)
    
    # update args 
    if args.num_seq > 10: # this is the base, if it is more than 10, we use the one specified in the bash script... just for automating the process
        prev_nq = args.num_seq
    else:
        prev_nq = None
    args = vars(args)
    args.update(config)
    args = argparse.Namespace(**args)
    if prev_nq is not None:
        args.num_seq = prev_nq
    
    # text file and expl dir
    text_file = f'text_{seed}.txt'
    args.expl_main_dir = os.path.join('./data',args.dataset,str(args.seed))
    # args.expl_main_dir = os.path.join('./data',args.dataset)
    os.makedirs(args.expl_main_dir,exist_ok=True)
    if '70B' in args.model_name:
        if args.num_shot >= 7:
            args.expl_main_dir = os.path.join(args.expl_main_dir,'cot')
        else:
            args.expl_main_dir = os.path.join(args.expl_main_dir,f'cot_{args.num_shot}')
            args.save_dir += f"_{args.num_shot}"
            os.makedirs(args.save_dir,exist_ok=True)
    else:
        model_size_pattern  = re.compile(r'(\d+B)')
        model_size = model_size_pattern.search(args.model_name)
        if model_size is not None:
            model_size = model_size.group(1)
        else:
            exit ('Wrong model, only support models with B size in the name')
        args.expl_main_dir = os.path.join(args.expl_main_dir,f'cot_{model_size}')
        args.save_dir += f"_{model_size}"
        os.makedirs(args.save_dir,exist_ok=True)
    
    make_dirs(args.expl_main_dir)
    args.expl_subdir = os.path.join(args.expl_main_dir,args.prompt_type)
    
    if args.prompt_type == 'cot_sc':
        args.save_dir += f'_{args.rank_type}'
        args.expl_subdir += f'_{args.rank_type}'
        
    elif args.prompt_type == 'cot_sec':
        args.save_dir += f'_{args.alignment}'
        args.expl_subdir += f'_{args.alignment}'
    
    if prev_nq is not None:
        args.expl_subdir += f'_seq{prev_nq}'
        args.save_dir += f'_seq{prev_nq}'
    
    make_dirs(args.save_dir)
    make_dirs(args.expl_subdir)
    
    text_path = os.path.join(args.save_dir, text_file)
    args.text_path = text_path
    out_file = os.path.join(args.save_dir, 'out_{}.txt'.format(seed))
    eval_results_file = os.path.join(args.save_dir, 'finished_{}.pkl'.format(seed))
    args.eval_results_file = eval_results_file
    args.out_file = out_file
    
    ## Check if perturbations done, if nan or missing, will not be removed from list and assessed.
    args.perturbation_type = ['noisy','paraphrase','cf']
    eval_las=True
    if os.path.exists(out_file):
        with open(out_file,'r') as fp:
            lines = fp.readlines()
            full_len = len(lines)
            for i,out_l in enumerate(lines):
                if out_l.startswith('Perturbation type:'):
                    p_type = out_l.split(':')[1].strip() 
                    if i+1 <= (full_len-1) and p_type in args.perturbation_type:
                        if lines[i+1].split(':')[1].strip() != 'nan':
                            args.perturbation_type.remove(p_type)
                if "LAS mean:" in out_l:
                    eval_las = False
                        
    ## check if generated explanation
    args.get_expl = False
    remaining_ds = []
    args.expl_path = os.path.join(args.expl_subdir,'{}.jsonl'.format(seed))
    if not os.path.exists(args.expl_path):
        args.get_expl = True
        print ('Explanation do not exists, Generating explanation to file at path: ',args.expl_path)
    else:
        source_path = 'data/{}/test.jsonl'.format(args.dataset)
        remaining_ds,args.get_expl = get_remaining_data(source_path,args.expl_path,args.max_ds_size)
        if args.get_expl:
            print (f'Remaining samples for {args.dataset}, {args.prompt_type} to generation explanation: {len(remaining_ds)}')
        else:
            args.expl_path = [os.path.join(args.expl_subdir,f'pertubated_{seed}_final.jsonl'),os.path.join('./data',args.dataset,f'cf_{seed}.jsonl')]
            print (f'Explanation exists, Loading explanation for {args.dataset}, {args.prompt_type} from file at path: ',args.expl_path)
    args.remaining_ds = remaining_ds
    
    ## Ensure stats for prompts are correct
    if args.prompt_type in ['cot_sc' ,'cot_sec']:
        assert args.num_seq >1, 'self consistency requires num_seq > 1'
    else:
        assert args.num_seq == 1, 'other prompts requires num_seq = 1 since greedy mode'
        args.sample = False
        args.num_seq = 1

    starting_t = time()
    tester = LlamaTester(args)

    # If script stops halfway, continue retrieving explanations
    if args.get_expl:
        tester.load_dataloader(split=args.split,data_path = None) # load using split
        if len(tester.loader) == 0: # for some reason, remaining explanations exist
            print ('All explanations gathered')
        else:
            tester.test_run()
            print (f'Finish Getting {args.prompt_type} explanation for {args.dataset} dataset, saving to {args.expl_path}')
        args.get_expl = False
    
    
    ## Load previous evluation keys if exist
    if os.path.exists(eval_results_file) and len(args.perturbation_type) == 3 or len(args.perturbation_type) == 0: 
        # ensure that args.perturbation_type is full if we are not done with evaluation, meaning out_file should not exist, if for some reason file exist before full ds evaluated, del out_file else we will restart the eval all over all.
        # if perturbation_type is less than the full set, we will do full eval for the entire perturbation type. Since TextCollator only takes in the perturbations in the list.
        try:
            args.eval_results = pickle.load(open(eval_results_file,'rb')) # load the keys pointing to instances to carry on evaluation from remaining keys
        except EOFError:
            args.eval_results = {}
    else:
        args.eval_results = {}
    

    ## Evaluation
    eval_path = [os.path.join(args.expl_subdir,f'pertubated_{seed}_final.jsonl'),os.path.join('./data',args.dataset,f'cf_{seed}.jsonl')]
    for ep in eval_path:
        if not os.path.exists(ep):
            exit(f'No eval path exists at {ep}, please generate by running get_eval.sh')
    args.expl_path = eval_path # set it to eval path again
    tester.args = args
    tester.load_dataloader(split=args.split,data_path = args.expl_path)
    if len(tester.loader) == 0 or len(args.perturbation_type)==0:
        print ('All evaluations gathered')
    else:
        print (f'Assess the this list of perturbations: {args.perturbation_type}')
        test_results = tester.test_run()
        print (f'Finish Getting {args.perturbation_type} results for {args.prompt_type} explanation for {args.dataset} dataset')
        if len(args.perturbation_type) != 3: # there are existing result, dont override it. (in any case there is existing results in finished_{seed}.pkl) can always grab from there.
            write_mode = 'a'
        else:
            write_mode = 'w'
        with open(out_file,write_mode) as f:
            for outer_k,outer_v in test_results.items():
                f.write('Perturbation type: {}\n'.format(outer_k))
                for inner_k, inner_v in outer_v.items():
                    inner_v = np.mean(inner_v) * 100
                    f.write('{}: {:.2f}\n'.format(inner_k,inner_v))
            for p_type,p_val in tester.args.eval_no_count.items():
                f.write(f'{p_type} no count: {p_val}')
                
    if eval_las:
        args.get_expl = True
        args.split = 'train'
        args.remaining_ds = []
        tester.args = args
        las_cot(tester,args,seed)

    print ('Time taken: ',time() - starting_t)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--save_dir', '-o', type=str)
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--max_dec_length', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--prompt_type', type=str, default='cot',choices = ['cot','cot_sbs','cot_sc','cot_qd','cot_cf','cot_refine','cot_sec'], 
    help = 'sbs is step-by-step, sec is self, sc is self-consistency, qd is question decomp, cf is counterfactual reasoning, refine is self-refine')
    parser.add_argument('--top_k', type=int, default= 50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--noise_variance', type=float, default=30.)
    # gpu and workers option
    parser.add_argument('--gpu_no', nargs = '+',type=int, required=False, default=1)
    parser.add_argument("--fp_16",  action='store_true',help = "if you want to enable 16-bit training then install apex and set this to true")
    parser.add_argument('--use_pipe', action='store_true',help='use pipeline function')
    parser.add_argument('--gpu_mem',nargs = '+',type=int, required=False, default=16,help = 'gpu memory for each device in gpu_no')
    parser.add_argument('--num_workers',type=int, required=False, default=16,help = 'number of workers, if using tgi, is similar to batch size')
    parser.add_argument('--tgi',action = 'store_true',help = 'use text generation inference')
    parser.add_argument('--port',type = int,default = '8000',help = 'Port used for text generation inference')
    parser.add_argument('--num_seq',type = int,default = 1,help = 'only for self-consistency')
    parser.add_argument('--split',type = str,default = 'test',help = 'to change split during LAS computation, only when simulator data not exist, is train, else always test')
    parser.add_argument('--num_shot', type=int, default=10, help = 'number of shots to use for few-shot, if high number means use all.')
    parser.add_argument('--max_refine', type=int, default=3, help = 'Number of times to refine the explanation')
    ## For LAS score
    parser.add_argument('--simulator_model',default = 't5-base',type = str)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--generative', type=bool, default=True,help = 'always true for cot')
    parser.add_argument('--max_ds_size', type=int, default=2000,help = 'set to max size of 2000 for loading explanations. Test is usually <2000 but train is longer for LAS.')
    parser.add_argument('--rank_type', type=str, default='max',choices=['max','random'],help = 'only random or max , used to select explanation for cot_sc')
    parser.add_argument('--alignment', type=str, default='all',choices=['all','overlap','entailment'],help = 'ablation for each component, all uses both entailment scores and overlap IoU to rank explanations')

    args = parser.parse_args()

    args.same_module = True # same module for all
    if not args.tgi:
        assert len(args.gpu_no) == len(args.gpu_mem), 'gpu_no and gpu_mem must be of same length'
        args.memory_map = {g:f'{gpu_m}GIB' for (g,gpu_m) in zip(args.gpu_no,args.gpu_mem)}
    try:
        args.num_choices = num_choices[args.dataset]
    except:
        print (args.dataset)
        print ('dataset not found')
        raise ValueError
    
    set_seed(args.seed)
    main(args, args.seed)