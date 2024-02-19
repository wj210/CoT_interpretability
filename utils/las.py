import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from utils.model_utils import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,get_linear_schedule_with_warmup,AdamW
from utils.data_helper import load_raw_dataset,T5Collator
from utils.data_collator_pinto import Data_Collator_for_Training,SimplerCollator
from torch.utils.data import DataLoader
from collections import defaultdict
import os
from utils.utils import *

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def train_store_fn(args,seed,ckpt_path=None):
    exist_data = 0
    if args.same_module:
        simulator_name = 'same_simulator'
    else:
        simulator_name = 'separate_simulator'
    if args.gen:
        simulator_name += '_gen'
    else:
        simulator_name += '_ngen'
    
    device = torch.device("cuda:{}".format(args.gpu_no[0]) if torch.cuda.is_available() else "cpu")
    for curr_split in ['train','test']:
        simulator_path = os.path.join('data',args.dataset, f'{curr_split}_{simulator_name}.jsonl')
        if not os.path.exists(simulator_path): # load the pretrained task+expl model, generate explanations and outputs to store them.
            print (f'simulator data not found for {curr_split}, generating now')
            print ('-'*80+'\n')
            simulator_ds = []
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
            loaded_dict  = torch.load(ckpt_path)
            state_dict = loaded_dict['state_dict']
            new_state_dict = clean_state_dict(state_dict)
            model.load_state_dict(new_state_dict)
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            split_ds = load_raw_dataset(curr_split, args)
            split_collator = Data_Collator_for_Training(tokenizer,args,dropout_context=args.dropout_context,split='test') # test condition but use train data
            split_loader = DataLoader(split_ds,collate_fn=split_collator,batch_size=args.eval_batch_size,num_workers=8)

            for s_b in tqdm(split_loader,desc='Generating explanations for simulator data',total = len(split_loader)):
                base_inps = s_b['input_ids'].to(device)
                base_mask = s_b['attention_mask'].to(device)
                base_ans_inps = s_b.get('ans_input_ids',None) # if same module, use this and append generated explanation to get answer.
                if base_ans_inps is not None:
                    base_ans_inps = base_ans_inps.to(device)
                curr_raw_data = s_b['raw_data'] # already have question and choices inside, only need answer and explanation for same module
                curr_choices = [x['choices'] for x in curr_raw_data]
                ans_choices = s_b.get('answer_choices',None)
                target_ids = s_b['target_ids'].to(device)
                
                with torch.no_grad(): # all operations use torch_no_grad, no gradients needed
                    if args.same_module: # if same module
                        gen_expl = model.generate(input_ids=base_inps,attention_mask = base_mask, num_beams=args.num_beams, top_p = args.top_p,max_length=args.max_dec_length, early_stopping=True)
                        clearned_expl = [clear_pad(x.tolist(),tokenizer,clear_expl=True) for x in gen_expl]
                        clearned_expl = [clearned_expl[i:i+args.num_choices] for i in range(0,len(clearned_expl),args.num_choices)]
                        for i,expl in enumerate(clearned_expl):
                            curr_raw_data[i]['explanation'] = expl
                    if args.gen: # if generative mode
                        if args.same_module:
                            base_inps,base_mask = preprocess_all(gen_expl.reshape(-1,args.num_choices,gen_expl.shape[-1]),base_ans_inps,ans_choices,tokenizer)
                        pred_out = model.generate(input_ids = base_inps,
                                attention_mask = base_mask,
                                top_p = args.top_p,
                                num_beams = args.num_beams,
                                early_stopping = True,
                                max_length = 32)
                        for pi,p in enumerate(pred_out):
                            pred_l,_ = extract_text_answer(clear_pad(p.tolist(),tokenizer),False,choices = curr_choices[pi])
                            # mapped_label = map_labels(pred_l,ans_choices[pi],take_last)
                            if pred_l == -1: # throw it away, cant get the answer out
                                continue
                            else:
                                # assert isinstance(mapped_label,int), 'mapped label is not int'
                                curr_raw_data[pi]['answer'] = pred_l
                                simulator_ds.append(curr_raw_data[pi])
                    else: 
                        if args.same_module:
                            base_inps,base_mask = preprocess_input(gen_expl,base_ans_inps,tokenizer)
                        pred_out = model(input_ids = base_inps,
                                    attention_mask = base_mask,
                                    labels = target_ids).logits
                        p_logprobs = get_log_probs(pred_out,target_ids,args.num_choices)
                        p_pred = torch.argmax(p_logprobs,dim=-1)
                        for pi,p in enumerate(p_pred):
                            curr_raw_data[pi]['answer'] = p.item()
                            simulator_ds.append(curr_raw_data[pi])
                # if len(simulator_ds) >= 2:
                #     break
                            
            with open(simulator_path,'w') as f:
                for data in simulator_ds:
                    f.write(json.dumps(data))
                    f.write('\n')
            exist_data += 1
            print('Storing simulator data for testing LAS score!, path at {}'.format(simulator_path))
            print ('-'*80+'\n')
        else:
            print ('Simulator {} data already exists, skip generating'.format(curr_split))
            print ('-'*80+'\n')
            exist_data += 1
            
    if exist_data >= 2:
        print ('All simulator data already exists, test simulator model now')
        print ('-'*80+'\n')
        simulator_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
        simulator_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        ckpt_dir = '/'.join(ckpt_path.split('/')[:-1])
        simulator_model_path = os.path.join(ckpt_dir,f'model_{seed}_simulator.ckpt')
        if not os.path.exists(simulator_model_path):
            print ('Simulator model not trained, train simulator model now')
            print ('-'*80+'\n')
            ds = load_raw_dataset('train', args, simulator_name=simulator_name,have_expl = True)
            collator = SimplerCollator(simulator_tokenizer,args,split='train')
            loader = DataLoader(ds,collate_fn=collator,batch_size=args.train_batch_size,num_workers=8)
            optimizer,scheduler = load_optimizer_n_scheduler(args,simulator_model,loader)
            for epoch in range(args.num_epoch):
                for b_no,b in tqdm(enumerate(loader),desc = 'Training simulator model',total = len(loader)):
                    optimizer.zero_grad()
                    input_ids = b['input_ids'].to(device)
                    attention_mask = b['attention_mask'].to(device)
                    target_ids = b['target_ids'].to(device)
                    labels = b['labels'].to(device)
                    logits = simulator_model(input_ids = input_ids,attention_mask = attention_mask,labels = target_ids).logits
                    log_probs = get_log_probs(logits,target_ids,args.num_choices)
                    loss = F.cross_entropy(log_probs,labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            # Save at the end of run
            torch.save(simulator_model.state_dict(),simulator_model_path)
        if os.path.exists(simulator_model_path):
            print ('Simulator model already trained, load it now and test')
            print ('-'*80+'\n')
            sm_state_dict = torch.load(simulator_model_path)
            simulator_model.load_state_dict(sm_state_dict)
            test_ds = load_raw_dataset('test', args, simulator_name=simulator_name,have_expl = True)
            test_collator = SimplerCollator(simulator_tokenizer,args,split='test')
            test_loader = DataLoader(test_ds,collate_fn=test_collator,batch_size=args.eval_batch_size,num_workers=8)
            las_preds = defaultdict(list)
            for test_b in tqdm(test_loader,desc = 'Getting LAS score',total = len(test_loader)):
                input_ids = test_b['input_ids'].to(device)
                attention_mask = test_b['attention_mask'].to(device)
                target_ids = test_b['target_ids'].to(device)
                labels = test_b['labels'].to(device)
                for inp_type in ['xe','e','x']:
                    if inp_type == 'xe':
                        attention_mask = test_b['attention_mask'].to(device)
                    elif inp_type == 'e':
                        attention_mask = test_b['e_mask'].to(device)
                    else:
                        attention_mask = test_b['x_mask'].to(device)
                    logits = simulator_model(input_ids = input_ids,attention_mask = attention_mask,labels = target_ids).logits
                    log_probs = get_log_probs(logits,target_ids,args.num_choices)
                    pred_l = torch.argmax(log_probs,dim=-1).tolist()
                    las_preds[inp_type].extend(pred_l)
                las_preds['labels'].extend(labels.tolist())
            las_results = compute_sim(las_preds)
            las_results = [round(x*100,2) for x in las_results]
            with open(args.out_file,'a')as f:
                f.write('LAS mean: {}, leaking diff: {}, nonleaking diff: {}\n'.format(las_results[0],las_results[1],las_results[2]))
            print ('Finish LAS scores')
            print ('-'*80+'\n')


def las_cot(model,args,seed):
    device = torch.device("cuda:{}".format(args.gpu_no[0]) if torch.cuda.is_available() else "cpu")
    simulator_path = os.path.join(args.expl_subdir,f'train_{seed}.jsonl')
    model.args.expl_path = simulator_path
    if os.path.exists(simulator_path):
        source_path = 'data/{}/train.jsonl'.format(args.dataset)
        args.remaining_ds,args.get_expl = get_remaining_data(source_path,simulator_path,args.max_ds_size)
    print (f'Length of training simulator data to get: {len(args.remaining_ds)}')
    if args.get_expl: 
        print (f'simulator data not found for {args.prompt_type}, generating now')
        print ('-'*80+'\n')
        model.load_dataloader(split='train',data_path = None)
        model.test_run()
        print('Storing simulator data for testing LAS score!, path at {}'.format(simulator_path))
        print ('If using TGI, stop the script to disrupt memory usage!') # the remaining portion do not require tgi to run.
        print ('-'*80+'\n')
        args.get_expl = False
    if not args.get_expl: # already existed, train simulator now, run the script again to test this
        print ('All simulator data already exists, test simulator model now')
        print ('-'*80+'\n')
        simulator_model = AutoModelForSeq2SeqLM.from_pretrained(args.simulator_model).to(device)
        simulator_tokenizer = AutoTokenizer.from_pretrained(args.simulator_model)
        simulator_model_path = os.path.join(args.save_dir,f'model_{seed}_simulator.ckpt')
        if not os.path.exists(simulator_model_path):
            print ('Simulator model not trained, train simulator model now')
            print ('-'*80+'\n')
            ds = load_raw_dataset('train', args,data_path = simulator_path,have_expl = True)
            collator = T5Collator(simulator_tokenizer,args,split='train')
            loader = DataLoader(ds,collate_fn=collator,batch_size=args.train_batch_size,num_workers=8)
            optimizer,scheduler = load_optimizer_n_scheduler(args,simulator_model,loader)
            for epoch in range(args.num_epoch):
                for b_no,b in tqdm(enumerate(loader),desc = 'Training simulator model',total = len(loader)):
                    optimizer.zero_grad()
                    input_ids = b['input_ids'].to(device)
                    attention_mask = b['attention_mask'].to(device)
                    text_labels = b['text_labels'].to(device)
                    loss = simulator_model(input_ids = input_ids,attention_mask = attention_mask,labels = text_labels).loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
            # Save at the end of run
            torch.save(simulator_model.state_dict(),simulator_model_path)
        if os.path.exists(simulator_model_path):
            print ('Simulator model already trained, load it now and test')
            print ('-'*80+'\n')
            sm_state_dict = torch.load(simulator_model_path)
            simulator_model.load_state_dict(sm_state_dict)
            test_ds = load_raw_dataset('test', args, data_path= os.path.join(args.expl_subdir,f'{seed}.jsonl'),have_expl = True)
            test_collator = T5Collator(simulator_tokenizer,args,split='test')
            test_loader = DataLoader(test_ds,collate_fn=test_collator,batch_size=args.eval_batch_size,num_workers=8)
            las_preds = defaultdict(list)
            for test_b in tqdm(test_loader,desc = 'Getting LAS score',total = len(test_loader)):
                input_ids = test_b['input_ids'].to(device)
                attention_mask = test_b['attention_mask'].to(device)
                labels = test_b['labels'].to(device)
                choices = test_b['choices']
                text_labels = test_b['text_labels'].to(device)
                for inp_type in ['xe','e','x']:
                    if inp_type == 'xe':
                        attention_mask = test_b['attention_mask'].to(device)
                    elif inp_type == 'e':
                        attention_mask = test_b['e_mask'].to(device)
                    else:
                        attention_mask = test_b['x_mask'].to(device)

                    pred_l = simulator_model.generate(input_ids = input_ids,
                            attention_mask = attention_mask,
                            max_length = 64)
                    pred_l = simulator_tokenizer.batch_decode(pred_l,skip_special_tokens=True)
                    all_pred_l = []
                    for pl,choice in zip(pred_l,choices):
                        pred_l,_ = extract_text_answer(pl,False,choices = choice)
                        all_pred_l.append(pred_l)
                    pred_l = all_pred_l

                    las_preds[inp_type].extend(pred_l)
                las_preds['labels'].extend(labels.tolist())
                
            las_results = compute_sim(las_preds)
            test_acc = (np.array(las_preds['xe']) == np.array(las_preds['labels'])).mean()
            las_results.append(test_acc)
            las_results = [round(x*100,2) for x in las_results]
            with open(args.out_file,'a')as f:
                f.write('\nLAS mean: {}, leaking diff: {}, nonleaking diff: {}, test accuracy: {}\n'.format(las_results[0],las_results[1],las_results[2],las_results[3]))
            print ('Finish LAS scores')
            print ('-'*80+'\n')
        
    
    