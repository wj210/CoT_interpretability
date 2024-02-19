import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from utils.model_utils import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,get_linear_schedule_with_warmup,AdamW
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import string

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_logger(name, log_path=None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if log_path:
        handler = logging.FileHandler(log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def make_dirs(path):
    os.makedirs(path,exist_ok=True)

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self,):
        super(LabelSmoothingLoss, self).__init__()

    def linear_combination(self, x, y, smoothing):
        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target, smoothing, nll=None):

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1) / n
        if nll is None:
            nll = F.nll_loss(log_preds, target, reduction='none')
        return self.linear_combination(loss, nll, smoothing).mean()


def preprocess_input(r,x,tokenizer):
    et = tokenizer.eos_token_id
    x_out = []
    for _r,_x in zip(r,x):
        r_eos = (_r == et).nonzero().tolist()[0][0]
        _r = _r[:r_eos+1] ## include the eos token
        if _r[0] == tokenizer.pad_token_id: # get rid of pad token id
            _r = _r[1:]
        _x_eos = (_x == et).nonzero().tolist()[0][0]
        _x = _x[:_x_eos]
        x_out.append(torch.cat([_x,_r],dim=-1))
    padded_out = pad_sequence(x_out,batch_first=True,padding_value=tokenizer.pad_token_id)
    mask_out = (padded_out != tokenizer.pad_token_id).float()
    return padded_out, mask_out

def preprocess_all(r,x,a,tokenizer): # r and x are reshaped into (bs, num choices, seq len)
    et = tokenizer.eos_token_id
    x_out = []
    ans_prefix = ' Answer: '
    for r_id,r in enumerate(r):
        curr_x = x[r_id]
        x_eos = (curr_x == et).nonzero().tolist()[0][0]
        curr_x = curr_x[:x_eos]
        for inner_r_id,inner_r in enumerate(r):
            ans_ = torch.tensor(tokenizer.encode(ans_prefix + a[r_id][inner_r_id],add_special_tokens=False)).to(x[0].device)
            if inner_r[0] == tokenizer.pad_token_id: # get rid of pad token id
                inner_r = inner_r[1:]
            r_eos = (inner_r == et).nonzero().tolist()
            if len(r_eos) > 0:
                r_eos = r_eos[0][0]
                inner_r = inner_r[:r_eos] ## exclude the eos token
            else: # try to find pad token instead
                r_pad = (inner_r == tokenizer.pad_token_id).nonzero().tolist()
                if len(r_pad) > 0:
                    r_pad = r_pad[0][0]
                    inner_r = inner_r[:r_pad]
            curr_x = torch.cat([curr_x,inner_r,ans_],dim=-1)
        curr_x = torch.cat([curr_x,torch.tensor([et]).to(x[0].device)],dim=-1)
        x_out.append(curr_x)
    padded_out = pad_sequence(x_out,batch_first=True,padding_value=tokenizer.pad_token_id)
    mask_out = (padded_out != tokenizer.pad_token_id).float()
    return padded_out, mask_out
    
    
def map_labels(x,choices,text_form=False):
    num_choices = len(choices)
    if text_form:
        mapping = {choices[i]:i for i in range(num_choices)}
    else:
        mapping = {chr(ord('a') + i):i for i in range(num_choices)}
    
    if x in mapping.keys():
        return mapping[x]
    else:
        return -1
    
## Adapted from https://github.com/peterbhase/LAS-NL-Explanations
def compute_sim(las_preds):
    for k,v in las_preds.items(): # have to convert to np array first
        if type(k) is not np.ndarray:
            las_preds[k] = np.array(v)
    labels = las_preds['labels']
    xe = las_preds['xe']
    e = las_preds['e']
    x = las_preds['x']    
    xe_correct = np.array(1*(labels==xe)) # find total correct predictions given expl and input
    x_correct = np.array(1*(labels==x)) # find total correct predictions given input
    e_correct = np.array(1*(labels==e)) # find total correct predictions given expl (considered as leaking)
    # baseline and leaking proxy variable
    baseline_correct = 1*(x_correct) # baseline is just the accuracy given input
    leaking = 1*(e_correct)
    leaked = np.argwhere(leaking.tolist()).reshape(-1) # get indices of leaked examples
    
    # get subgroups
    nonleaked = np.setdiff1d(np.arange(len(e_correct)), leaked) # non leaked examples
    if leaked.shape[0] > 0:
        xe_correct_leaked = xe_correct[leaked]
        e_correct_leaked = e_correct[leaked]
        x_correct_leaked = x_correct[leaked]
    else:
        xe_correct_leaked = np.array([0])
        e_correct_leaked = np.array([0])
        x_correct_leaked = np.array([0])
    if nonleaked.shape[0] > 0:
        xe_correct_nonleaked = xe_correct[nonleaked]
        e_correct_nonleaked = e_correct[nonleaked]
        x_correct_nonleaked = x_correct[nonleaked]
    else:
        xe_correct_nonleaked = np.array([0])
        e_correct_nonleaked = np.array([0])
        x_correct_nonleaked = np.array([0])
        
    num_leaked = len(leaked)
    num_non_leaked = len(xe) - num_leaked

    for split in [leaked,nonleaked]:
        las_0_1 = []
        if split.shape[0] > 0:
            las_0_1.append(np.mean(xe_correct[split])- np.mean(baseline_correct[split]))
        else:
            las_0_1.append(0)
        unweighted_mean = np.mean(las_0_1)
    # unweighted_mean = np.mean([np.mean(xe_correct[split]) - np.mean(baseline_correct[split]) for split in [leaked,nonleaked]]) # first computes the LAS1 and LAS0 

    nonleaking_diff = np.mean(xe_correct_nonleaked) - np.mean(baseline_correct[nonleaked]) if nonleaked.shape[0] > 0 else 0
    leaking_diff = np.mean(xe_correct_leaked) - np.mean(baseline_correct[leaked]) if leaked.shape[0] > 0 else 0
    return [unweighted_mean, leaking_diff, nonleaking_diff]
     
def load_optimizer_n_scheduler(args,model,loader):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    ns = len(loader)
    t_total = (ns // args.grad_step) * args.num_epoch
    warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    ) 
    return optimizer, scheduler
    
def clean_state_dict(state_dict,prefix = 'model.'): # if loaded from lightning checkpoint, clear the model prefix
    new_state_dict = {}
    for k,v in state_dict.items():
        if k.startswith(prefix):
            new_k = k.replace(prefix,'')
            new_state_dict[new_k] = v
    return new_state_dict

def get_unique_keys(ds):
    """
    given a list of dict, return a dict mapping unique keys to the data itself.
    the keys are question.choices and data is the dict
    """
    uniq_keys = {}
    for d in ds:
        if 'question' in d:
            q = d['question']
        else:
            q = d['context']
        c = d['choices']
        d_key = f"{q}.{'.'.join(c)}"
        if d_key not in uniq_keys:
            uniq_keys[d_key] = d
        else:
            continue
    return uniq_keys

def detect_words(expls,edits,originals,cot_type='cot'):
    """
    expl are the explanation generated
    edits are the edited words to turn a question into counterfactual
    original are the original words edited.
    Measure the unfaithfulness of the counterfactuals, if not a single edited word inside the explanation, considered unfaithful.
    """
    cf_unfaithfulness = []
    for expl,edit,original in zip(expls,edits,originals):
        # full = False # if need full matching
        if ',' in edit: # during generation of these edits, we use comma to separate non-contiguous words
            edit = edit.replace(',',' ')
        if ',' in original:
            original = original.replace(',',' ')
        edit = re.sub(' +', ' ', edit)
        original = re.sub(' +', ' ', original)
        
        diff_words = set(edit.split()) - set(original.split())
        stop_words = set(stopwords.words('english'))
        cleaned_diff_words = [x for x in diff_words if x not in stop_words] # take away stop words
        if len(cleaned_diff_words) == 0: # means the edit is a stop word, so we use the original word instead
            cleaned_diff_words = list(diff_words)
        
        # if len(cleaned_diff_words) <= 2:
        #     full = True
        
        overlap = False
        if cot_type == 'cot_qd': # only look at answers.
            for ex in expl: # answer
                if ex.startswith('A'):
                    overlap = if_subset(cleaned_diff_words,ex.split())
                    if overlap:
                        break
        else:
            overlap = if_subset(cleaned_diff_words,expl.split())
        cf_unfaithfulness.append(1.0 - float(overlap))
    return np.mean(cf_unfaithfulness)


def if_subset(s1,s2,full=False):
    porter = PorterStemmer()
    if not isinstance(s1,set):
        s1 = set(s1)
    if not isinstance(s2,set):
        s2 = set(s2)
    # reduce words to stem
    s1 = set([remove_punct(porter.stem(w)) for w in s1])
    s2 = set([remove_punct(porter.stem(w)) for w in s2])
    if len(s1) == 0:
        return False
    overlap = (len(s1.intersection(s2))/len(s1))
    
    if full:
        if overlap == 1:
            return True
    else:
        if overlap > 0.0:
            return True

    return False

def remove_punct(s):
    punctuations = re.escape(string.punctuation)
    output_string = re.sub(f'[{punctuations}]', '', s)
    return output_string

def compute_intersection_tokens(t1,t2):
    """
    t1 = target string 
    t2 = comparsion string
    Currently used to compute overlap percentage between a target text sentence and a context sentence (for cot_sec, compare between an explanation and a context (question and answer)
    """
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    t1 = set([remove_punct(porter.stem(t)) for t in t1.split() if t not in stop_words])
    t2 = set([remove_punct(porter.stem(t)) for t in t2.split() if t not in stop_words])
    return len(t1.intersection(t2))/len(t1)/(len(t1.union(t2)))

def get_probs_by_token(logprobs,expl,args,tokenizer):
    """
    Given logprobs of a sequence, return the probability of the token that is the answer to the explanation
    if does not entail, return reciprocal probs. (assuming the majority of probs is assigned to yes/no, others are insignificant)
    """
    reciprocal = False
    if 'yes' in expl:
        token_id = args.yes_token_id
    elif 'Yes' in expl:
        token_id = args.yes_token_id_cap
    elif 'no' in expl:
        token_id = args.no_token_id
        reciprocal = True
    elif 'No' in expl:
        token_id = args.no_token_id_cap
        reciprocal = True
    else:
        print ('Answer not found for nli')
        return 0
    tokenized = tokenizer.encode(expl,add_special_tokens=False)
    token_pos = tokenized.index(token_id)
    prob =  torch.exp(logprobs[token_pos]).item()
    if reciprocal:
        return 1-prob
    else:
        return prob

def get_remaining_data(source_path,target_path,max_num=2000):
    remaining_ds = []
    with open(target_path,'r') as f:
        tar_data = [json.loads(line) for line in f]
        tar_ds = get_unique_keys(tar_data) 
    with open(source_path,'r') as f:
        source_data = [json.loads(line) for line in f]
        source_ds = get_unique_keys(source_data)
        
    remaining_num = max_num - len(tar_ds)
    
    missing_keys = list(set(source_ds.keys()) - set(tar_ds.keys()))
    if len(missing_keys) > 0 and remaining_num > 0: # if less than max num, add the missing keys to collect
        for mk in missing_keys[:remaining_num]:
            remaining_ds.append(source_ds[mk])
        return remaining_ds, True
    else:
        return remaining_ds, False