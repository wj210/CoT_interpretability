import json
import os
import numpy as np
from collections import defaultdict

dataset = 'strategyqa'
cots = ['cot','cot_qd','cot_refine','cot_sc','cot_sec']
seed = 41
num_shot = [1,3]

  
for ms in num_shot:
    all_results = {}  
    for cot in cots:
        if ms < 10:
            out_key = f'llama_{cot}_{str(ms)}'
        else:
            out_key = f'llama_{cot}'

        result_path = f'checkpoints/{dataset}/{out_key}/out_{seed}.txt'

        result_keys = {'original': 'acc','paraphrase':'label_flip','noisy':'label_flip','cf':'unfaithfulness'}
        result_direction = {'paraphrase':-1,'noisy':1,'cf':-1,'LAS':1}
        results = {}

        start = False
        start_key = ''
        with open(result_path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if not start  and start_key == '':
                    for k in result_keys.keys():
                        if k in l:
                            start = True
                            start_key = k
                            break
                else:
                    if result_keys[start_key] in l:
                        results[start_key] = float(l.split(':')[-1].strip())
                        start = False
                        start_key = ''
                if 'LAS' in l:
                    results['LAS'] = float(l.split('mean:')[-1].split(',')[0].strip())
        all_results[out_key] = results

    if len(all_results.keys()) <5:
        for res_k,res_v in all_results.items():

            print_order = ['original','paraphrase','cf','noisy','LAS']

            overleaf_str = []
            for k in print_order:
                overleaf_str.append(np.round(res_v[k],2))
            print (res_k)
            print (' & '.join([str(x) for x in overleaf_str]))
    else:
        normalized_r =defaultdict(list)
        first_k = list(all_results.keys())[0]
        for inner_k in all_results[first_k].keys():
            if inner_k =='original':
                continue
            dir = result_direction[inner_k]
            if dir == -1:
                curr_r = [100-all_results[o_k][inner_k] for o_k in all_results.keys()]
            else:
                curr_r = [all_results[o_k][inner_k] for o_k in all_results.keys()]
            # print (curr_r)
            max_r = np.max(curr_r)
            min_r = np.min(curr_r)
            # print (max_r,min_r)
            for outer_k in all_results.keys():
                if dir == -1:
                    # norm_r = ((100-all_results[outer_k][inner_k]) - min_r)/(max_r - min_r)
                    norm_r = (100-all_results[outer_k][inner_k])
                else:
                    norm_r = all_results[outer_k][inner_k]
                    # norm_r = (all_results[outer_k][inner_k] - min_r)/(max_r - min_r)
                normalized_r[outer_k].append(norm_r)

        for nk,nv in normalized_r.items():
            # print (nk,nv)
            # print (f"{nk}: {np.round(np.mean(nv)*100,2)}")
            faithfulness = np.round(nv[-2],2)
            utility = np.round(nv[-1],2)
            print (f"{nk}: faithfulness: {faithfulness} & utility: {utility}")
        


