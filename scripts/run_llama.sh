#!/bin/bash

project_dir='.'

model_name="TheBloke/Llama-2-70B-chat-GPTQ"
max_dec_length=64
eval_batch_size=16 # set the batch size according to number of max request in tgi.sh if using tgi, if num_shot > 1, total batch size = eval_batch_size * num_shot. If not TGI, just depends on the gpu memory.
prompt_type='cot_sec' # specify the cot-type here
port=8078 # MAKE SURE THE PORT IS SAME NUMBER AS THE PORT GIVEN IN tgi.sh !! (if using TGI, --tgi flag) else ignore (if using standard, best set gpu_no to 4 gpus , ie 0,1,2,3 since model is huge, and use torchrun )
num_shot=10 # Only set to 10 for cot_sec and cot_sc, else wasting resources

## Note that the below script consist of 3 steps 
# 1). Generate base explanations and outputs 
# 2). Generate perturbations (add mistakes, paraphrase, counterfactuals)
# 3). Eval the explanations on the perturbations as well as LAS (same as step 1, just that step 1 terminates when the pertubations are not found)
# Comment out if running all 3 together (1 single script gives u the results saved under save_dir specified below.)

# IF using tgi, run tgi.sh first and once connected, run the rest of the script, make sure port is similar in both scripts
for prompt_type in cot  # specify the cot-type here
do
    for dataset in obqa # specify dataset
    do
        save_dir="${project_dir}/checkpoints/${dataset}/llama_${prompt_type}"
        mkdir -p $save_dir

        python llama_run.py \
        --dataset $dataset \
        --save_dir $save_dir \
        --model_name $model_name \
        --max_dec_length $max_dec_length \
        --eval_batch_size $eval_batch_size \
        --prompt_type $prompt_type \
        --num_workers $eval_batch_size \
        --num_shot $num_shot \
        --tgi \
        --port $port \
        --gpu_no 0 \
        --alignment all \

        # python openai_gen.py \
        # --dataset $dataset \
        # --batch_size 16 \
        # --perturbation_types add_mistakes para \
        # --cot_type $prompt_type \
        # --num_shot 10 \
        # --model_size 70 \
        # --path_suffix _all_seq$num_seq

        # python llama_run.py \
        # --dataset $dataset \
        # --save_dir $save_dir \
        # --model_name $model_name \
        # --max_dec_length $max_dec_length \
        # --eval_batch_size $eval_batch_size \
        # --prompt_type $prompt_type \
        # --num_workers $eval_batch_size \
        # --num_shot $num_shot \
        # --tgi \
        # --port $port \
        # --gpu_no 0 \
        # --alignment all \
        # --num_seq $num_seq
    
    done
done

