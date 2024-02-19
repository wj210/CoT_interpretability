#!/bin/bash
dataset=qasc

# num shot - just for naming purpose
# batchsize - number of parallel api calls
# cot_type - type of prompt (only need for mistakes and para)
# model_size - model size (naming purpose)

# Data is saved under data/$dataset/cot/$cot_type/perturbated_$seed

python openai_gen.py \
--dataset $dataset \
--batch_size 1 \
--perturbation_types add_mistakes \
--cot_type cot_sc \
--num_shot 10 \
--model_size 70

## For generating cf, the cot_type does not matter, just any will do

python openai_gen.py \
--dataset $dataset \
--cot_type cot \
--batch_size 2 \
--perturbation_types cf
