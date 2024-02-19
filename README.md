# How Interpretable are Reasoning Explanations from Prompting Large Language Models?


**Install requirements**
`pip install -r requirements.txt`

dataset is derived from [PINTO](https://github.com/wangpf3/pinto-faithful-language-reasoning)

This project is based on 3 steps

**1) Generating base CoT reasoning chains**
- Run `scripts/run_llama.sh`, `llama_run.py` gets the base explanation first and terminates when step 2 is not fufilled.
- Set prompt_type to [`cot`, `cot_sc`, `cot_qd`, `cot_refine`, `cot_sec`], where `cot_sec` is the proposed entailment alignment approach
- Set dataset to [`obqa`, `qasc` `strategyqa`]

**2) Get modified CoT chains for testing**
- After getting base explanation saved in `data/$seed/cot/$prompt_type/$seed.jsonl`, run `scripts/get_eval.sh`, set `OPENAI_API_KEY` to your api key
- `perturbation_types` set it to add_mistakes and para for each different cot_type, which is equals to $prompt_type, ie it generates the perturbations according to the CoT type used
- when `perturbation_types` is set to cf, it generates for each dataset, where it is shared across all cot_type, since the counterfactuals are on the input question and answer and not on the individual cot types.
- GPT3.5 is used for mistakes and para while GPT4 for cf.

**3) Run evaluation on CoT reasoning chains**
- Run the same command in step 1, this time round, when the perturbated datasets are detected, it will run the evaluation portion in the code. LAS do not require any perturbated explanations, just need a student model, we use T5-base.
- The results are in `checkpoints/$dataset/$prompt_type/out_$seed.txt
- In any event, when the evaluation gets interrupted, the prior results are saved in `finished_$seed`.pkl, such that when you run the code again, it filters out all evaluated samples, this is to save resources and prevent restarting from the beginning.
- Important note, the script first check out file for any results, if in any event, the results file exist and it contains incomplete results, ie there are completed results for paraphrase, but not add_mistakes and cf, you should delete the entire results file before running, else the script reruns the evaluation for all 3 perturbations. If this is not a concern, then ignore it. LAS is not affected by this, as it does not have any saved results in `finished_$seed`.pkl.

**Using TGI or AutoGPTQ**
- Using TGI is much faster as compared to standard generation from `model.generate`, install TGI from [Text-generation-inference](https://github.com/huggingface/text-generation-inference), this codebase uses TGI in local mode, though you can easily use docker, just modify `scripts/tgi.sh`.
- **Remember to first run `tgi.sh` before running `run_llama.sh` and set the port correctly.
- If using GPTQ, install it via [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), in which case, you can ignore tgi commands, but use `torchrun` in the main script and set it to number of gpus required by memory, since it is quantized, 2 >40GB should be sufficient.

**Notes**
- The generate hps are in `configs` according to each cot_type, main difference is in num_seq for cot_sc and cot_sec.
- Templates for few-shot are in `template`, where `refine_template` are for self-refine, `prompt_template` is for the rest and `perturbation_template` are for step 2.
- Main model used is `TheBloke/Llama-2-70B-chat-GPTQ` from huggingface. Any usage installations can also be referenced from [Llama-GPTQ](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ)



