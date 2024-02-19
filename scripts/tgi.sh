REVISION=main
max_input_length=2000  # input length
max_total_length=2500 # output length
port=8078
master_port=29401 
mem_frac=0.4
num_seq=1 # set to 10 for sc-cot or sea-cot
model=mistralai/Mistral-7B-Instruct-v0.1 
sharded=false
requests=250 # num parallel requests
num_gpu=2

export CUDA_VISIBLE_DEVICES=5,6

# if using local
text-generation-launcher --model-id $model --num-shard $num_gpu --port $port --max-input-length $max_input_length --master-port $master_port --cuda-memory-fraction $mem_frac --max-best-of $num_seq --sharded $sharded --max-total-tokens $max_total_length --disable-custom-kernels --max-concurrent-requests $requests

# docker
# docker run --gpus 2 --shm-size 1g -p $port:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.3 --num-shard 4 --sharded $sharded --model-id $model --quantize gptq --revision $REVISION --max-input-length $max_input_length --master-port $master_port --cuda-memory-fraction $mem_frac --max-best-of $num_seq 