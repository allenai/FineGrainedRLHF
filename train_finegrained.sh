accelerate launch \
    --main_process_port 29500 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 4 \
    --mixed_precision bf16 \
    --multi_gpu \
    train_finegrained.py --config fine_grained_config.yml