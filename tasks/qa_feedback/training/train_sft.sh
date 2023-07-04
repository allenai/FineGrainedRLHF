torchrun --nproc_per_node 1 --standalone --nnodes=1 ./sft/run_sft.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --bf16 \
    --num_train_epochs 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --train_file ./tasks/qa_feedback/data/train_1k.json \
    --validation_file ./tasks/qa_feedback/data/dev.json \
    --output_dir ./tasks/qa_feedback/model_outputs/t5-large-1k-train \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=128 \
    --predict_with_generate \
    --generation_max_length 200 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --report_to wandb \
    --metric_for_best_model rougeLsum


# Uncomment the following to train on full training dataset

# torchrun --nproc_per_node 1 --standalone --nnodes=1 ./sft/run_sft.py \
#     --model_name_or_path t5-large \
#     --do_train \
#     --do_eval \
#     --bf16 \
#     --num_train_epochs 10 \
#     --evaluation_strategy epoch \
#     --save_strategy epoch \
#     --train_file ./tasks/qa_feedback/data/train.json \
#     --validation_file ./tasks/qa_feedback/data/dev.json \
#     --output_dir ./tasks/qa_feedback/model_outputs/t5-large-full-train \
#     --overwrite_output_dir \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=128 \
#     --predict_with_generate \
#     --generation_max_length 200 \
#     --save_total_limit 2 \
#     --load_best_model_at_end \
#     --report_to wandb \
#     --metric_for_best_model rougeLsum
