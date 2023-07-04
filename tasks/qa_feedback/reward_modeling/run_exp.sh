set -e

n_gpus=2

# train reward model for NF-ERR_subsentence
torchrun --nproc_per_node 2 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/NF-ERR_subsentence/train.json \
                --validation_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --test_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/NF-ERR_subsentence \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 12 \
                --per_device_eval_batch_size 12 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.01 \
                --warmup_ratio 0.1


# train reward model for F-ERR_sentence
torchrun --nproc_per_node 2 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/F-ERR_sentence/train.json \
                --validation_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --test_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/F-ERR_sentence \
                --do_train \
                --do_eval \
                --do_predict \
                --bf16 \
                --num_train_epochs 50 \
                --per_device_train_batch_size 12 \
                --per_device_eval_batch_size 12 \
                --evaluation_strategy epoch \
                --logging_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end \
                --metric_for_best_model overall_accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.000005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1

# test inference
torchrun --nproc_per_node 2 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path  ./tasks/qa_feedback/model_outputs/F-pretrained \
                --validation_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --test_file ./tasks/qa_feedback/data/F-ERR_sentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/F-pretrained \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 12 \
                --max_seq_length 2048

# train reward model for COMP
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --validation_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/comp \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 6 \
                --per_device_eval_batch_size 6 \
                --eval_steps 200 \
                --evaluation_strategy steps \
                --logging_steps 200 \
                --logging_strategy steps \
                --save_steps 200 \
                --save_strategy steps \
                --load_best_model_at_end \
                --metric_for_best_model accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.00005 \
                --weight_decay 0.001 \
                --warmup_ratio 0.1 \
                --remove_unused_columns False

# train reward model for baseline
torchrun --nproc_per_node 4 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path allenai/longformer-base-4096 \
                --train_file ./tasks/qa_feedback/data/train_feedback.json \
                --validation_file ./tasks/qa_feedback/data/dev_feedback.json \
                --test_file ./tasks/qa_feedback/data/dev_feedback.json \
                --output_dir ./tasks/qa_feedback/model_outputs/baseline \
                --do_train \
                --do_eval \
                --bf16 \
                --max_steps 6000 \
                --per_device_train_batch_size 6 \
                --per_device_eval_batch_size 6 \
                --eval_steps 200 \
                --evaluation_strategy steps \
                --logging_steps 200 \
                --logging_strategy steps \
                --save_steps 200 \
                --save_strategy steps \
                --load_best_model_at_end \
                --metric_for_best_model accuracy \
                --max_seq_length 2048 \
                --report_to wandb \
                --save_total_limit 2 \
                --learning_rate 0.00001 \
                --weight_decay 0.01 \
                --warmup_ratio 0.1 \
                --remove_unused_columns False

# inference for getting mean std of COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path ./tasks/qa_feedback/model_outputs/baseline-pretrained \
                --validation_file ./tasks/qa_feedback/data/train_feedback.json \
                --test_file ./tasks/qa_feedback/data/train_feedback.json \
                --output_dir ./tasks/qa_feedback/model_outputs/baseline-pretrained \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 128 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True


# inference for getting mean std of baseline
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path ./tasks/qa_feedback/model_outputs/comp-pretrained \
                --validation_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --test_file ./tasks/qa_feedback/data/COMP_sequence/train.json \
                --output_dir ./tasks/qa_feedback/model_outputs/comp-pretrained \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 128 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True