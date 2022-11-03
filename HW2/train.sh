MODEL_NAME="bert-base-chinese"

if [ ! -d data/origin ]; then
    mkdir -p data/origin
    wget https://www.dropbox.com/s/ou7a07f2af2kxrf/ntu-adl-hw2-fall-2022.zip?dl=1 -O data.zip
    unzip data.zip -d data/origin && rm data.zip
fi

if [ ! -d data/preprocessed ]; then
    python3 src/preprocess.py \
    --data_dir data/origin \
    --output_dir data/preprocessed
fi

if [ ! -d ckpt/"${MODEL_NAME}"/multiple_choice ]; then
    accelerate launch src/Multiple_Choice/run_multiple_choice_no_trainer.py \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file data/preprocessed/train_swag.json \
    --validation_file data/preprocessed/valid_swag.json \
    --max_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --output_dir ckpt/"${MODEL_NAME}"/multiple_choice \
    --seed 2022
fi

# --dataset_name \
# --dataset_config_name \

# --config_name \
# --tokenizer_name \
# --use_slow_tokenizer \

# --per_device_eval_batch_size \

# --weight_decay \

# --max_train_steps \

# --lr_scheduler_type \
# --num_warmup_steps \

# --model_type \
# --debug \
# --hub_model_id \
# --hub_token \
# --checkpointing_steps \
# --resume_from_checkpoint \
# --with_tracking \
# --report_to \

# if [ ! -d ckpt/"${MODEL_NAME}"/question_answering ]; then
    accelerate launch src/Question_Answering/run_question_answering_no_trainer.py \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file data/preprocessed/train_squad.json \
    --validation_file data/preprocessed/valid_squad.json \
    --max_seq_length 512 \
    --pad_to_max_length \
    --per_device_train_batch_size 1 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --output_dir ckpt/"${MODEL_NAME}"/question_answering \
    --seed 2022
# fi

# --dataset_name \
# --dataset_config_name \

# --preprocessing_num_workers
# --do_predict \
# --test_file \

# --config_name \
# --tokenizer_name \
# --use_slow_tokenizer \

# --per_device_eval_batch_size \

# --weight_decay \

# --max_train_steps \

# --lr_scheduler_type \
# --num_warmup_steps \

# --doc_stride \
# --n_best_size \
# --null_score_diff_threshold \
# --version_2_with_negative \
# --max_answer_length \
# --max_train_samples \
# --max_eval_samples \
# --overwrite_cache \
# --max_predict_samples \
# --model_type \
# --push_to_hub \
# --hub_token \
# --checkpointing_steps \
# --resume_from_checkpoint \
# --with_tracking \
# --report_to \
