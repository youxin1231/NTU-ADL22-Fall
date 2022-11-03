# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
if [ ! -d data/origin ]; then
    mkdir -p data/origin
    wget https://www.dropbox.com/s/ou7a07f2af2kxrf/ntu-adl-hw2-fall-2022.zip?dl=1 -O data.zip
    unzip data.zip -d data/origin && rm data.zip
fi

python3 preprocess.py \
--data_dir data/origin \
--output_dir data/preprocessed

accelerate launch src/Multiple_Choice/run_multiple_choice.py \
--train_file data/preprocessed/train_swag.json \
--validation_file data/preprocessed/valid_swag.json \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path bert-base-chinese \
--per_device_train_batch_size 1 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--output_dir tmp/test-swag-no-trainer \
--seed 2022

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

accelerate launch run_qa_no_trainer.py \
--train_file data/preprocessed/train_squad.json \
--validation_file data/preprocessed/valid_squad.json \
--max_seq_length 512 \
--pad_to_max_length \
--model_name_or_path bert-base-chinese \
--per_device_train_batch_size 1 \
--learning_rate 3e-5 \
--num_train_epochs 3 \
--gradient_accumulation_steps 2 \
--output_dir tmp/debug_squad \
--seed 2022

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