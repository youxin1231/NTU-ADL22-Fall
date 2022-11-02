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

accelerate launch run_swag_no_trainer.py \
--train_file data/train.json \
--validation_file data/valid.json \
--context_file data/context.json \
--max_length 512 \
--pad_to_max_length \
--model_name_or_path bert-base-chinese \
--per_device_train_batch_size 1 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--output_dir /tmp/test-swag-no-trainer \
--seed 2022 \

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
