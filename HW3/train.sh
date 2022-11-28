if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=186ejZVADY16RBfVjzcMcz9bal9L3inXC
    unzip data.zip && rm data.zip
fi

if [ ! -d ckpt ]; then
    accelerate launch src/train.py \
    --model_name_or_path google/mt5-small \
    --train_file data/train.jsonl \
    --validation_file data/public.jsonl \
    --max_source_length 256 \
    --max_target_length 64 \
    --pad_to_max_length \
    --text_column maintext \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-3 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 4 \
    --output_dir ckpt \
    --seed 2022
fi