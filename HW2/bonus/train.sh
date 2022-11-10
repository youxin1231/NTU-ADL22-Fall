MODEL_NAME="bert-base-uncased"

if [ ! -d data ]; then
    mkdir data
    kaggle competitions download -c intent-classification-ntu-adl-hw1-fall-2022
    unzip intent-classification-ntu-adl-hw1-fall-2022.zip -d data/intent && rm intent-classification-ntu-adl-hw1-fall-2022.zip && rm data/intent/sampleSubmission.csv
    kaggle competitions download -c slot-tagging-ntu-adl-hw1-fall-2022
    unzip slot-tagging-ntu-adl-hw1-fall-2022.zip -d data/slot && rm slot-tagging-ntu-adl-hw1-fall-2022.zip && rm data/slot/sampleSubmission-2.csv
fi

if [ ! -d ckpt/"${MODEL_NAME##*/}"/intent ]; then
    accelerate launch src/train_intent.py \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file data/intent/train.json \
    --validation_file data/intent/eval.json \
    --max_length 50 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --num_train_epochs 8 \
    --gradient_accumulation_steps 2 \
    --output_dir ckpt/"${MODEL_NAME##*/}"/intent \
    --seed 2022
fi

# if [ ! -d ckpt/"${MODEL_NAME##*/}"/slot ]; then
    accelerate launch src/train_slot.py \
    --model_name_or_path "${MODEL_NAME}" \
    --train_file data/slot/train.json \
    --validation_file data/slot/eval.json \
    --max_length 50 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --weight_decay 1e-2 \
    --num_train_epochs 8 \
    --gradient_accumulation_steps 2 \
    --output_dir ckpt/"${MODEL_NAME##*/}"/slot \
    --seed 2022
# fi