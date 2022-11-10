MODEL_NAME="bert-base-uncased"

if [ ! -d data/origin ]; then
    mkdir -p data/origin
    kaggle competitions download -c intent-classification-ntu-adl-hw1-fall-2022
    unzip intent-classification-ntu-adl-hw1-fall-2022.zip -d data/origin/intent && rm intent-classification-ntu-adl-hw1-fall-2022.zip && rm data/origin/intent/sampleSubmission.csv
    kaggle competitions download -c slot-tagging-ntu-adl-hw1-fall-2022
    unzip slot-tagging-ntu-adl-hw1-fall-2022.zip -d data/origin/slot && rm slot-tagging-ntu-adl-hw1-fall-2022.zip && rm data/origin/slot/sampleSubmission-2.csv
fi

accelerate launch src/train_intent.py \
--model_name_or_path "${MODEL_NAME}" \
--train_file data/origin/intent/train.json \
--validation_file data/origin/intent/eval.json \
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