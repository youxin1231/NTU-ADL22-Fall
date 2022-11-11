MODEL_NAME="bert-base-uncased"

accelerate launch bonus/src/test_intent.py \
--test_file "${1}" \
--ckpt_dir bonus/ckpt/"${MODEL_NAME##*/}"/intent \
--batch_size 32 \
--output_file "${2}"