MODEL_NAME="bert-base-uncased"

accelerate launch src/test_intent.py \
--test_file "${1}" \
--ckpt_dir ckpt/"${MODEL_NAME##*/}"/intent \
--batch_size 32 \
--output_file "${2}"