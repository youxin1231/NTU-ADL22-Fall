MODEL_NAME="bert-base-uncased"

python3 bonus/src/test_slot.py \
--test_file "${1}" \
--model_name_or_path bonus/ckpt/"${MODEL_NAME##*/}"/slot \
--batch_size 100 \
--max_length 128 \
--device cuda \
--pred_file "${2}"