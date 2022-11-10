MODEL_NAME="bert-base-uncased"

python3 src/test_slot.py \
--test_file "${1}" \
--model_name_or_path ckpt/"${MODEL_NAME##*/}"/slot \
--batch_size 100 \
--max_length 50 \
--device cuda \
--pred_file "${2}"