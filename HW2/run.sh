MODEL_NAME="bert-base-chinese"

python3 src/preprocess.py \
--test_preprocess \
--context_file "${1}" \
--test_file "${2}" \
--output_file data/preprocessed/test_swag.json

python3  src/Multiple_Choice/test_multiple_choice.py \
--model_name_or_path ckpt/"${MODEL_NAME}"/multiple_choice \
--test_file data/preprocessed/test_swag.json \
--pred_file data/multiple_choice_pred.json \
--max_length 512 \
--batch_size 32 \
--device cuda

'''
python3  src/Multiple_Choice/test_question_answering.py \
--model_name_or_path ckpt/"${MODEL_NAME}"/question_answering \
--test_file data/multiple_choice_pred.json \
--pred_file "${3}"
'''