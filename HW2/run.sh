MODEL_NAME = "bert_base_chinese"

python3 src/preprocess.py \
--test_preprocess \
--context_file "${1}" \
--test_file "${2}" \
--output_file data/preprocessed/test_swag.json

python3  src/Multiple_Choice/run_multiple_choice.py \
--model_name_or_path ckpt/"${MODEL_NAME}"/multiple-choice \
--test_file data/preprocessed/test_swag.json
--output_file data/multiple_choice_pred.json \


python3  src/Multiple_Choice/run_multiple_choice.py \
--model_name_or_path ckpt/"${MODEL_NAME}"/question_answering \
--test_file data/multiple_choice_pred.json
--output_file "${3}" \