MULTIPLE_CHOICE_MODEL_NAME="hfl/chinese-roberta-wwm-ext"
QUESTION_ANSWERING_MODEL_NAME="hfl/chinese-roberta-wwm-ext"

if [ ! -f data/preprocessed/test_swag.json ]; then
python3 src/preprocess.py \
--test_preprocess \
--context_file "${1}" \
--test_file "${2}" \
--output_file data/preprocessed/test_swag.json
fi

python3  src/Multiple_Choice/test_multiple_choice.py \
--model_name_or_path ckpt/"${MULTIPLE_CHOICE_MODEL_NAME##*/}"/multiple_choice \
--test_file data/preprocessed/test_swag.json \
--pred_file data/multiple_choice_pred.json \
--max_length 512 \
--batch_size 64 \
--device cuda

python3  src/Question_Answering/test_question_answering.py \
--model_name_or_path ckpt/"${QUESTION_ANSWERING_MODEL_NAME##*/}"/question_answering \
--test_file data/multiple_choice_pred.json \
--pred_file "${3}" \
--max_length 512 \
--batch_size 64 \
--device cuda
