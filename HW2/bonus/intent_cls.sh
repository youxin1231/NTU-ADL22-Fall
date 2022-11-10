MODEL_NAME="bert-base-uncased"

accelerate launch src/test_intent.py \
--test_file data/origin/intent/test.json \
--ckpt_dir ckpt/"${MODEL_NAME##*/}"/intent \
--batch_size 32 \
--output_file intent_result.csv