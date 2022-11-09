python3 src/train_intent.py \
--train_file data/intent/train.json \
--eval_file data/intent/eval.json \
--batch_size 1 \
--learning_rate 3e-5 \
--num_epochs 1 \
--output_dir ckpt/intent \
--seed 2022 \
--device cuda 