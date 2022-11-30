CUDA_VISIBLE_DEVICES=0 python3 src/test.py \
    --model_name_or_path ckpt \
    --test_file "${1}" \
    --max_source_length 256 \
    --max_target_length 64 \
    --pad_to_max_length \
    --text_column maintext \
    --batch_size 64 \
    --output_file "${2}" \
    --num_beams 4 \
    # --do_sample \
    # --top_k 50 \
    # --top_p 0.1 \
    # --temperature 0.5\