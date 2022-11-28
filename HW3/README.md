## Environment
```shell
pip install -e tw_rouge
pip install -r requirements.txt
```

## Download
```shell
# To download the fine-tuned models for reproducing.
bash download.sh
```

## Reproduce
### (Public: 0.78481)
```shell
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Summarization (Title Generation)

### Train
```shell
accelerate launch src/run.py \
--model_name_or_path <model_name> \
--train_file <train_file> \
--validation_file <valid_file> \
--max_source_length <max_source_len> \
--max_target_length <max_target_len> \
--pad_to_max_length \
--text_column <text_column> \
--summary_column <summary_column> \
--per_device_train_batch_size <batch_size> \
--per_device_eval_batch_size <batch_size> \
--learning_rate <lr> \
--num_train_epochs <num_epoch> \
--gradient_accumulation_steps <gradient_acc> \
--output_dir <output_dir> \
--seed <seed>

```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **max_source_len:** Length limit of sample input texts.
* **max_target_len:** Length limit of sample output title.
* **pad_to_max:** Training with padding to max length.
* **text_column:** The name of the column in the datasets containing the full texts.
* **summary_column:** The name of the column in the datasets containing the summaries.
* **batch_size:** Number of samples in one batch.
* **lr:** Learning rate.
* **num_epoch:** Number of epochs.
* **gradient_acc:** Gradient accumulation steps.
* **output_dir:** Directory to the output checkpoint.
* **seed:** Random seed number.

#### Hyperparameters:
|max_source_len|max_target_len|pad_to_max|batch_size|lr|num_epoch|gradient_acc|seed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|256|64|True|4|1e-3|3|4|2022|


### Test
```shell
python3  src/test.py \
--model_name_or_path <model_path> \
--test_file <test_file> \
--pred_file <pred_file> \
--max_length <max_len> \
--batch_size <batch_size> \
--device <device>
```
* **model_path:** Path to the checkpoint directory.
* **test_file:** Path to test file.
* **pred_file:** Path to output file.
* **max_len:** Length limit of sample text string.
* **batch_size:** Number of samples in one batch.
* **device:** Device to run the testing.