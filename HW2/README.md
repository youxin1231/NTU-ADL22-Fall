## Environment
```shell
pip install -r requirements.txt
```
## Download
```shell
# To download the fine-tuned models for reproducing.
bash download.sh
```

## Reproduce
### (Public: 0.78481, Private: 0.79765)
```shell
bash run.sh /path/to/context.json /path/to/test.json /path/to/pred.csv
```

## MultipleChoice

### Train
```shell
accelerate launch src/Multiple_Choice/train_multiple_choice.py \
--model_name_or_path <model_name> \
--train_file <train_file> \
--validation_file <valid_file> \
--max_length <max_len> \
--pad_to_max_length \
--per_device_train_batch_size <batch_size> \
--learning_rate <lr> \
--num_train_epochs <num_epoch> \
--gradient_accumulation_steps <gradient_acc> \
--output_dir <output_dir> \
--seed <seed>

```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **max_len:** Length limit of sample text string.
* **pad_to_max:** Training with padding to max length.
* **batch_size:** Number of samples in one batch.
* **lr:** Learning rate.
* **num_epoch:** Number of epochs.
* **gradient_acc:** Gradient accumulation steps.
* **output_dir:** Directory to the output checkpoint.
* **seed:** Random seed number.

#### Hyperparameters:
|max_len|pad_to_max|batch_size|lr|num_epoch|gradient_acc|seed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|512|True|4|3e-5|10|4|2022|


### Test
```shell
python3  src/Multiple_Choice/test_multiple_choice.py \
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

## Question Answering

### Train
```shell
accelerate launch src/Question_Answering/train_question_answering.py \
--model_name_or_path <model_name> \
--train_file <train_file> \
--validation_file <valid_file> \
--max_seq_length <max_len> \
--pad_to_max_length \
--per_device_train_batch_size <batch_size> \
--learning_rate <lr> \
--num_train_epochs <num_epoch> \
--gradient_accumulation_steps <gradient_acc> \
--output_dir <output_dir> \
--seed <seed>

```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **max_len:** Length limit of sample text string.
* **pad_to_max:** Training with padding to max length.
* **batch_size:** Number of samples in one batch.
* **lr:** Learning rate.
* **num_epoch:** Number of epochs.
* **gradient_acc:** Gradient accumulation steps.
* **output_dir:** Directory to the output checkpoint.
* **seed:** Random seed number.

#### Hyperparameters:
|max_len|pad_to_max|batch_size|lr|num_epoch|gradient_acc|seed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|512|True|4|3e-5|10|4|2022|


### Test
```shell
python3  src/Multiple_Choice/test_question_ansering.py \
--model_name_or_path <model_path> \
--test_file <test_file> \
--pred_file <pred_file> \
--max_length <max_len> \
--batch_size <batch_size> \
--device <device>
```
* **model_name:** Path to the checkpoint directory.
* **test_file:** Path to test file.
* **pred_file:** Path to output file.
* **max_len:** Length limit of sample text string.
* **batch_size:** Number of samples in one batch.
* **device:** Device to run the testing.

## Bonus

## Intent Classification

### Train
```shell
accelerate launch bonus/src/train_intent.py \
--model_name_or_path <model_name> \
--train_file <train_file> \
--validation_file <valid_file> \
--max_length <max_len> \
--pad_to_max_length \
--per_device_train_batch_size <batch_size> \
--per_device_eval_batch_size <batch_size> \
--learning_rate <lr> \
--weight_decay <weight_decay> \
--num_train_epochs <num_epoch> \
--gradient_accumulation_steps <gradient_acc> \
--output_dir <output_dir> \
--seed <seed>
```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **max_len:** Length limit of sample text string.
* **pad_to_max:** Training with padding to max length.
* **batch_size:** Number of samples in one batch.
* **lr:** Learning rate.
* **weight_decay:** Weight deacy of learning rate.
* **num_epoch:** Number of epochs.
* **gradient_acc:** Gradient accumulation steps.
* **output_dir:** Directory to the output checkpoint.
* **seed:** Random seed number.

#### Hyperparameters:
|max_len|pad_to_max|batch_size|lr|weight_decay|num_epoch|gradient_acc|seed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|50|True|32|3e-5|1e-2|8|2|2022|

### Test
```shell
python3 bonus/src/test_intent.py \
--test_file <test_file> \
--ckpt_dir <ckpt_dir> \
--batch_size <batch_size> \
--output_file <output_file>
```
* **test_file:** Path to test file.
* **ckpt_dir:** Path to checkpoint directory.
* **batch_size:** Number of samples in one batch.
* **output_file:** Path to the result file.

### Reproduce(Public: 0.948, Private: 0.95155)
```shell
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging

### Train
```shell
accelerate launch bonus/src/train_slot.py \
--model_name_or_path <model_name> \
--train_file <train_file> \
--validation_file <valid_file> \
--max_length <max_len> \
--pad_to_max_length \
--per_device_train_batch_size <batch_size> \
--per_device_eval_batch_size <batch_size> \
--learning_rate <lr> \
--weight_decay <weight_decay> \
--num_train_epochs <num_epoch> \
--gradient_accumulation_steps <gradient_acc> \
--output_dir <output_dir> \
--seed <seed>
```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **max_len:** Length limit of sample text string.
* **pad_to_max:** Training with padding to max length.
* **batch_size:** Number of samples in one batch.
* **lr:** Learning rate.
* **weight_decay:** Weight deacy of learning rate.
* **num_epoch:** Number of epochs.
* **gradient_acc:** Gradient accumulation steps.
* **output_dir:** Directory to the output checkpoint.
* **seed:** Random seed number.

#### Hyperparameters:
|max_len|pad_to_max|batch_size|lr|weight_decay|num_epoch|gradient_acc|seed|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|128|True|32|3e-5|1e-2|8|2|2022|

### Test
```shell
python3 bonus/src/test_slot.py \
--model_name_or_path <model_path> \
--test_file <test_file> \
--batch_size <batch_size> \
--max_length <max_len> \
--device <device> \
--pred_file <pred_file>
```
* **model_path:** Path to checkpoint directory.
* **test_file:** Path to test file.
* **batch_size:** Number of samples in one batch.
* **max_len:** Length limit of sample text string.
* **device:** Device to run the testing.
* **pred_file:** Path to the output csv file.

### Reproduce(Public: 0.80536, Private: 0.81939)
```shell
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```