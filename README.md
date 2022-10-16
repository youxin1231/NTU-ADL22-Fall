## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Download
```shell
# To download the models for reproducing.
bash download.sh
```

## Intent detection

### Train
```shell
python train_intent.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --num_epoch <num_epoch>
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **max_len**: Length limit of sample text string.
* **hidden_size**: LSTM hidden state dimension.
* **num_layers**: LSTM number of layers.
* **dropout**: Model dropout rate.
* **bidirectional** biLSTM if True.
* **lr** Learning rate.
* **batch_size** Number of samples in one batch.
* **device** Device to run the training.
* **num_epoch** Number of epochs.
#### My Hyperparameters:
|max_len|hidden_size|num_layer|dropout|bidirectional|lr|batch_size|num_epoch|
|-|-|-|-|-|-|-|-|
|150|256|2|0.1|True|1e-3|64|150|


### Test
```shell
python test_intent.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --batch_size <batch_size> --device <device>
```
* **test_file**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to save the model file.
* **pred_file**: Path to the output csv file.
* **max_len**: Length limit of sample text string.
* **hidden_size**: LSTM hidden state dimension.
* **num_layers**: LSTM number of layers.
* **dropout**: Model dropout rate.
* **bidirectional** biLSTM if True.
* **batch_size** Number of samples in one batch.
* **device** Device to run the testing.

### Reproduce (Public: 0.92177, Private: 0.92008)
#### The best performing model on kaggle was washed out by me accidentally.And this is the second one, so the accuracy is different from that on kaggle(Public: 0.92577, Private: 0.91955).
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging

### train
```shell
python train_slot.py --data_dir <data_dir> --cache_dir <chche_dir> --ckpt_dir <ckpt_dir> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --lr <lr> --batch_size <batch_size> --device <device> --num_epoch <num_epoch>
```
* **data_dir**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_dir**: Directory to save the model file.
* **max_len**: Length limit of sample text string.
* **hidden_size**: LSTM hidden state dimension.
* **num_layers**: LSTM number of layers.
* **dropout**: Model dropout rate.
* **bidirectional** biLSTM if True.
* **lr** Learning rate.
* **batch_size** Number of samples in one batch.
* **device** Device to run the training.
* **num_epoch** Number of epochs.
#### My Hyperparameters:
|max_len|hidden_size|num_layer|dropout|bidirectional|lr|batch_size|num_epoch|
|-|-|-|-|-|-|-|-|
|256|1500|3|0.2|True|1.5*1e-3|64|100|

### Test
```shell
python test_slot.py --test_file <test_file> --cache_dir <chche_dir> --ckpt_path <ckpt_path> --pred_file <pred_file> --max_len <max_len> --hidden_size <hidden_size> --num_layers <num_layers> --dropout <dropout> --bidirectional <bidirectional> --batch_size <batch_size> --device <device>
```
* **test_file**: Directory to the dataset.
* **cache_dir**: Directory to the preprocessed caches.
* **ckpt_path**: Path to save the model file.
* **pred_file**: Path to the output csv file.
* **max_len**: Length limit of sample text string.
* **hidden_size**: LSTM hidden state dimension.
* **num_layers**: LSTM number of layers.
* **dropout**: Model dropout rate.
* **bidirectional** biLSTM if True.
* **batch_size** Number of samples in one batch.
* **device** Device to run the testing.

### Reproduce (Public: 0.78337, Private: 0.78403)
```shell
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```