import argparse
from pathlib import Path

from datasets import ClassLabel, load_dataset
from random import randint
import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    default_data_collator,
)
from tqdm import tqdm
import json
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the test data on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ckpt/bert-base-chinese/slot",
        help="Path to pretrained model or model identifier from huggingface.co/models."
        )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/preprocessed/test_swag.json",
        help="A csv or a json file containing the test data."
        )
    parser.add_argument(
        "--pred_file", 
        type=Path, 
        default="slot_result.csv",
        help="A csv or a json file containing the output file."
        )

    # Data
    parser.add_argument("--max_length", type=int, default=512)

    # Data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()

    return args
def preprocess(args):
    test_path = Path(args.test_file)
    test_data = json.loads(test_path.read_text())
    id2tag_path = Path(args.model_name_or_path + f'/id2tag.json')
    id2tag = json.loads(id2tag_path.read_text())
    for i in test_data:
        tmp = []
        for j in range(len(i['tokens'])):
            tmp.append(id2tag[str(randint(0, 8))])
        i['tags'] = tmp
    test_path.write_text(json.dumps(test_data, indent=2))

def main():
    args = parse_args()
    preprocess(args)

    # Load dataset:
    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    features = raw_datasets["test"].features
    column_names = raw_datasets["test"].column_names
    label_column_name = column_names[2]
    text_column_name = "tokens"

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["test"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    tokenizer_name_or_path =  args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
            # Reorganize `label_list` to match the ordering of the model.
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
    
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocess the dataset
    padding = "max_length"

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_raw_datasets["test"]

    # DataLoaders creation:
    data_collator = default_data_collator
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    
    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if args.device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    # Predict
    print("***** Running Token Classification Prediction *****")
    print(f"Number of data = {len(raw_datasets['test'])}")
    print(f"Batch size = {args.batch_size}")
    model.to(args.device)

    pred = []
    model.eval()
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch["input_ids"] = batch["input_ids"].to(args.device)
        batch["attention_mask"] = batch["attention_mask"].to(args.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(args.device)
        batch["labels"] = batch["labels"].to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        preds, refs = get_labels(predictions, labels)
        pred += preds

    output_path = Path(args.pred_file)
    with output_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tags'])
        for idx, i in enumerate(pred):
            tmp = ''
            for idx2, j in enumerate(i):
                if idx2 == len(i)-1:
                    tmp += str(j)
                else:
                    tmp += str(j) + ' '
            writer.writerow([str(raw_datasets['test'][idx]['id']), tmp])

if __name__ == "__main__":
    main()