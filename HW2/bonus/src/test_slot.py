import argparse
from pathlib import Path

from datasets import load_dataset

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

def main():
    args = parse_args()

    id2tag_path = Path(args.model_name_or_path + f'/id2tag.json')
    id2tag = json.loads(id2tag_path.read_text())
    num_labels = len(id2tag)

    # Load dataset:
    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    text_column_name = "tokens"

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
        # Reorganize `label_list` to match the ordering of the model.
        label_list = [model.config.id2label[i] for i in range(num_labels)]
        label_to_id = {l: i for i, l in enumerate(label_list)}
    
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
    dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    # Predict
    print("***** Running Token Classification Prediction *****")
    print(f"Number of data = {len(raw_datasets['test'])}")
    print(f"Batch size = {args.batch_size}")
    model.to(args.device)

    model.eval()
    pred = []
    for step, batch in enumerate(tqdm(dataloader)):
        batch["input_ids"] = batch["input_ids"].to(args.device)
        batch["attention_mask"] = batch["attention_mask"].to(args.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        pred += predictions.tolist()

    with args.pred_file.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'tags'])

        for i in range(len(pred)):
            tmp = ''
            for j in range(len(raw_datasets['test'][i]['tokens'])):
                tmp += str(id2tag[str(pred[i][j])]) + ' '
            writer.writerow([raw_datasets['test'][i]['id'], tmp])

if __name__ == "__main__":
    main()