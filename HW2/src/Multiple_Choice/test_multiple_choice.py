import argparse
from pathlib import Path
from itertools import chain

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Predict the test data on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="ckpt/bert-base-chinese/multiple_choice",
        help="Path to pretrained model or model identifier from huggingface.co/models."
        )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/preprocessed/test_swag.json",
        help="A csv or a json file containing the training data."
        )
    parser.add_argument(
        "--pred_file", 
        type=Path, 
        default="data/pred/multiple_choice_pred.json",
        help="A csv or a json file containing the training data."
        )

    # Data
    parser.add_argument("--max_length", type=int, default=512)

    # Data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    

    def preprocess_function(examples):
        first_sentences = [[context] * 4 for context in examples[context_name]]
        question_headers = examples[question_header_name]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        return tokenized_inputs

    # DataLoaders creation:
    extension = args.test_file.split(".")[-1]
    data_files = {}
    data_files["test"] = args.test_file
    raw_datasets = load_dataset(extension, data_files=data_files)

    ending_names = [f"ending{i}" for i in range(4)]
    context_name = "sent1"
    question_header_name = "sent2"
    padding = "max_length"
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets['test'].column_names
        )
    
    dataloader = DataLoader(processed_datasets['test'], collate_fn=default_data_collator, batch_size=args.batch_size)

    # Predict
    print("***** Running Predicting *****")
    model.eval()
    pred = []
    for step, batch in enumerate(tqdm(dataloader)):
        batch["input_ids"] = batch["input_ids"].to(args.device)
        batch["attention_mask"] = batch["attention_mask"].to(args.device)
        batch["token_type_ids"] = batch["token_type_ids"].to(args.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        pred += int(predictions)

    raw_data = raw_datasets['test']
    output_data = []
    for i in range(len(raw_data)):
        tmp = {}
        tmp['context'] = raw_data[i]['ending'+str(pred[i])]
        tmp['id'] = raw_data[i]['video-id']
        tmp['question'] = raw_data[i]['sent1']
        tmp['title'] = 'test'
        output_data.append(tmp)
    
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    args.pred_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')
if __name__ == "__main__":
    main()