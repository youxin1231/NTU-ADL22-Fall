from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json

import torch
from torch import nn
from transformers import BertTokenizer
import csv
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT model on intent classification")
    
    parser.add_argument(
        "--test_file", type=Path, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--ckpt_dir", type=Path, default=None, help="Directory of check point files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--device", type=str, default='cuda', help="Which device to run the training process."
    )
    parser.add_argument("--output_file", type=Path, default=None, help="Path of output file.")

    args = parser.parse_args()
    return args
    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels):

        self.labels = [-1 for text in df['text']]
        self.texts = [tokenizer(text, 
                                padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, labels_len, dropout = 0.2):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(args.ckpt_dir)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, labels_len)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def evaluate(model, test_data, tokenizer, labels):

    test = Dataset(test_data, tokenizer, labels)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

    model.to(args.device)
    pred = []
    with torch.no_grad():

        for test_input in tqdm(test_dataloader):
            test_input = test_input[0]
            mask = test_input['attention_mask'].to(args.device)
            input_id = test_input['input_ids'].squeeze(1).to(args.device)

            output = model(input_id, mask)

            for i in output.argmax(dim=1).tolist():
                pred.append(list(labels.keys())[list(labels.values()).index(i)])
    return pred
def main():

    test_df = pd.read_json(args.test_file)
    json_path = args.ckpt_dir / f'intent2idx.json'
    labels = json.loads(json_path.read_text())

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model = torch.load(args.ckpt_dir / f'model')
    pred = evaluate(model, test_df, tokenizer, labels)

    with open(args.output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'intent'])
        for i in range(len(pred)):
            print(test_df[i]['id'], pred[i])
            writer.writerow([test_df[i]['id'], pred[i]])
  

if __name__ == '__main__':
    args = parse_args()
    main()