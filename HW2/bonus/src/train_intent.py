from pathlib import Path
import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn
from transformers import BertTokenizer
from transformers import BertModel
from torch.optim import AdamW

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT model on intent classification")
    
    parser.add_argument(
        "--train_file", type=Path, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--eval_file", type=Path, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--device", type=str, default='cuda', help="Which device to run the training process."
    )
    parser.add_argument("--num_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--output_file", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=2022, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels):

        self.labels = [labels[label] for label in df['intent']]
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

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, labels_len)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def main():
    np.random.seed(args.seed)

    train_df = pd.read_json(args.train_file)
    eval_df = pd.read_json(args.eval_file)
    all_label = list(set(train_df['intent'].tolist() + eval_df['intent'].tolist()))
    
    labels = {}
    for i, label in enumerate(all_label):
        labels[str(label)] = i

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train(BertClassifier(len(labels)), tokenizer, train_df, eval_df, labels, args.learning_rate, args.num_epochs)

def train(model, tokenizer, train_data, val_data, labels, learning_rate, epochs):

    train, val = Dataset(train_data, tokenizer, labels), Dataset(val_data, tokenizer, labels)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr= args.learning_rate)

    model.to(args.device)

    epoch_pbar = trange(args.num_epochs, desc="Epoch")
    for epoch_num in epoch_pbar:

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in train_dataloader:

                train_label = train_label.to(args.device)
                mask = train_input['attention_mask'].to(args.device)
                input_id = train_input['input_ids'].squeeze(1).to(args.device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(args.device)
                    input_id = val_input['input_ids'].squeeze(1).to(args.device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

if __name__ == '__main__':
    args = parse_args()
    main()