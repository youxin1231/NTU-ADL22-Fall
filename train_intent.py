import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import shuffle
from typing import Dict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_one_epoch(dataloader, model, optimizer):
    model.train()

    y_pred, y_true = [], []
    total_loss = 0

    for batch in dataloader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)

        out_dict = model(batch)
        pred = out_dict['pred']
        loss = out_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        pred_idx = torch.max(pred, 1)[1]
        y_pred += pred_idx.tolist()
        y_true += batch['intent'].tolist()
        total_loss += loss
    
    train_loss = total_loss.float() / len(dataloader)
    train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc

def eval_acc(dataloader, model):
    model.eval()

    y_pred, y_true = [], []
    total_loss = 0

    for batch in dataloader:
        batch['text'] = batch['text'].to(args.device)
        batch['intent'] = batch['intent'].to(args.device)

        out_dict = model(batch)
        pred = out_dict['pred']
        loss = out_dict['loss']
    
        pred_idx = torch.max(pred, 1)[1]
        y_pred += pred_idx.tolist()
        y_true += batch['intent'].tolist()
        total_loss += loss

    val_loss = total_loss.float() / len(dataloader)
    val_acc = accuracy_score(y_true, y_pred)
    return val_loss, val_acc

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    dataloaders = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loss, train_acc = train_one_epoch(dataloaders[TRAIN], model, optimizer)

        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_loss, val_acc = eval_acc(dataloaders[DEV], model)
        print(f"Train loss: {train_loss:.2f} - acc: {train_acc:.2f}. Validation loss: {val_loss:.2f} - acc: {val_acc:.2f}.")
        
        if val_acc > best_acc:
            best_acc = val_acc

            ckpt_path = args.ckpt_dir / 'best.pt'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }, ckpt_path)

    # TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=256)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=150)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)