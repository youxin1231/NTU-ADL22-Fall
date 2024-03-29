import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_one_epoch(dataloader, model, optimizer):
    model.train()

    y_pred, y_true, y_pred_flat, y_true_flat = [], [], [], []
    total_loss = 0

    for batch in dataloader:
        batch_size = len(batch['tokens'])
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        length = batch['length']

        out_dict = model(batch)
        loss = out_dict['loss']
        pred_idx = out_dict['pred_idx']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for i in range(batch_size):
            y_pred.append(pred_idx[i,:length[i]].tolist())
            y_true.append(batch['tags'][i,:length[i]].tolist())
            y_pred_flat += pred_idx[i,:length[i]].tolist()
            y_true_flat += batch['tags'][i,:length[i]].tolist()
        total_loss += loss
    train_loss = total_loss / len(dataloader)
    train_token_acc = accuracy_score(y_true_flat, y_pred_flat)

    joint_correct = 0
    data_size = len(y_pred)
    for i in range(data_size):
        if y_pred[i] == y_true[i]:
            joint_correct += 1
    train_joint_acc = joint_correct / data_size

    return train_loss, train_token_acc, train_joint_acc

def eval_acc(dataloader, dataset, model):
    model.eval()

    y_pred, y_true, y_pred_flat, y_true_flat = [], [], [], []
    total_loss = 0

    for batch in dataloader:
        batch_size = len(batch['tokens'])
        batch['tokens'] = batch['tokens'].to(args.device)
        batch['tags'] = batch['tags'].to(args.device)
        length = batch['length']

        out_dict = model(batch)
        loss = out_dict['loss']
        pred_idx = out_dict['pred_idx']
        
        for i in range(batch_size):
            y_pred.append(pred_idx[i,:length[i]].tolist())
            y_true.append(batch['tags'][i,:length[i]].tolist())
            y_pred_flat += pred_idx[i,:length[i]].tolist()
            y_true_flat += batch['tags'][i,:length[i]].tolist()
        total_loss += loss
    val_loss = total_loss / len(dataloader)
    val_token_acc = accuracy_score(y_true_flat, y_pred_flat)

    joint_correct = 0
    data_size = len(y_pred)
    for i in range(data_size):
        if y_pred[i] == y_true[i]:
            joint_correct += 1
    val_joint_acc = joint_correct / data_size

    y_true_label, y_pred_label = [], []
    for i in range(len(y_true)):
        t, p = [], []
        for j in range(len(y_true[i])):
            t.append(dataset.idx2label(y_true[i][j]))
            p.append(dataset.idx2label(y_pred[i][j]))
        y_true_label.append(t)
        y_pred_label.append(p)
    print(classification_report(y_true_label, y_pred_label, scheme=IOB2, mode='strict'))

    return val_loss, val_token_acc, val_joint_acc

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    dataloaders = {
        split: DataLoader(split_dataset, args.batch_size, shuffle=True, collate_fn=split_dataset.collate_fn) for split, split_dataset in datasets.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets[TRAIN].num_classes, args.max_len).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loss, train_token_acc, train_joint_acc = train_one_epoch(dataloaders[TRAIN], model, optimizer)

        # TODO: Evaluation loop - calculate accuracy and save model weights
        val_loss, val_token_acc, val_joint_acc = eval_acc(dataloaders[DEV], datasets[DEV], model)
        print(f"Train loss: {train_loss:.2f} - joint_acc: {train_joint_acc:.2f} - token_acc: {train_token_acc:.2f}. Validation loss: {val_loss:.2f} - joint_acc: {val_joint_acc:.2f} - token_acc: {val_token_acc:.2f}.")
        
        if val_joint_acc > best_acc:
            best_acc = val_joint_acc

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
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=256)

    # model
    parser.add_argument("--hidden_size", type=int, default=1500)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1.5*1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
