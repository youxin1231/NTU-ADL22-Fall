from random import sample
from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len

import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text = [s['text'].split() for s in samples]
        length = torch.tensor([min(len(s), self.max_len) for s in text]) 
        text = self.vocab.encode_batch(text)
        text = torch.tensor(text)
        id = [s['id'] for s in samples]

        if 'intent' in samples[0].keys():
            intent = [self.label2idx(s['intent']) for s in samples]
            intent = torch.tensor(intent)
        else:
            intent = torch.zeros(len(samples), dtype=torch.long)
        return {
        'text': text,
        'intent': intent,
        'id': id,
        'length': length
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        tokens = [s['tokens'] for s in samples]
        length = torch.tensor([min(len(s), self.max_len) for s in tokens])
        tokens = self.vocab.encode_batch(tokens, self.max_len)
        id = [s['id'] for s in samples]
        tokens = torch.tensor(tokens)

        if 'tags' in samples[0].keys():
            tags = [[self.label2idx(tag) for tag in s['tags']] for s in samples]
            tags = pad_to_len(tags, self.max_len, 0)
            tags = torch.tensor(tags)
        else:
            tags = torch.tensor([[0] * self.max_len for l in length])
        
        return {
        'tokens': tokens,
        'tags': tags,
        'id': id,
        'length': length
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]