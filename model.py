from typing import Dict
from unicodedata import bidirectional

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:

        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.lstm = nn.LSTM(input_size=embeddings.size(1),
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional,
                        dropout=self.dropout,
                        batch_first=True)
        self.fc = nn.Linear(self.encoder_output_size, self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        self.lstm.flatten_parameters()
        x, y = batch['text'], batch['intent']
        embed_x = self.embed(x)
        packed_x = pack_padded_sequence(embed_x, batch['length'], batch_first=True, enforce_sorted=False)
        
        packed_out, (hidden, cell) = self.lstm(packed_x)

        unpacked_out, unpacked_len = pad_packed_sequence(packed_out, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), axis=-1) 
        else:
            hidden = hidden[-1]

        pred = self.fc(hidden)
        loss = nn.CrossEntropyLoss()
        loss_val = loss(pred, y)
        return {
            'pred': pred,
            'loss': loss_val
        }

class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        max_len
    ) -> None:

        super().__init__(embeddings, hidden_size, num_layers, dropout, bidirectional, num_class)
        self.max_len = max_len
        self.fc = nn.Linear(self.encoder_output_size, self.max_len * self.num_class)
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        self.lstm.flatten_parameters()
        x, y = batch['tokens'], batch['tags']
        embed_x = self.embed(x)
        packed_x = pack_padded_sequence(embed_x, batch['length'], batch_first=True, enforce_sorted=False)
        
        packed_out, (hidden, cell) = self.lstm(packed_x)

        unpacked_out, unpacked_len = pad_packed_sequence(packed_out, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat((hidden[-1], hidden[-2]), axis=-1) 
        else:
            hidden = hidden[-1]

        pred = self.fc(hidden)
        loss = nn.CrossEntropyLoss()
        loss_val = loss(pred, y)
        return {
            'pred': pred,
            'loss': loss_val
        }
