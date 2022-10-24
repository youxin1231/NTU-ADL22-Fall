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
        self.fc = nn.Linear(self.encoder_output_size, self.num_class)
        self.transition = nn.Parameter(torch.randn(self.num_class, self.num_class))

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional:
            return 2 * self.hidden_size
        return self.hidden_size

    def get_idx(self, tokens_len):
        batch_idx = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(tokens_len)])
        tok_idx = torch.cat([torch.arange(0, t_len) for t_len in tokens_len])
        return batch_idx, tok_idx

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        self.lstm.flatten_parameters()
        x, y = batch['tokens'], batch['tags']
        mask = (x != 0)

        embed_x = self.embed(x)
        packed_x = pack_padded_sequence(embed_x, batch['length'], batch_first=True, enforce_sorted=False)
        hidden, _ = self.lstm(packed_x)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        pred_score = self.fc(hidden)

        pred_idx = torch.max(pred_score, dim=2)[1]

        idx = self.get_idx(batch['length'])
        loss_F = nn.CrossEntropyLoss()
        loss = loss_F(pred_score[idx], y[idx])
        # loss, pred_idx = self.CRF(pred_score, y, mask)
        return {
            'loss': loss,
            'pred_idx': pred_idx,
        }

    def CRF(self, pred_score, y, mask) -> Dict[str, torch.Tensor]:
        batch_size, len = y.shape
        score = torch.gather(pred_score, dim=2, index=y.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[y[:, :-1], y[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)
        pred_idx = [[[i] for i in range(self.num_class)]] * batch_size

        d1 = torch.unsqueeze(pred_score[:, 0], dim=1)
        d2 = d1.clone().detach()
        for i in range(1, len):
            n_unfinished = mask[:, i].sum()
            d_uf = d1[: n_unfinished]
            emit_and_transition = pred_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            log_sum = new_d_uf
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)
            log_sum = log_sum - max_v 
            d_uf1 = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d1 = torch.cat((d_uf1, d1[n_unfinished:]), dim=0)

            d_uf2, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()
            pred_idx[: n_unfinished] = [[pred_idx[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d2 = torch.cat((torch.unsqueeze(d_uf2, dim=1), d2[n_unfinished:]), dim=0)

        # loss
        d1 = d1.squeeze(dim=1)
        max_d = d1.max(dim=-1)[0]
        d_n = max_d + torch.logsumexp(d1 - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d_n
        loss = -llk
        loss = sum(loss)

        # pred_idx
        d2 = d2.squeeze(dim=1)
        _, max_idx = torch.max(d2, dim=1)
        max_idx = max_idx.tolist()
        pred_idx = [pred_idx[b][k] for b, k in enumerate(max_idx)]

        return loss, pred_idx

