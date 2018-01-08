from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist import SerializableModule

class GRU2D(SerializableModule):
    def __init__(self, input_size, hidden_size, num_layers=0, dropout=0, omnidirectional=True):
        super().__init__()
        self.n_stacks = n_stacks = 4 if omnidirectional else 1
        self.md_gru = [MDGRUCell(input_size, hidden_size, m_dim=2) for _ in range(n_stacks)]
        if num_layers > 0:
            self.gru = nn.GRU(hidden_size * n_stacks, hidden_size, num_layers, dropout=dropout)
        else:
            self.gru = None

        for i, cell in enumerate(self.md_gru):
            self.add_module("md_gru_cell{}".format(i), cell)
        self.omnidirectional = omnidirectional
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def encode_omni(self, x, hidden, i):
        cell = self.md_gru[i]
        return cell(x, hidden)

    def forward(self, x):
        # x: (batch, input_size, h, w)
        batch_size, input_size, h, w = x.size()
        x = x.permute(2, 3, 0, 1)
        rnn_out1 = []
        last_out = [None, None]
        for i in range(h):
            rnn_out = []
            for j in range(w):
                last_out[0] = rnn_out1[-1][j] if len(rnn_out1) > 0 else None
                last_out[1] = rnn_out[-1] if len(rnn_out) > 0 else None
                rnn_out.append(self.encode_omni(x[i][j], last_out, 0))
            rnn_out1.append(torch.stack(rnn_out))
        rnn_out1 = torch.stack(rnn_out1)

        rnn_out2 = []
        last_out = [None, None]
        for i in range(h):
            rnn_out = []
            for j in reversed(range(w)):
                last_out[0] = rnn_out2[-1][j] if len(rnn_out2) > 0 else None
                last_out[1] = rnn_out[-1] if len(rnn_out) > 0 else None
                rnn_out.append(self.encode_omni(x[i][j], last_out, 1))
            rnn_out2.append(torch.stack(list(reversed(rnn_out))))
        rnn_out2 = torch.stack(rnn_out2)

        rnn_out3 = []
        last_out = [None, None]
        for i in reversed(range(h)):
            rnn_out = []
            for j in reversed(range(w)):
                last_out[0] = rnn_out3[-1][j] if len(rnn_out3) > 0 else None
                last_out[1] = rnn_out[-1] if len(rnn_out) > 0 else None
                rnn_out.append(self.encode_omni(x[i][j], last_out, 2))
            rnn_out3.append(torch.stack(list(reversed(rnn_out))))
        rnn_out3 = torch.stack(list(reversed(rnn_out3)))

        rnn_out4 = []
        last_out = [None, None]
        for i in reversed(range(h)):
            rnn_out = []
            for j in range(w):
                last_out[0] = rnn_out4[-1][j] if len(rnn_out4) > 0 else None
                last_out[1] = rnn_out[-1] if len(rnn_out) > 0 else None
                rnn_out.append(self.encode_omni(x[i][j], last_out, 3))
            rnn_out4.append(torch.stack(rnn_out))
        rnn_out4 = torch.stack(list(reversed(rnn_out4)))
        rnn_out = torch.cat([rnn_out1, rnn_out2, rnn_out3, rnn_out4], 3)
        rnn_out = rnn_out.permute(2, 3, 0, 1).contiguous()
        rnn_out = rnn_out.view(rnn_out.size(0), rnn_out.size(1), -1)
        rnn_out = rnn_out.permute(2, 0, 1)
        if self.gru:
            rnn_out = self.gru(self.dropout(rnn_out))[0]
        rnn_out = rnn_out.permute(2, 1, 0).contiguous()
        rnn_out = rnn_out.view(batch_size, self.hidden_size, h, w).contiguous()
        return rnn_out.permute(0, 2, 3, 1)

class MDGRUCell(SerializableModule):
    def __init__(self, input_size, hidden_size, bias=True, m_dim=1, mode="mean"):
        super().__init__()
        self.cells = [nn.GRUCell(input_size, hidden_size, bias=bias) for _ in range(m_dim)]
        for i, c in enumerate(self.cells):
            self.add_module("cell{}".format(i), c)
        self.hidden_size = hidden_size
        self.m_dim = m_dim
        self.mode = mode

    def forward(self, x, hidden=None):
        batch_size, input_size = x.size()
        if hidden == None:
            hidden = [None] * self.m_dim
        hidden_arr = []
        for h in hidden:
            if h is None:
                hidden_arr.append(Variable(torch.zeros(batch_size, self.hidden_size)).cuda())
            else:
                hidden_arr.append(h)
        output_arr = [cell(x, h) for cell, h in zip(self.cells, hidden_arr)]
        if self.mode == "mean":
            out = torch.mean(torch.stack(output_arr), 0)
        elif self.mode == "max":
            pass
        return out