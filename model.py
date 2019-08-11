import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2l = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.h2l(output)
        return output, hidden

    def init_hidden(self):
        h_0 = torch.zeros(self.n_layers, 1, self.hidden_size)
        c_0 = torch.zeros(self.n_layers, 1, self.hidden_size)
        return (h_0, c_0)
