import torch
import random
import torch.nn as nn
from dataset import QuoteDataset
import numpy as np
from model import CharLSTM
import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("model.pt", map_location=device)
model.eval()
dataset = QuoteDataset("author-quote.txt")

with torch.no_grad():
    (h, c) = model.init_hidden()
    h, c = h.to(device), c.to(device)
    hidden = (h, c)
    soft = nn.Softmax(dim=0)

    x = torch.zeros(dataset.n_chars)
    x = x.to(device)
    x[0] = 1

    quote = ""

    for i in range(150):  # Average lenght of quotes was ~ 125
        output, hidden = model(x.view(1, 1, -1), hidden)
        output = soft(output.squeeze())
        output = output.cpu()
        output = output.numpy()

        char_ind = np.random.choice(dataset.n_chars, p=output.ravel())

        x = torch.zeros(dataset.n_chars)
        x = x.to(device)
        x[char_ind] = 1

        if char_ind == dataset.n_chars - 1:
            break
        else:
            quote += str(dataset.idx_to_char[char_ind])

    print('"' + quote + '"')
