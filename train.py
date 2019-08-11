import torch
from model import CharLSTM
from dataset import QuoteDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

dataset = QuoteDataset("author-quote.txt")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# hyper-parameters
n_chars = dataset.n_chars
HIDDEN_SIZE = 500
INPUT_SIZE = n_chars
OUTPUT_SIZE = n_chars
N_LAYERS = 1
PRINT_EVERY = 1000
avg_loss = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CharLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    for sample_no, sample in enumerate(dataloader):
        model.zero_grad()
        input = sample["input"]
        target = sample["target"]

        h, c = model.init_hidden()
        h, c = h.to(device), c.to(device)
        hidden = (h, c)
        input, target = input.to(device), target.to(device)
        output, hidden = model(input, hidden)
        loss = criterion(output.squeeze(), target.squeeze())
        avg_loss += loss.item() / input.size()[1]
        optimizer.step()

        if (sample_no + 1) % PRINT_EVERY == 0:
            out = "Epoch: {}, sample_no: {}, loss: {}"
            print(out.format(epoch, sample_no + 1, avg_loss / PRINT_EVERY))
            avg_loss = 0

# Save the model
torch.save(model, "model.pt")
