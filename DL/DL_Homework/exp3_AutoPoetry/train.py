import torch
import torch.nn as nn
from model import PoetryModel
import matplotlib.pyplot as plt

# 超参数设置
embedding_dim = 128
hidden_dim = 256
learning_rate = 5e-3
epochs = 4
model_path = 'model.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
verbose = True

def train(dataloader, ix2word, word2ix):
    model = PoetryModel(len(word2ix), embedding_dim, hidden_dim)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    loss_list = []
    batch_count = 0

    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous().to(device)
            input, target = data[:-1, :], data[1:, :]

            output, _ = model(input)
            output = output.view(-1, output.size(-1))  # flatten logits
            target = target.contiguous().view(-1)      # flatten labels

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            batch_count += 1

            if batch_idx % 900 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data[1]), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))

    torch.save(model.state_dict(), 'model.pth')

    # 绘制 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()

