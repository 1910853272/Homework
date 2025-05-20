# train.py
import time
import torch
import random
import os
from models.transformer import Transformer
from utils.tokenizer import Tokenizer
from utils.dataset import Batch
from utils.loss_utils import LabelSmoothing, SimpleLossCompute, NoamOpt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train():
    en_path = 'sample\TM-training-set\english.txt'
    ch_path = 'sample\TM-training-set\chinese.txt'
    tokenizer = Tokenizer(en_path, ch_path, count_min=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(tokenizer, device=device)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    model = model.to(device)

    criterion = LabelSmoothing(tokenizer.get_vocab_size(), tokenizer.word_2_index['<pad>'], smoothing=0.1)
    optimizer = NoamOpt(256, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    lossF = SimpleLossCompute(model.generator, criterion, optimizer)

    data_loader = tokenizer.get_dataloader(tokenizer.data_)
    batchs = [Batch(src, tgt, tokenizer=tokenizer, device=device) for src, tgt in data_loader]

    print("Start training with vocab size:", tokenizer.get_vocab_size())
    loss_all = []
    epochs = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        t_start = time.time()
        for i, batch in enumerate(batchs):
            out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = lossF(out, batch.trg_y, batch.ntokens)
            total_loss += loss

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss per token: {loss / batch.ntokens:.4f}, Time elapsed: {time.time() - t_start:.2f}s")
                t_start = time.time()

        avg_loss = total_loss / len(batchs)
        loss_all.append(avg_loss)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(), f'model/translation_epoch{epoch}.pt')

    with open('loss.txt', 'w', encoding='utf-8') as f:
        f.write(str(loss_all))

if __name__ == '__main__':
    train()
