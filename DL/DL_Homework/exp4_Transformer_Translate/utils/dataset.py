# utils/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return len(self.src)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, trg=None, tokenizer=None, device='cuda'):
        src = src.to(device).long()
        trg = trg.to(device).long()
        self.src = src
        self.__pad = tokenizer.word_2_index['<pad>']
        self.src_mask = (src != self.__pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, self.__pad)
            self.ntokens = (self.trg_y != self.__pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
