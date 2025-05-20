# utils/predict.py
import torch
import numpy as np
from torch.autograd import Variable
from nltk.corpus import words
from nltk.tokenize.treebank import TreebankWordDetokenizer

from utils.dataset import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, ys, src_mask,
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, i])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


def predict(data, model, tokenizer, device='cuda'):
    with torch.no_grad():
        results = []
        for i in range(len(data)):
            src = torch.from_numpy(np.array(data[i])).long().to(device).unsqueeze(0)
            src_mask = (src != tokenizer.word_2_index['<pad>']).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len=100, start_symbol=tokenizer.word_2_index['<bos>'])

            translation = []
            for j in range(1, out.size(1)):
                sym = tokenizer.index_2_word[out[0, j].item()]
                if sym == '<eos>':
                    break
                translation.append(sym)

            if translation and translation[0].lower() in words.words():
                results.append(TreebankWordDetokenizer().detokenize(translation))
            else:
                results.append("".join(translation))
        return results
