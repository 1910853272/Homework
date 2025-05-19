import torch
from torch.utils.data import DataLoader
import numpy as np
from train import train
from generate_tang import generate_tang
from generate_cang import generate_cang
import matplotlib.pyplot as plt


def prepareData():
    datas = np.load("tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=16, shuffle=True, num_workers=0)
    return dataloader, ix2word, word2ix

if __name__ == '__main__':
    dataloader, ix2word, word2ix = prepareData()

    # 训练模型
    train(dataloader, ix2word, word2ix)

    # 生成唐诗
    start_words = '湖光秋月两相和'
    poem = generate_tang(start_words, ix2word, word2ix)
    print("生成唐诗：", ''.join(poem))

    # 生成藏头诗
    acrostic = generate_cang(start_words, ix2word, word2ix)
    print("生成藏头诗：", ''.join(acrostic))
