# utils/tokenizer.py
import random
import jieba
from nltk.tokenize import word_tokenize

class Tokenizer:
    def __init__(self, en_path, ch_path, count_min=5):
        self.en_path = en_path
        self.ch_path = ch_path
        self.__count_min = count_min

        self.en_data = self.__read_ori_data(en_path)
        self.ch_data = self.__read_ori_data(ch_path)

        self.index_2_word = ['unK', '<pad>', '<bos>', '<eos>']
        self.word_2_index = {'unK': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}

        self.en_set = set()
        self.en_count = {}

        self.__count_word()
        self.mx_length = 40
        self.data_ = []
        self.__filter_data()
        random.shuffle(self.data_)
        self.test = self.data_[-1000:]
        self.data_ = self.data_[:-1000]

    def __read_ori_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n')[:-1]
        return data

    def __count_word(self):
        for data in self.en_data:
            for word in word_tokenize(data):
                self.en_count[word] = self.en_count.get(word, 0) + 1
        for k, v in self.en_count.items():
            self.word_2_index[k] = len(self.index_2_word) if v >= self.__count_min else 0
            if v >= self.__count_min:
                self.index_2_word.append(k)

        self.en_set = set()
        self.en_count = {}
        for data in self.ch_data:
            for word in jieba.cut(data):
                self.en_count[word] = self.en_count.get(word, 0) + 1
        for k, v in self.en_count.items():
            self.word_2_index[k] = len(self.index_2_word) if v >= self.__count_min else 0
            if v >= self.__count_min:
                self.index_2_word.append(k)

    def __filter_data(self):
        for i in range(len(self.en_data)):
            self.data_.append([self.en_data[i], self.ch_data[i], 0])
            self.data_.append([self.ch_data[i], self.en_data[i], 1])

    def en_cut(self, data):
        tokens = word_tokenize(data)
        if len(tokens) > self.mx_length:
            return 0, []
        return 1, [self.word_2_index.get(w, 0) for w in tokens]

    def ch_cut(self, data):
        tokens = list(jieba.cut(data))
        if len(tokens) > self.mx_length:
            return 0, []
        return 1, [self.word_2_index.get(w, 0) for w in tokens]

    def encode_all(self, data):
        src, tgt, labels = [], [], []
        for d in data:
            if d[2] == 0:
                l1, s = self.en_cut(d[0])
                l2, t = self.ch_cut(d[1])
            else:
                l1, t = self.en_cut(d[1])
                l2, s = self.ch_cut(d[0])
            if l1 and l2:
                src.append(s)
                tgt.append(t)
                labels.append(d[2])
        return labels, src, tgt

    def decode(self, data):
        return self.index_2_word[data]

    def get_vocab_size(self):
        return len(self.index_2_word)

    def get_dataset_size(self):
        return len(self.en_data)
