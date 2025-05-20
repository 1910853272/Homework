# run.py
import argparse
from train import train
from evaluate import eval_model
from utils.tokenizer import Tokenizer

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer Machine Translation")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                        help='Run mode: train or eval')
    parser.add_argument('--model_path', type=str, default='model/translation_epoch25.pt',
                        help='Path to saved model (for evaluation)')
    parser.add_argument('--src', type=str, default='sample\TM-training-set\english.txt', help='Path to English source file')
    parser.add_argument('--tgt', type=str, default='sample\TM-training-set\chinese.txt', help='Path to Chinese target file')
    args = parser.parse_args()

    tokenizer = Tokenizer(args.src, args.tgt, count_min=3)

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        eval_model(args.model_path, tokenizer)
