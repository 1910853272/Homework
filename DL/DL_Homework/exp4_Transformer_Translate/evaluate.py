# evaluate.py
import torch
import jieba
import re
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.predict import predict

smooth = SmoothingFunction().method1

def is_english_sentence(sentence):
    return bool(re.search(r'[a-zA-Z]', sentence))

def compute_bleu4(tokenizer, random_indices, model, device):
    m1, m2, m3, m4 = [], [], [], []
    model.eval()
    da = [tokenizer.test[i] for i in random_indices]
    labels, x, _ = tokenizer.encode_all(da)

    with torch.no_grad():
        y = predict(x, model, tokenizer, device)

    if len(y) != len(labels):
        return 0

    for idx, label in zip(random_indices, labels):
        ref = tokenizer.test[idx][1]
        hyp = y.pop(0)
        if is_english_sentence(ref):
            m1.append(ref)
            m2.append([hyp])
        else:
            m3.append(list(jieba.cut(ref)))
            m4.append([list(jieba.cut(hyp))])

    bleu_scores = [sacrebleu.sentence_bleu(candidate, refs).score for candidate, refs in zip(m1, m2)]
    bleu_scores += [sentence_bleu(hyp, ref, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
                    for ref, hyp in zip(m3, m4)]
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0


def eval_model(model_path, tokenizer, batch_size=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    scores = []
    for i in range(0, len(tokenizer.test), batch_size):
        indices = list(range(i, min(i + batch_size, len(tokenizer.test))))
        score = compute_bleu4(tokenizer, indices, model, device)
        if score > 0:
            scores.append(score)

    final_score = sum(scores) / len(scores) if scores else 0
    print(f"Average BLEU-4 score: {final_score:.2f}")
    return final_score

if __name__ == '__main__':
    from models.transformer import Transformer
    from utils.tokenizer import Tokenizer

    tokenizer = Tokenizer('sample\TM-training-set\english.txt', 'sample\TM-training-set\chinese.txt', count_min=3)
    eval_model('model/translation_epoch25.pt', tokenizer)