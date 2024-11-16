import numpy as np
import torch
from torch.utils import data
from transformers import AutoTokenizer
from itertools import chain

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def save_vocab(fpath, target_list):
    tag_set = list(set(tuple(chain.from_iterable(target_list))))
    # NER 태그 기본 구조 정의
    VOCAB = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', 'O']
    # BIO 태그 추가
    bio_tags = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    VOCAB.extend(bio_tags)
    VOCAB = tuple(VOCAB)
    with open(fpath,'w') as save:
        save.write(" ".join(VOCAB))

def load_vocab(fpath):
    with open(fpath,'r') as f:
        f = f.readline()
        VOCAB = f.strip().split()
    return VOCAB



class DataLoader(data.Dataset):
    def __init__(self, fpath, args):
        entries = open(fpath, 'r').read().strip().split('\n\n')
        source_list, target_list = [], []
        for lines in entries:
            words = []
            ner_tags = []
            for line in lines.splitlines():
                if line.strip():
                    try:
                        # 라인 형식: index|word POS chunk NER
                        parts = line.strip().split('|')[1].split()
                        if len(parts) >= 4:  # NER 태그가 있는 경우
                            word = parts[0]
                            ner_tag = parts[3]  # NER 태그는 마지막 컬럼
                        else:
                            word = parts[0]
                            ner_tag = 'O'
                    except (IndexError, ValueError):
                        continue
                    
                    words.append(word)
                    ner_tags.append(ner_tag)
            
            if words:  # 빈 문장 제외
                source_list.append(words)
                target_list.append(ner_tags)

        self.source_list = source_list
        self.target_list = target_list
        
        if fpath == args.trainset:
            save_vocab(args.vocab_path, target_list)
        
        self.vocab = load_vocab(args.vocab_path)
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.vocab)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.vocab)}

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, idx):
        source, target = self.source_list[idx], self.target_list[idx]

        source = ['[CLS]'] + source + ['[SEP]']
        target = ['[CLS]'] + target + ['[SEP]']
        x, y = [], []
        is_heads = []
        
        for w, t in zip(source, target):
            tokens = tokenizer.tokenize(w) if w not in ('[CLS]', '[SEP]') else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            is_head = [1] + [0]*(len(tokens)-1)

            t = [t] + ["[PAD]"]*(len(tokens)-1)
            yy = [self.tag2idx[each] for each in t]

            x.extend(xx)
            y.extend(yy)
            is_heads.extend(is_head)

        x_seqlen, y_seqlen = len(x), len(y)

        return x, x_seqlen, source, y, y_seqlen, target, is_heads

def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    x_seqlens = f(1)
    sources = f(2)
    y_seqlens = f(4)
    target = f(5)
    is_heads = f(6)

    x_maxlen = np.array(y_seqlens).max()
    y_maxlen = np.array(y_seqlens).max()

    f = lambda x, maxlen: [sample[x]+[0]*(maxlen-len(sample[x])) for sample in batch]
    x = f(0, x_maxlen)
    y = f(3, y_maxlen)

    f = torch.LongTensor

    return f(x), x_seqlens, sources, f(y), y_seqlens, target, is_heads
