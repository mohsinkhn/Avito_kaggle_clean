import numpy as np
from collections import Counter
import itertools

def grams2(src_words):
    return list(zip(src_words, src_words[1:]))


def ngrams(n, f, prune_after=10000):
    for i, line in enumerate(f):
        words = str(line).split()
        yield grams2(words)
        
        
class NgramTokenizer:
    def __init__(self, max_features=20000, ngram=2):
        self.max_features = max_features
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None
        self.ngram = ngram

    def fit_transform(self, texts):
        tokenized = []
        # doc_freq = Counter()
        n = len(texts)
        tokenized = ngrams(self.ngram, texts, prune_after=3000000)
        doc_freq = Counter(itertools.chain.from_iterable(tokenized))

        # vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab = [t[0] for t in doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        # doc_freq = [doc_freq[t] for t in vocab]

        # self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx
        print("Vocab building done")
        max_len = 0
        result_list = []
        tokenized = ngrams(self.ngram, texts, prune_after=3000000)
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = grams2(str(texts[i]).split())
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1