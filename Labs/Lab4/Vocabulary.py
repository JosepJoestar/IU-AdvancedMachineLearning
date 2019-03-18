import numpy as np

from collections import Counter


class Voc:
    def __init__(self):
        self._initialize_data()

    def _initialize_data(self, text=True):
        if text:
            self.text = []
        self.num_words = 0
        self.word2count = Counter()
        self.word2index = {}

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word, count=1, text=True):
        if text:
            self.text.append(word)
        self.word2count.update({word: count})
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.num_words += 1

    def prune(self, nth_keep):
        keep_words = self.word2count.most_common(nth_keep)
        self._initialize_data(text=False)
        for word, count in keep_words:
            self.add_word(word, count, text=False)

    def store_words_freqs(self):
        voc_freqs = self.word2count.most_common()
        self.words, frequencies = zip(*voc_freqs)
        self.frequencies = np.array(frequencies)
        self.frequencies_sum = sum(self.frequencies)

    def store_central_context(self):
        self.context = []
        for i, word in enumerate(self.text):
            if word in self.word2index:
                word_id = self.word2index[word]
                self.context.append((i, word, word_id))
