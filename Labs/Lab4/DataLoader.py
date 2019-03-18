import re
import csv
import logging
import unicodedata
import numpy as np

from Vocabulary import Voc

BUFFER_LENGTH = 50000


class Loader:
    def __init__(self, datafile, preprocess=False, prune=0):
        """
        Loads data into a VOC object

        :param datafile: Path to where is the dataset file located
        :param preprocess: Apply normalization, ascii... to the data
        :param prune: Nth more frequent words to keep (0 if keep all)
        """
        self.datafile = datafile
        self.preprocess = preprocess
        self.prune = prune

        # Create VOC from data
        self.voc = self._load_data()

        # Initialize negative samples random buffer
        self.random_buffer = self._generate_random_buffer(first=True)
        self.index_buffer = 0

    # Batching Methods

    def next_batch(self, n_contexts, context_size, k):
        batch_size = n_contexts * (2 * context_size + k)
        logging.debug('Generating epoch batches (batch_size ~= {})...'.format(batch_size))

        i, x_central_batch, x_samples_batch, y_batch = 0, [], [], []

        total_contexts = len(self.voc.context)
        while i < total_contexts:
            word_pos_text, word, word_id = self.voc.context[i]
            i += 1

            #  Positive samples
            context_words_ids = self._get_context_words(word_pos_text, context_size)
            j, total_context_words = 0, len(context_words_ids)
            while j < total_context_words:
                x_central_batch.append(word_id)
                x_samples_batch.append(context_words_ids[j])
                y_batch.append(1.0)
                j += 1

            #  Negative Samples
            negative_words_ids = self._get_negative_samples(k, word)
            j, total_negative_words = 0, len(negative_words_ids)
            while j < total_negative_words:
                x_central_batch.append(word_id)
                x_samples_batch.append(negative_words_ids[j])
                y_batch.append(0.0)
                j += 1

            # Yield batches when size >= BATCH_SIZE
            if len(y_batch) >= batch_size:
                yield i, x_central_batch, x_samples_batch, y_batch
                x_central_batch, x_samples_batch, y_batch = [], [], []

        yield i, x_central_batch, x_samples_batch, y_batch

    def _get_context_words(self, pos, cs):
        begin = max(0, pos - cs)
        end = min(pos + cs + 1, len(self.voc.text))
        return [self.voc.word2index[cw]
                for cw in self.voc.text[begin:pos] + self.voc.text[pos + 1:end]
                if cw in self.voc.word2index]

    def _get_negative_samples(self, k, c):
        if self.index_buffer + k > 50000:
            self.random_buffer += self._generate_random_buffer()
            self.index_buffer = 0

        k_negative_samples = self.random_buffer[self.index_buffer:self.index_buffer + k]
        self.index_buffer += k

        return [self.voc.word2index[nw] for nw in k_negative_samples if nw != c]

    def _generate_random_buffer(self, first=False):
        if first:
            unigram = self.voc.frequencies / self.voc.frequencies_sum
            modified_unigram = np.power(unigram, 0.75)
            self.modified_unigram_weighs = modified_unigram / sum(modified_unigram)
        return np.random.choice(self.voc.words, BUFFER_LENGTH, p=self.modified_unigram_weighs).tolist()

    # MetaData Methods

    def generate_metadata_projector(self):
        with open('metadata.tsv', mode='w', newline='') as f:
            tsv_output = csv.writer(f, delimiter='\t')
            for i in list(self.voc.word2index.keys()):
                tsv_output.writerow([i])

    # Loader Methods

    def _load_data(self):
        """Populates and returns a VOC object"""
        logging.info('Start preparing training data...')
        data = self._read_data()
        voc = Voc()

        logging.info('Counting words...')
        for line in data:
            voc.add_sentence(line)

        logging.info('Counted words: {}'.format(voc.num_words))

        if self.prune > 0:
            logging.info('Pruning Vocabulary...')
            voc.prune(self.prune)
            logging.info('Counted words (after prune): {}'.format(voc.num_words))

        logging.info('Storing words frequencies...')
        voc.store_words_freqs()

        logging.info('Storing word contexts...')
        voc.store_central_context()

        return voc

    def _read_data(self):
        """Reads data from the given file and returns it splitted up in lines"""
        with open(self.datafile, mode='r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            if self.preprocess:
                lines = [Loader._normalize_string(s) for s in lines]
            return lines

    @staticmethod
    def _normalize_string(s):
        """Lowercase, trim, and remove non-letter characters"""
        s = Loader._unicode_to_ascii(s.lower().strip())
        s = re.sub(r'([.!?])', r' \1', s)
        s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
        s = re.sub(r'\s+', r' ', s).strip()
        return s

    @staticmethod
    def _unicode_to_ascii(s):
        """Turn a Unicode string to plain ASCII"""
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')
