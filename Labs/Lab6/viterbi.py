import csv
import logging
from collections import defaultdict
from math import log
from operator import add
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


class Viterbi:
    """
    Viterbi algorithm implementation for Hidden Markov Models. By using a dynamic programming strategy it keeps the best
    values for the previous time step and uses those to calculate the current one instead of calculating the whole path.
    """

    def __init__(self, train_data_path: str, word2pos_distribution_path: str):
        """
        Reads training formatted as N lines of "Word, POS". Consecutive lines are meant to be in the same sentence.
        A empty line in the data file indicates a new sentence. Stores the log-prior probabilities,
        log-transition probabilities and log-emission probabilities.
        :param train_data_path: Path for the training data.
        :param word2pos_distribution_path: Path for the word2pos distribution data.
        """
        pos_prev = None
        occurrences_counter = defaultdict(int)
        transitions_counter = defaultdict(lambda: defaultdict(int))

        with open(train_data_path, mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    # New sentence, reset previous
                    pos_prev = None
                else:
                    word, pos_cur = line.split()
                    occurrences_counter[pos_cur] += 1
                    if pos_prev is not None:  # Skip first word as we need a previous one
                        transitions_counter[pos_cur][pos_prev] += 1
                    pos_prev = pos_cur

        self.C = Viterbi._estimate_prior_probabilities(occurrences_counter)
        self.A = Viterbi._estimate_transition_probabilities(transitions_counter)
        self.B, self.pos_tags = self._estimate_emission_probabilities(word2pos_distribution_path)

    def predict(self, observations: List[str]) -> List[str]:
        """
        Predicts the POS for a given sentence.
        :param observations: List of words that form the sentence.
        :return: List of POS corresponding to <seq>.
        """
        pos_prediction = []

        # Prior probabilities for each POS
        # Emission probabilities for each POS from the first word
        p_current = list(map(lambda x: self.C[x], self.pos_tags))
        p_current = list(map(add, self.B[observations[0]], p_current))

        # Get maximum probability POS and its probability
        pos = self._get_maximum_probability_pos(p_current)
        pos_prediction.append(pos)

        # Calculated with t-1 probabilities
        for word in observations[1:]:
            p_previous = p_current
            p_current = []

            for i, pos_cur in enumerate(self.pos_tags):
                p_current_pos = []
                for j, pos_prev in enumerate(self.pos_tags):
                    # Previous prob. + Transition prob. + Emission prob.
                    p = p_previous[j] + self.A[pos_cur][pos_prev] + self.B[word][i]
                    p_current_pos.append(p)

                p_current.append(max(p_current_pos))

            # Get maximum probability POS and its probability
            pos = self._get_maximum_probability_pos(p_current)
            pos_prediction.append(pos)

        return pos_prediction

    def _get_maximum_probability_pos(self, p):
        max_idx = np.array(p).argmax()
        return self.pos_tags[max_idx]

    def _estimate_emission_probabilities(self, path: str) -> Tuple[Dict[str, List[float]], List[str]]:
        """
        Loads the distribution of probabilities for each the possible POS given a word and estimates the emission
        probabilities dividing it by the estimated prior probabilities.
        :param path: Path for the distribution data.
        :return: Emission probabilities as a map { word: [p(POS_0), ..., p(POS_N)] } and list of POS
        """
        with open(path, mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            # Skip first empty cell
            header = next(reader)[1:]
            B = defaultdict(lambda: [log(1 / len(header))] * len(header))

            for line in reader:
                word, logits = line[0], line[1:]
                for i in range(len(header)):
                    B[word][i] = float(logits[i]) - self.C[header[i]]

        logging.info(f'Estimated sparse {len(B)}x{len(header)} emission probabilities')
        return B, header

    @staticmethod
    def _estimate_prior_probabilities(d: Dict[str, int]) -> Dict[str, float]:
        """
        Calculates and stores the estimation of the prior probabilities for the given data.
        :param d: Dictionary mapping each POS to its number of occurrences.
        :return: Prior probabilities as a map { word: p(word) }
        """
        C = {}
        n = sum(d.values())

        for pos, occurrences in d.items():
            C[pos] = log(occurrences / n)

        logging.info(f'Estimated {len(d)} prior probabilities')
        return C

    @staticmethod
    def _estimate_transition_probabilities(d: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """
        Calculates and stores the sparse estimation of the transition probabilities for the given data.
        :param d: Dictionary mapping each POS to the number of occurrences of same following POS.
        :return: Transition probabilities as a map { POSi_t: p(POSj_t-1) }
        """
        A = defaultdict(lambda: defaultdict(lambda: -1e10))
        for pos_cur, pos_prev_counter in d.items():
            n = sum(pos_prev_counter.values())

            for pos_prev, occurrences in pos_prev_counter.items():
                A[pos_cur][pos_prev] = log(occurrences / n)

        logging.info(f'Estimated sparse {len(d)}x{len(d)} transition probabilities')
        return A


def evaluate_hmm(alg: Viterbi, path: str, print_output: bool):
    logging.info('Evaluating HMM...')
    sentences, sentences_pos = [], []
    sentence, sentence_pos = [], []
    with open(path, mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':  # Empty line refers to new sentence
                if sentence:
                    sentences.append(sentence)
                    sentences_pos.append(sentence_pos)
                    sentence, sentence_pos = [], []
            else:
                word, pos = line.split()
                sentence.append(word)
                sentence_pos.append(pos)
        # Add possible remaining last sentence
        if sentence:
            sentences.append(sentence)
            sentences_pos.append(sentence_pos)

        correct_pos, total_pos = 0, 0
        for idx in tqdm(range(len(sentences)), ncols=100):
            pos = alg.predict(sentences[idx])
            if pos is None:
                continue

            if print_output:
                print(f' '.join(sentences[idx]))
                print(' '.join(sentences_pos[idx]))
                print(' '.join(pos))
                print('-' * 150)

            correct_pos += sum(map(lambda x: x[0] == x[1], zip(pos, sentences_pos[idx])))
            total_pos += len(sentences_pos[idx])

    print(f'Tagged properly {correct_pos}/{total_pos} POS tags')
    print(f'Accuracy: {correct_pos / total_pos}')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    viterbi = Viterbi(train_data_path='train_pos.txt',
                      word2pos_distribution_path='tag_logit_per_word.tsv')

    evaluate_hmm(alg=viterbi, path='test_pos.txt', print_output=False)
