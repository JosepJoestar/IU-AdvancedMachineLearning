import csv
import logging
from collections import defaultdict
from math import log10
from operator import add
from typing import Dict, List, Tuple

import numpy as np

NEG_INF = -999999999999


class Viterbi:
    """
    Viterbi algorithm implementation for Hidden Markov Models. By using a dynamic programming strategy it keeps the best
    values for the previous time step and uses those to calculate the current one instead of calculating the whole path.
    """

    def __init__(self, train_data_path: str, word2pos_distribution_path: str):
        self._read_data(train_data_path, word2pos_distribution_path)

    def predict(self, seq: List[str]) -> Tuple[List[str], List[float]]:
        """
        Predicts the POS for a given sentence.
        :param seq: List of words that form the sentence.
        :return: List of POS corresponding to <seq>.
        """
        hidden_prob, hidden_pos = [], []

        # Use prior probabilities in first time step

        # Prior probabilities for each POS
        p_current = list(map(lambda x: self.prior_probabilities[x], self.pos_tags))
        # Emission probabilities for each POS from the first word
        p_current = list(map(add, self.emission_probabilities[seq[0]], p_current))

        # Get maximum probability POS and its probability
        prob, pos = self._get_maximum_probability_pos(p_current)
        hidden_prob.append(prob)
        hidden_pos.append(pos)
    
        # Calculated with t-1 probabilities
        for word in seq[1:]:
            p_previous = p_current
            p_current = []

            for i, pos_t in enumerate(self.pos_tags):
                # TODO: Remove this if
                if word not in self.emission_probabilities:
                    return None, None

                p_emi = self.emission_probabilities[word][i]
                p_current_pos = []

                for j, pos_t_1 in enumerate(self.pos_tags):
                    if pos_t_1 in self.transition_probabilities and \
                            pos_t in self.transition_probabilities[pos_t_1]:
                        p_trans = self.transition_probabilities[pos_t_1][pos_t]
                    else:
                        p_trans = NEG_INF  # We consider a big small number as log(0), for unseen events.
                    p = p_emi + p_trans + p_previous[j]
                    p_current_pos.append(p)

                p_current.append(max(p_current_pos))

            # Get maximum probability POS and its probability
            prob, pos = self._get_maximum_probability_pos(p_current)
            hidden_prob.append(prob)
            hidden_pos.append(pos)

        return hidden_pos, hidden_prob

    def _read_data(self, train_path: str, dist_path: str):
        """
        Reads training formatted as N lines of "Word, POS".
        Consecutive lines are meant to be in the same sentence.
        A empty line in the data file indicates a new sentence.
        Stores the log-prior, log-transition probabilities and log-emission probabilities.
        :param train_path: Path for the training data.
        :param dist_path: Path for the word2pos distribution data.
        """
        previous_tag = None
        occurrences_counter = defaultdict(int)
        transitions_counter = defaultdict(lambda: defaultdict(int))

        with open(train_path, mode='r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':  # Empty line refers to new sentence
                    previous_tag = None
                else:
                    word, tag = line.split()

                    occurrences_counter[tag] += 1
                    if previous_tag is not None:  # Skip first word as we need a previous one
                        transitions_counter[tag][previous_tag] += 1

                    previous_tag = tag

        self.prior_probabilities = Viterbi._estimate_prior_probabilities(occurrences_counter)
        self.transition_probabilities = Viterbi._estimate_transition_probabilities(transitions_counter)
        self.emission_probabilities, self.pos_tags = Viterbi._estimate_emission_probabilities(dist_path)

    def _get_maximum_probability_pos(self, p):
        max_idx = np.array(p).argmax()
        return p[max_idx], self.pos_tags[max_idx]

    @staticmethod
    def _estimate_prior_probabilities(counter: Dict[str, int], log=True) -> Dict[str, float]:
        """
        Calculates and stores the estimation of the prior probabilities for the given data.
        :param counter: Dictionary mapping each POS to its number of occurrences.
        :param log: Apply log scale to the calculated probabilities (default: True).
        :return: Prior probabilities as a map { word: p(word) }
        """
        prior_prob = {}

        n = sum(counter.values())
        for tag, occurrences in counter.items():
            p = occurrences / n
            prior_prob[tag] = log10(p) if log else p

        logging.info(f'Estimated {len(counter)} prior probabilities')
        return prior_prob

    @staticmethod
    def _estimate_transition_probabilities(counter: Dict[str, Dict[str, int]], log=True) -> Dict[str, Dict[str, float]]:
        """
        Calculates and stores the sparse estimation of the transition probabilities for the given data.
        :param counter: Dictionary mapping each POS to the number of occurrences of same following POS.
        :param log: Apply log scale to the calculated probabilities (default: True).
        :return: Transition probabilities as a map { POSi_t: p(POSj_t-1) }
        """
        transition_prob = {}

        for tag, next_tag_counter in counter.items():
            transition_prob[tag] = {}

            n = sum(next_tag_counter.values())
            for next_tag, occurrences in next_tag_counter.items():
                p = occurrences / n
                transition_prob[tag][next_tag] = log10(p) if log else p

        logging.info(f'Estimated sparse {len(counter)}x{len(counter)} transition probabilities')
        return transition_prob

    @staticmethod
    def _estimate_emission_probabilities(path: str) -> Tuple[Dict[str, List[float]], List[str]]:
        """
        Loads the distribution of probabilities for each the possible POS given a word and estimates the emission
        probabilities dividing it by the estimated prior probabilities.
        :param path: Path for the distribution data.
        :return: Emission probabilities as a map { word: [p(POS_0), ..., p(POS_N)] } and list of POS
        """
        word2pos_prob = {}
        with open(path, mode='r') as file:
            reader = csv.reader(file, delimiter='\t')
            header = next(reader)[1:]  # Skip first empty cell

            for line in reader:
                word = line[0]
                if word not in word2pos_prob:  # Skip already inserted word distributions
                    distribution = line[1:]
                    word2pos_prob[word] = list(map(float, distribution))

        logging.info(f'Estimated sparse {len(word2pos_prob)}x{len(header)} emission probabilities')
        return word2pos_prob, header


def evaluate_hmm(alg: Viterbi, path: str, print_output: bool):
    logging.info('Evaluating HMM...')
    sentences, sentences_pos = [], []
    sentence, sentence_pos = [], []
    with open(path, mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':  # Empty line refers to new sentence
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

        for idx in range(len(sentences)):
            pos, _ = alg.predict(sentences[idx])
            if pos is None:
                continue

            if print_output:
                print(' '.join(sentences[idx]))
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
