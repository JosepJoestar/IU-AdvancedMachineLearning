import argparse
import logging

from DataLoader import Loader
from Word2Vec import Word2Vec

N = 5000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', action='store_true')
    parser.add_argument('--vv', action='store_true')
    args = parser.parse_args()

    if args.vv:
        print('Super Verbose Mode enabled')
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.v:
        print('Verbose Mode enabled')
        logging.getLogger().setLevel(logging.INFO)

    data = Loader('wikipedia_sample_tiny.txt', prune=N)
    data.generate_metadata_projector()

    model = Word2Vec(data)
    model.train()
