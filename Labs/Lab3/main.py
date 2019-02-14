import os
import argparse
import logging

def main(pretrained_graph_path, dataset_path, testset_path):
    from model import FacialRecognition
    from dataloader import load_inception_graph
    
    load_inception_graph(pretrained_graph_path)

    model = FacialRecognition(dataset_path, testset_path)
    model.train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_path', nargs=1)
    parser.add_argument('dataset_path', nargs=1)
    parser.add_argument('--v', action='store_true')
    parser.add_argument('--vv', action='store_true')
    parser.add_argument('--test', default='test_set.csv')
    args = parser.parse_args()
    
    if args.vv:
        print('Super Verbose Mode enabled')
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.v:
        print('Verbose Mode enabled')
        logging.getLogger().setLevel(logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        

    main(args.graph_path[0], args.dataset_path[0], args.test)