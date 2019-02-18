import argparse
import logging

def main(pretrained_graph_path, dataset_path):
    from model import FacialRecognition
    from dataloader import load_inception_graph
    
    load_inception_graph(pretrained_graph_path)

    model = FacialRecognition(dataset_path, 'test_set.csv')
    model.train()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_path', nargs=1)
    parser.add_argument('dataset_path', nargs=1)
    parser.add_argument('--v', action='store_true')
    parser.add_argument('--vv', action='store_true')
    args = parser.parse_args()
    
    if args.vv:
        print('Super Verbose Mode enabled')
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.v:
        print('Verbose Mode enabled')
        logging.getLogger().setLevel(logging.INFO)
        

    main(args.graph_path[0], args.dataset_path[0])