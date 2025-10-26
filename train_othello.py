import logging
import coloredlogs
import torch

from algorithms.coach import Coach
from algorithms.model import Model

from othello.model import NeuralNet
from othello.game import OthelloGame as Game

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = {
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),

    'self-play': {
        'numIters': 1000,
        'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,        #
        'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
        'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint': './temp/',
        'numItersForTrainExamplesHistory': 20,
    },
    
    'model': {
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'neuralnet': {
            'num_channels': 512,
            'dropout': 0.3,
        }
    }
    
}


def main():
    log.info('Loading %s...', Game.__name__)
    game = Game(6)

    log.info('Loading %s...', Model.__name__)
    model = Model(
        NeuralNet(game, args['model']['neuralnet']), 
        game, args['model']
        )

    if args['load_model']:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        model.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    coach = Coach(game, model, args['self-play'])

    if args['load_model']:
        log.info("Loading 'trainExamples' from file...")
        coach.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    coach.learn()


if __name__ == "__main__":
    main()