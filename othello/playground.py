import numpy as np

from algorithms.arena import Arena
from algorithms.search import MCTS
from algorithms.model import Model

from othello.model import NeuralNet
from othello.game import OthelloGame
from othello.train import args 
from othello.players import *

"""
use this script to play any two agents against each other, 
or play manually with any agent.
"""



mini_othello = True  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

n1 = Model(
        NeuralNet(g, args['model']['neuralnet']), 
        g, args['model']
        )
if mini_othello:
    n1.load_checkpoint('temp/','best.pth.tar')
else:
    assert 0
    n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')

mcts1 = MCTS(g, n1, args['self-play'])
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = Model(
        NeuralNet(g, args['model']['neuralnet']), 
        g, args['model']
        )
    if mini_othello:
        n2.load_checkpoint('temp/','best.pth.tar')
    else:
        assert 0
        n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')

    mcts2 = MCTS(g, n2, args['self-play'])
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    player2 = n2p  

arena = Arena(n1p, player2, g, display=OthelloGame.display)
print(arena.playGames(2, verbose=True))