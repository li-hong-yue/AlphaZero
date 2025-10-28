import numpy as np

from algorithms.arena import Arena
from algorithms.search import MCTS
from algorithms.model import Model

from gomoku.model import NeuralNet
from gomoku.game import GomokuGame
from gomoku.train import args 
from gomoku.players import *

"""
use this script to play any two agents against each other, 
or play manually with any agent.
"""

human_vs_cpu = False


g = GomokuGame(5)

# all players
rp = RandomPlayer(g).play

n1p = RandomPlayer(g).play

player2 = RandomPlayer(g).play

'''

n1 = Model(
        NeuralNet(g, args['model']['neuralnet']), 
        g, args['model']
        )

n1.load_checkpoint('temp/','best.pth.tar')

mcts1 = MCTS(g, n1, args['self-play'])
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = Model(
        NeuralNet(g, args['model']['neuralnet']), 
        g, args['model']
        )

    n2.load_checkpoint('temp/','best.pth.tar')

    mcts2 = MCTS(g, n2, args['self-play'])
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    player2 = n2p  
'''
arena = Arena(n1p, player2, g, display=GomokuGame.display)
print(arena.playGames(2, verbose=True))