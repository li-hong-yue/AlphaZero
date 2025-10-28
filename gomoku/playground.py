import numpy as np

from algorithms.arena import Arena
from algorithms.search import MCTS
from algorithms.model import Model

from gomoku.model import NeuralNet
from gomoku.game import GomokuGame
from gomoku.train import args 
from gomoku.players import RandomPlayer, HumanPlayer

"""
use this script to play any two agents against each other, 
or play manually with any agent.
"""
human_vs_cpu = False


g = GomokuGame(6)

n1 = Model(NeuralNet(g, args['model']['neuralnet']), g, args['model'])
n1.load_checkpoint('temp/','best.pth.tar')

mcts1 = MCTS(g, n1, args['self-play'])
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

player2 = RandomPlayer(g).play

if human_vs_cpu:
    player2 = HumanPlayer(g).play
 

arena = Arena(n1p, player2, g, display=GomokuGame.display)
print(arena.playGames(2, verbose=True))