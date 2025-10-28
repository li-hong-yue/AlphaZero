import logging
import math
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        # (state, action)
        self.Qsa = defaultdict(float)  # stores Q values for s,a 
        self.Nsa = defaultdict(int)  # stores #times edge s,a was visited

        # state
        self.Ns = defaultdict(int)   # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Ends = {}  # stores game.getGameEnded ended for board s
        self.Valids = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        s = self.game.stringRepresentation(canonicalBoard)
        for _ in range(self.args['numMCTSSims']):
            self.search(canonicalBoard)
        counts = [self.Nsa[(s, a)] for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        # end game state
        if s not in self.Ends:
            self.Ends[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Ends[s] in [-1, 0, 1]: # terminal node
            return -self.Ends[s]

        # leaf node state
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            self.Valids[s] = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = (self.Ps[s] + 1e-4) * self.Valids[s]  # smoothing + masking invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])  # normalize
            self.Ns[s] = 0
            return -v


        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if self.Valids[s][a]:
                u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)
        
        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1

        return -v