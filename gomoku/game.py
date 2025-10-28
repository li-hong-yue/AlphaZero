import numpy as np

from algorithms.game import Game
 
class GomokuGame(Game):
    square_content = {
        -1: "X",
        +0: "-",
        +1: "O"
    }

    @staticmethod
    def getSquarePiece(x):
        return GomokuGame.square_content[x]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        return np.zeros((self.n, self.n), dtype=int)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n ** 2

    def getNextState(self, board, player, action):
        new_board = np.copy(board)
        row = action // self.n
        col = action % self.n
        new_board[row, col] = player
        return (new_board, -player)

    def getValidMoves(self, board, player=None):
        return np.array(board==0).flatten()
    
    def _has_five_in_a_row(self, board, val):
        """Return True if `val` has 5 in a row in any direction."""
        n = self.n
        for i in range(n):
            for j in range(n):
                if board[i, j] != val:
                    continue
                # horizontal
                if j <= n - 5 and np.all(board[i, j:j+5] == val):
                    return True
                # vertical
                if i <= n - 5 and np.all(board[i:i+5, j] == val):
                    return True
                # diagonal down-right
                if i <= n - 5 and j <= n - 5 and np.all([board[i+k, j+k] == val for k in range(5)]):
                    return True
                # diagonal up-right
                if i >= 4 and j <= n - 5 and np.all([board[i-k, j+k] == val for k in range(5)]):
                    return True
        return False

    def getGameEnded(self, board, player):
        """
        Return:
            0 if game not ended,
            1 if current player has won,
           -1 if current player has lost.
        board: np.array of shape (n, n) with values in {-1, 0, 1}
        player: +1 or -1 (whose perspective)
        """
        if self._has_five_in_a_row(board, 1):
            return 1 if player == 1 else -1
        if self._has_five_in_a_row(board, -1):
            return 1 if player == -1 else -1
        if np.sum(self.getValidMoves(board)) == 0:
            return 0
        return 'inplay'
        

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def getSymmetries(self, board, pi):
        # returns a list of (board, pi)
        # mirror, rotational
        assert(len(pi) == self.n**2) 
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()  # board is an np array

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(GomokuGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")