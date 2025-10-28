import logging
from tqdm import tqdm

log = logging.getLogger(__name__)
 

class Arena():
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it. 
            Is necessary for verbose mode.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one  game.
        Returns: 1 if player1 won, -1 if player2 won, 0 if draw
        """
        players = {
            1: self.player1, 
            -1: self.player2,
        }
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players.values():
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) not in [-1,0,1]:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players.values():
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num=20, verbose=False):
        """
        Plays num games in which 
        player1 starts num/2 games and 
        player2 starts num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        assert num % 2 == 0
        num = num // 2
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc=f"player 1 starts {num} games"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc=f"player 2 starts {num} games"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
        
        self.player1, self.player2 = self.player2, self.player1
        
        return oneWon, twoWon, draws