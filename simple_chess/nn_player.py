from __future__ import print_function
import numpy as np
import pickle
from nn_tools import *


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class PolicyValueNetNumpy:
    """policy-value network in numpy """

    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params

    def policy_value_fn(self, board, current_player, last_move):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions, moves, players = [], [], []

        for i in range(len(board)):
            for j in range(len(board)):
                state = move_to_state(7-i, j)
                er = board[i][j]
                if er == 0:
                    legal_positions.append(state)
                else:
                    moves.append(state)
                    players.append(1 if er == 1 else 2)

        current_state = self.current_state_collection(moves, players, current_player, last_move)

        X = current_state.reshape(-1, 4, 8, 8)
        # first 3 conv layers with ReLu nonlinearity
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i + 1]))
        # policy head
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = softmax(X_p)
        # value head
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value

    def current_state_collection(self, moves, players, current_player, last_move):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        moves = np.array(moves)
        players = np.array(players)
        square_state = np.zeros((4, 8, 8))
        if len(moves) != 0:
            move_curr = moves[players == current_player]
            move_oppo = moves[players != current_player]
            square_state[0][move_curr // self.board_width,
                            move_curr % self.board_height] = 1.0
            square_state[1][move_oppo // self.board_width,
                            move_oppo % self.board_height] = 1.0
            # indicate the last move location
            square_state[2][last_move // self.board_width,
                            last_move % self.board_height] = 1.0
        if len(moves) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]




class Game(object):

    def __init__(self, board, **kwargs):
        self.board = board

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)



def move_to_state(x=0, y=0, width=8):
    '''
    For a 8x8 board game, start from 0,0 to 7,7
    7,0 ~ ~ ~ ~ 7,7
    ~ ~ ~ ~ ~ ~ ~
    0,0 ~ ~ ~ ~ 0,7
    :param x:
    :param y:
    :return: compressed state
    '''

    return x * width + y


def state_to_move(state=0, width=8):
    h = state // width
    w = state % width
    return h, w


def load_model(model_path: str = './model/best_policy_8_8_5.model'):
    policy_param = pickle.load(open(model_path, 'rb', ), encoding='bytes')
    width, height = 8, 8
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    return best_policy.policy_value_fn


if __name__ == '__main__':
    model = load_model()
    # mcts_player = build_nn_player()

    board = Board()
    board.init_board(0)

    # my first move
    board.do_move(move_to_state(4, 5))
    board.do_move(move_to_state(5, 6))
    board.do_move(move_to_state(1, 0))
    board.do_move(move_to_state(2, 1))

    for action, prob in model(board)[0]:
        print(action, prob)


    # AI's next move
    action = mcts_player.get_action(board)
    print(state_to_move(action))

    # graphic current board
    board.do_move(action)
