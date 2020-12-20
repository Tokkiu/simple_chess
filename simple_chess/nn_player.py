from __future__ import print_function
import numpy as np
import pickle
from nn_tools import *


class ModelLoader:
    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params

    def predictor(self, board, current_player, last_move):
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
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i + 1]))
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = softmax(X_p)
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value

    def current_state_collection(self, moves, players, current_player, last_move):
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


def load_model(model_path: str = './model/8_8_model'):
    policy_param = pickle.load(open(model_path, 'rb', ), encoding='bytes')
    width, height = 8, 8
    best_policy = ModelLoader(width, height, policy_param)
    return best_policy.predictor

model = load_model('../model/8_8_model')

def prefict_probs(next_state, current_player, last_move):
    moves, probs = [], []
    predicted, value = model(next_state, current_player, last_move)
    for move, prob in predicted:
        loc = state_to_move(move)
        moves.append((7 - loc[0], loc[1]))
        probs.append(prob)

    indexs = list(reversed(np.argsort(probs)))
    return np.array(moves)[indexs], np.array(probs)[indexs], value


if __name__ == '__main__':

    state = np.zeros((8,8))
    state[4][3] = 1
    state[4][4] = 1
    state[4][5] = 1
    state[5][3] = -1
    state[5][4] = -1
    moves, probs, value = prefict_probs(state, -1, move_to_state(4,5))

    print(moves, probs, value)

