import random
from copy import deepcopy
import numpy as np
from Board import Board

from nn_player import load_model
from nn_player import Board as NBoard

model = load_model('../model/best_policy_8_8_5.model')

class MCTSNode(object):
    '''
    MCTSNode in the MCT
    '''

    def __init__(self, board, player, parent, position):
        self.board = board
        self.player = player
        self.parent = parent
        self.position = position

        self.children = []

        # Max children number
        self.max_expend_num = 5
        if len(board.availables) < self.max_expend_num:
            self.max_expend_num = len(board.availables)

        self.win_times = 0
        self.visited_times = 0

        self._untried_actions = None
        self._untried_far_actions = []


    def fully_expanded(self):
        '''
        Wether the node is fully expanded
        '''
        return len(self.children) == self.max_expend_num


    def get_best_child(self, score_method):
        '''
        Score all the children by the score method
        Return a best one
        '''
        if len(self.children) == 0:
            print('The node has no child!')
            return None
        children_scores = [(child, score_method(child)) for child in self.children]
        best_child = max(children_scores, key=lambda x: x[1])[0]
        return best_child


    def expand_child(self):
        '''
        If the node is not a fully expanded node, add a child.
        The child is a player node different from the parent, and it has a different board
        The child put a chess in a random position
        '''
        if len(self.children) < self.max_expend_num:
            next_state = np.copy(self.board.state)

            pos_list, prob_list, score = self.predict_probs(next_state)
            children = [tuple(child.position) for child in self.children]
            for pos in pos_list[:self.max_expend_num]:

                # Creating the child instance and add it into the children attribute
                if tuple(pos) not in children:
                    # Create board instance and take the move
                    state = Board(np.copy(next_state), self.board.n_in_row)
                    state.move(pos, self.player * -1)
                    new_child = MCTSNode(state, self.player * -1, self, pos)

                    self.children.append(new_child)
                    return new_child

            # pos = self.untried_actions.pop(0)
            # state = Board(np.copy(next_state), self.board.n_in_row)
            # state.move(pos, self.player * -1)
            # new_child = MCTSNode(state, self.player * -1, self, pos)
            #
            # self.children.append(new_child)
            # return new_child

        return None



    def predict_probs(self, next_state):
        me = 1
        current_player = 1 if self.player == me else 2

        last_move = -1
        if self.parent:
            last_pos = self.parent.position
            last_move = move_to_state(7 - last_pos[0], last_pos[1])

        moves, probs = [], []
        predicted, value = model(next_state, current_player, last_move)
        for move, prob in predicted:
            loc = state_to_move(move)
            moves.append((7-loc[0], loc[1]))
            probs.append(prob)

        indexs = list(reversed(np.argsort(probs)))
        return np.array(moves)[indexs], np.array(probs)[indexs], value

def state_to_move(state=0, width=8):
    h = state // width
    w = state % width
    return h, w


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

