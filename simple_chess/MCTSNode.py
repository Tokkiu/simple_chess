import random
from copy import deepcopy
import numpy as np
from Board import Board

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
    
    # Should be replaced by NN
    @property
    def untried_actions(self):
        if self._untried_actions is None:
            # pattern_children = list((set(self.board.unavailables)|set(self.find_naive_pattern()))-set(self.board.unavailables))
            pattern_children = self.find_naive_pattern()
            nearest_children = list(set(self.find_nearest_position_first()) - set(pattern_children))
            small_board_children = list(set(self.small_board_strategy())-set(pattern_children)-set(nearest_children))
            available_children = list(set(deepcopy(self.board.availables)) - set(pattern_children) - set(nearest_children) \
                -set(small_board_children))
            self._untried_actions = pattern_children + nearest_children + small_board_children + available_children
            # available_children = deepcopy(self.board.availables)
            # self._untried_actions = available_children

        return self._expand_child
        

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
            move_pos = self.untried_actions.pop(0)

            # Create board instance and take the move
            state = Board(next_state, self.board.n_in_row)
            state.move(move_pos, self.player * -1)

            # Creating the child instance and add it into the children attribute
            new_child = MCTSNode(state, self.player * -1, self, move_pos)
            self.children.append(new_child)
            return new_child
        else:
            return None
    

    def find_naive_pattern(self):
        '''
        Find the pattern in this state
        Choose the best position in the availables
        '''
        children_list = self.board.find_position_by_pattern()
        return children_list


    def small_board_strategy(self):
        '''
        Randomly select the center position
        '''
        unavailables_num = len(self.board.unavailables)
        s_len = int((1/2)*(unavailables_num/2)+7/2)
        # print(s_len)
        if s_len >= 8:
            # Small board size equal to the original board size
            return []
        boundary_gap = int((8-s_len)/2)
        small_board_position = [(boundary_gap+i, boundary_gap+j) for i in range(s_len) for j in range(s_len) \
            if self.board.state[boundary_gap+i][boundary_gap+j] == 0]
        sb_len = len(small_board_position)
        return random.sample(small_board_position, int(sb_len/2))
        

    def find_nearest_position_first(self):
        '''
        The method find nearest position sets
        '''

        nearest_positions = set() # create a set
        h, w = self.board.state.shape[0], self.board.state.shape[1]
        unavailables = self.board.unavailables
        for i, j in unavailables:
            # up down right left
            if i < h - 1:
                nearest_positions.add((i+1, j))
            if i > 0:
                nearest_positions.add((i-1, j))
            if j < w - 1:
                nearest_positions.add((i, j+1))
            if j > 0:
                nearest_positions.add((i, j-1))
            # diag
            if i < h - 1 and j < w - 1:
                nearest_positions.add((i+1, j+1))
            if i > 0 and j < w -1:
                nearest_positions.add((i-1, j+1))
            if i < h -1 and j > 0:
                nearest_positions.add((i+1, j-1))
            if i > 0 and j > 0:
                nearest_positions.add((i-1, j-1))
        # remove unavailables in nearest position
        nearest_positions = list(set(nearest_positions) - set(unavailables))
        return nearest_positions