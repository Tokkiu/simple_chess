import numpy as np
import time
import random

from copy import deepcopy

from MCTSNode import MCTSNode
from Board import Board

def pprint_tree(node, file=None, _prefix="", _last=True, level = 0, max_depth=1):
    print(_prefix, "`- " if _last else "|- ", {"Win Num":node.win_times, "Visit Num":node.visited_times,  "Pos":node.position}, sep="", file=file)
    _prefix += "   " if _last else "|  "
    child_count = len(node.children)
    if level >= max_depth:
        return
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last, level+1)


class MCTSAgent(object):
    '''
    Use MCTS to choose a position
    '''

    def __init__(self, board, max_decision_time, max_simulation_times):
        self.board =  Board(board, 5) # Chess board
        self.max_decision_time = max_decision_time
        self.max_simulation_times = max_simulation_times

        self.simulation_times = 0 # the times of simulation
        self.begin_time = time.time() # start the MCTS time
        self.time_limit = 10
        self.temperature = 0.5

    
    def choose_position(self, last_player, last_position):
        '''
        Choose a position
        '''
        # One availables position left in the board
        if len(self.board.availables) == 1:
            return self.board.availables[0]
        root = MCTSNode(deepcopy(self.board), last_player, None, last_position)  # root is the current state of board
        position = self.monte_carlo_tree_search(root).position
        return position
        

    def monte_carlo_tree_search(self, root):
        '''
        MCTS process
        '''
        time_mcts = time.time()
        # When it is within the number of time
        while time.time() - time_mcts < self.time_limit:

            # Step 1: Selection 
            leaf = self.traverse(root)  # leaf is unvisited node

            # Step 2: Expansion
            if not self.is_terminal(leaf):
                leaf_child = leaf.expand_child()

            # Step 3: Roll out
            simulation_result = self.rollout(leaf_child)

            # Step 4: Backpropagateion
            self.backpropagate(leaf_child, simulation_result)


        # Gettin the total simulation count
        self.simulation_times = root.visited_times
        # Print tree for debugging
        pprint_tree(root)

        print('Simulation number: {}'.format(self.simulation_times))
        print('Simulation time: {}'.format(time.time()-time_mcts))
        return root.get_best_child(self.uct)
    

    def backpropagate(self, node, result):
        '''
        Back propagate the result. 
        Update win times and visited times
        '''
        while node is not None:
            if node.player == result:
                node.win_times += 1
            node.visited_times += 1
            node = node.parent


    def rollout(self, node):
        '''
        Simulate a game up to game overf
        '''

        current_state = deepcopy(node.board)
        player = node.player
        is_over, winner = current_state.check_game_result()

        while not is_over:
            player = -1*player
            position = self.rollout_policy(current_state)
            #print(position,type(position))
            current_state.move(position, player)
            is_over, winner = current_state.check_game_result()
        return winner


    def is_terminal(self, node):
        '''
        Wether the node is terminal, which means someone win or no availables position.
        '''
        is_terminal, _ = node.board.check_game_result()
        return is_terminal


    def rollout_policy(self, board):
        '''
        Random positions.
        '''
        return random.choice(board.availables)


    def traverse(self, node):
        '''
        Traverse all the nodes and find the node that is not fully expeanded.

        (For the traverse function, to avoid using up too much time or resources, you may start considering only 
        a subset of children (e.g 10 children). Increase this number or by choosing this subset smartly later.)
        '''
        while not self.is_terminal(node):
            if not node.fully_expanded():
                return node
            else:
                node = node.get_best_child(self.uct)
        return node


    def uct(self, node):
        '''
        Score function
        '''
        return (node.win_times/node.visited_times) + self.temperature*np.sqrt(np.log(node.parent.visited_times)/node.visited_times)
