import numpy as np
class Board(object):

    def __init__(self, state, n_in_row):

        self.state = state
        self.n_in_row = n_in_row
        self.availables = [(x,y)  for y in range(self.state.shape[1]) for x in range(self.state.shape[0]) if self.state[x][y] == 0]
        self.unavailables = [(x,y)  for y in range(self.state.shape[1]) for x in range(self.state.shape[0]) if self.state[x][y] != 0]
            

    def move(self, position, player):

        x,y = position
        if self.state[x,y] == 0:
            self.state[x,y] = player
            self.availables.remove((x,y))
            self.unavailables.append((x,y))
            return True
            
        else:
            return False


    def check_game_result(self):

        if(len(self.unavailables) < self.n_in_row + 2):
            return False, None

        state = self.state
        height = state.shape[0]
        width = state.shape[1]
        for chess in self.unavailables:
            row = chess[0]; col = chess[1]
            # Check in vertical
            if row <= height-self.n_in_row:
                if np.sum([state[row+i][col] for i in range(self.n_in_row)]) == self.n_in_row or np.sum([state[row+i][col] for i in range(self.n_in_row)]) == -1*self.n_in_row  :
                    return True, state[row][col]
            # Check in horizontal
            if col <= width-self.n_in_row:
                if np.sum([state[row][col+i] for i in range(self.n_in_row)]) == self.n_in_row or np.sum([state[row][col+i] for i in range(self.n_in_row)]) == -1*self.n_in_row:
                    return True, state[row][col]
            # Check in diagonal
            if (row <= height-self.n_in_row) and (col <= width-self.n_in_row):
                if np.sum([state[row+i][col+i] for i in range(self.n_in_row)]) == self.n_in_row or np.sum([state[row+i][col+i] for i in range(self.n_in_row)]) == -1*self.n_in_row:
                    return True, state[row][col]
            # Check in anti-diagonal
            if (row <= height-self.n_in_row) and (col >= self.n_in_row-1):
                if np.sum([state[row+i][col-i] for i in range(self.n_in_row)]) == self.n_in_row or np.sum([state[row+i][col-i] for i in range(self.n_in_row)]) == -1*self.n_in_row:
                    return True, state[row][col]

        # No one wins till no vacancy in state
        if len(self.availables) == 0:
            return True, None

        return False, None




