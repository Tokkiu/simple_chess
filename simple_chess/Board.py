import numpy as np
class Board(object):

    def __init__(self, state, n_in_row):
        '''
        state: store the current chess state
        availables: read the available sites in the type of tuple (x,y)
        '''
        self.state = state
        self.n_in_row = n_in_row
        self.availables = [(i,j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i][j] == 0]
        self.unavailables = [(i,j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i][j] != 0]


    def move(self, position, player):
        '''
        input: position, player
        output: if move successfully, change the self.state,self.availables and return True;
                else, return False
        '''
        x,y = position
        if self.state[x,y] != 0:
            return False
        else:
            self.state[x,y] = player
            self.availables.remove((x,y))
            self.unavailables.append((x,y))
            return True


    def check_game_result(self):
        '''
        Check the game result

        output: is_over: bool
                winner: if no winner, return None, else return winner
        '''
        # unavailables = [(i,j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i][j] != 0]
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




