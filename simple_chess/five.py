import pygame
import numpy as np
import time
from MCTS import MCTSAgent

from numba import jit


def draw_board(screen):
    """
    This function draws the board with lines.
    input: game windows
    output: none
    """
    global M
    M = 8
    d = int(560 / (M - 1))
    black_color = [0, 0, 0]
    board_color = [241, 196, 15]
    screen.fill(board_color)
    for h in range(0, M):
        pygame.draw.line(screen, black_color, [40, h * d + 40], [600, 40 + h * d], 1)
        pygame.draw.line(screen, black_color, [40 + d * h, 40], [40 + d * h, 600], 1)


def draw_stone(screen, mat):
    """
    This functions draws the stones according to the mat. It draws a black circle for matrix element 1(human),
    it draws a white circle for matrix element -1 (computer)
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """
    black_color = [0, 0, 0]
    white_color = [255, 255, 255]
    M = len(mat)
    d = int(560 / (M - 1))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] == 1:
                pos = [40 + d * j, 40 + d * i]
                pygame.draw.circle(screen, black_color, pos, 18, 0)
            elif mat[i][j] == -1:
                pos = [40 + d * j, 40 + d * i]
                pygame.draw.circle(screen, white_color, pos, 18, 0)


def render(screen, mat):
    """
    Draw the updated game with lines and stones using function draw_board and draw_stone
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """

    draw_board(screen)
    draw_stone(screen, mat)
    pygame.display.update()

@jit(nopython=True)
def check_for_done(mat):
    """
    please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
    or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
    input:
        2D matrix representing the state of the game
    output:
        none
    """
    done = False
    for i in range(M-4):
        for j in range(M):
            temp = np.sum(mat[j,i:i+5])
            if temp == 5:
                return (True, 1)
            elif temp == -5:
                return (True, -1)

            temp = np.sum(mat[i:i+5,j])
            if temp == 5:
                return (True, 1)
            elif temp == -5:
                return (True, -1)

            if j+4<M:
                temp = mat[i,j] + mat[i+1,j+1] + mat[i+2,j+2] + mat[i+3,j+3] + mat[i+4,j+4]
                if temp == 5:
                    return (True, 1)
                elif temp == -5:
                    return (True, -1)
            if j+4<M:
                temp = mat[M-i-1, j] + mat[M-i-2, j + 1] + mat[M-i-3, j + 2] + mat[M-i-4, j + 3] + mat[M-i-5, j + 4]
                if temp == 5:
                    return (True, 1)
                elif temp == -5:
                    return (True, -1)

    if not (mat == 0).any():
        return (True, 0)

    return done, None



def update_by_pc(mat, row, col):
    """
    This is the core of the game. Write your code to give the computer the intelligence to play a Five-in-a-Row game
    with a human
    input:
        2D matrix representing the state of the game.
    output:
        2D matrix representing the updated state of the game.
    """
    availables = np.where(mat==0)
    if len(availables[0])==1:
        mat[availables[0][0]][availables[1][0]] = -1
    else:
        mcts = MCTSAgent(mat, 10, 10000)
        position = mcts.choose_position(-1, (row,col))
        mat[position[0]][position[1]] = -1

    return mat


def main():
    global M
    M = 8

    pygame.init()
    screen = pygame.display.set_mode((640, 640))
    pygame.display.set_caption('Five-in-a-Row')
    done = False
    mat = np.zeros((M, M))
    d = int(560 / (M - 1))
    draw_board(screen)
    pygame.display.update()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                (x, y) = event.pos
                row = round((y - 40) / d)
                col = round((x - 40) / d)
                mat[row][col] = 1
                render(screen, mat)
                # check for win or tie
                # print message if game finished
                # otherwise contibue

                print(check_for_done(mat))
                is_over, winner =check_for_done(mat)
                if is_over:
                    done = True
                    break

                # get the next move from computer/MCTS
                # check for win or tie
                # print message if game finished
                # otherwise contibue

                mat = update_by_pc(mat, row, col)
                render(screen, mat)

                is_over, winner = check_for_done(mat)
                if is_over:
                    done = True
                    break

    print("Winner is :", winner)

    pygame.quit()


if __name__ == '__main__':
    main()

