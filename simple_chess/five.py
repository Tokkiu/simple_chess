import pygame
import numpy as np
import time
from MCTS import MCTSAgent

from numba import jit

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

def draw_board(screen,background):
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
    screen.blit(background, (0, 0))
    for h in range(0, M):
        pygame.draw.line(screen, WHITE, [40, h * d + 40], [600, 40 + h * d], 1)
        pygame.draw.line(screen, WHITE, [40 + d * h, 40], [40 + d * h, 600], 1)
    #画星位
    circle_center = [
        (120,120),
        (120,520),
        (520,120),
        (520,520),
    ]
    for cc in circle_center:
        pygame.draw.circle(screen, WHITE, cc, 5)


def draw_stone(screen,mat,whitestone,blackstone):
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
                pos = [20+d * j, 20+d * i]
                screen.blit(blackstone,pos)
                #pygame.draw.circle(screen, black_color, pos, 18, 0)
            elif mat[i][j] == -1:
                pos = [20+d * j, 20+ d * i]
                screen.blit(whitestone,pos)
                #pygame.draw.circle(screen, white_color, pos, 18, 0)



def render(screen, mat,whitestone,blackstone,background):
    """
    Draw the updated game with lines and stones using function draw_board and draw_stone
    input:
        screen: game window, onto which the stones are drawn
        mat: 2D matrix representing the game state
    output:
        none
    """

    draw_board(screen,background)
    draw_stone(screen, mat,whitestone,blackstone)
    pygame.display.update()


@jit(nopython=True)

def check_for_done(mat):
    """
    please write your own code testing if the game is over. Return a boolean variable done. If one of the players wins
    or the tie happens, return True. Otherwise return False. Print a message about the result of the game.
    input:
        2D matrix representing the state of the game
    output:
        (done=bool,winner=1 for black -1 for white and 0 for tie,None for continue)
    """
    done = False
    for i in range(M-4):
        for j in range(M):
            temp = np.sum(mat[j,i:i+5])
            if temp == 5:
                print("win 1")
                winner=1
                return True,winner
            elif temp == -5:
                print("win -1")
                winner=-1
                return True,winner

            temp = np.sum(mat[i:i+5,j])
            if temp == 5:
                print("win 1")
                winner=1
                return True,winner
            elif temp == -5:
                print("win -1")
                winner=-1
                return True,winner

            if j+4<M:
                temp = mat[i,j] + mat[i+1,j+1] + mat[i+2,j+2] + mat[i+3,j+3] + mat[i+4,j+4]
                if temp == 5:
                    print("win 1")
                    winner=1
                    return True,winner
                elif temp == -5:
                    print("win -1")
                    winner=-1
                    return True,winner
            if j+4<M:
                temp = mat[M-i-1, j] + mat[M-i-2, j + 1] + mat[M-i-3, j + 2] + mat[M-i-4, j + 3] + mat[M-i-5, j + 4]
                if temp == 5:
                    print("win 1")
                    winner=1
                    return True,winner
                elif temp == -5:
                    print("win -1")
                    winner=-1
                    return True,winner

    if not (mat == 0).any():
        print("tie")
        winner=0
        return True,winner

    return done,None



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

def show_go_screen(surf,winner,startground,start_rect):
    note_height = 10
    surf.blit(startground, start_rect)
    draw_text(surf, 'Simple Five in a row', 60, 640 // 2, note_height + 640 // 4, WHITE)
    draw_text(surf, 'Press any key to start', 22, 640 // 2, note_height + 640 // 2,WHITE)
    pygame.display.update()
    waiting = True
    clock = pygame.time.Clock()
    while waiting:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                 waiting = False

def show_end_screen(surf, winner):
    y = 8
    if winner is not None:
        if winner==1:
            draw_text(surf, 'You {0} !'.format('win!' ),
                  64, 640 // 2, y, RED)
            draw_text(surf, 'it is so simple !', 64, 640 // 2, y + 640 // 4, WHITE)
        elif winner==-1:
            draw_text(surf, 'You {0} !'.format('lose!'),
                      64, 640 // 2, y, RED)
            draw_text(surf, 'it is not that simple !', 64, 640 // 2, y + 640 // 4, WHITE)
        elif winner==0:
            draw_text(surf, 'You {0} !'.format('get a tie!'),
                      64, 640 // 2, y, RED)
            draw_text(surf, 'Try again !', 64, 640 // 2, y + 640 // 4, WHITE)

    pygame.display.update()
    waiting = True
    clock = pygame.time.Clock()
    while waiting:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYUP:
                waiting = False



def draw_text(surf, text, size, x, y, color=WHITE):
    font_name = pygame.font.get_default_font()
    font = pygame.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)

def main():
    global M
    M = 4
    #px
    WIDTH =640
    HEIGHT=640
    FPS = 30
    ##px
    pygame.init()
    # 初始化mixer （因为下文我们需要用到音乐）
    pygame.mixer.init()


    background_img = pygame.image.load('board.png')  # 棋盘背景图
    startground_img= pygame.image.load('background.png')  # 开始页面背景图2
    whitestone_image = pygame.image.load('whitestone1.png')
    blackstone_image = pygame.image.load('blackstone1.png')
    # icon = pygame.image.load('icon.png')

    # 加载各种资源
    hit_sound = pygame.mixer.Sound('boo.wav')
    back_music = pygame.mixer.music.load('background.mp3')
    pygame.mixer.music.set_volume(0.4)
    all_sprites = pygame.sprite.Group()

    background = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
    startground = pygame.transform.scale(startground_img, (WIDTH, HEIGHT))
    whitestone= pygame.transform.scale(whitestone_image, (40, 40))
    blackstone= pygame.transform.scale(blackstone_image, (40, 40))
    back_rect = background.get_rect()
    start_rect=startground.get_rect()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Simple Five-in-a-Row')
    # pygame.display.set_icon(icon)
    clock = pygame.time.Clock()
    clock.tick(FPS)
    done = False
    mat = np.zeros((M, M))
    d = int(560 / (M - 1))
    done = False
    pygame.mixer.music.play(loops=-1)
    winner=None
    show_go_screen(screen, winner, startground, start_rect)
    render(screen, mat, whitestone, blackstone, background)
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                hit_sound.play()
                (x, y) = event.pos
                row = round((y - 40) / d)
                col = round((x - 40) / d)
                mat[row][col] = 1
                render(screen, mat,whitestone,blackstone,background)
                # check for win or tie
                # print message if game finished
                # otherwise contibue
                check=check_for_done(mat)
                winner=check[1]
                if check[0]:
                    done=True
                    break


                # get the next move from computer/MCTS
                # check for win or tie
                # print message if game finished
                # otherwise contibue

                mat = update_by_pc(mat, row, col)
                render(screen, mat,whitestone,blackstone,background)
                check = check_for_done(mat)
                winner=check[1]
                if check[0]:
                    done=True

    show_end_screen(screen, winner)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

if __name__ == '__main__':
    main()

